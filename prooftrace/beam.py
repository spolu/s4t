import argparse
import datetime
import gzip
import os
import pickle
import random
import re
import torch
import typing

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, INV_PREPARE_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm

from utils.config import Config
from utils.log import Log


class Model:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._load_dir = config.get('prooftrace_load_dir')

        self._modules = {
            'E': E(self._config).to(self._device),
            'T': T(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

    def load(
            self,
    ):
        if self._load_dir:
            Log.out(
                "Loading prooftrace LM", {
                    'load_dir': self._load_dir,
                })
            if os.path.isfile(self._load_dir + "/model_E.pt"):
                self._modules['E'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_T.pt"):
                self._modules['T'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_T.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_PH.pt"):
                self._modules['PH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_PH.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_VH.pt"):
                self._modules['VH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_VH.pt",
                        map_location=self._device,
                    ),
                )

        self._modules['E'].eval()
        self._modules['T'].eval()
        self._modules['PH'].eval()
        self._modules['VH'].eval()

        return self

    def infer(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            action_embeds = self._modules['E'](act)
            argument_embeds = self._modules['E'](arg)

            hiddens = self._modules['T'](action_embeds, argument_embeds)

            heads = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            prd_actions, prd_lefts, prd_rights = \
                self._modules['PH'](heads, hiddens, targets)
            prd_values = self._modules['VH'](heads, targets)

            return (
                prd_actions, prd_lefts, prd_rights,
                prd_values,
            )


class Head:
    def __init__(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            value: float,
    ) -> None:
        self._prd_actions = prd_actions
        self._prd_lefts = prd_lefts
        self._prd_rights = prd_rights
        self._value = value

    def apply(
            self,
            ptra: ProofTraceActions,
            repl: REPL,
            beta_width: int,
            head_width: int,
    ) -> typing.List[
        typing.Tuple[float, Action],
    ]:
        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(self._prd_actions.cpu()).topk(a_count)
        top_lefts = torch.exp(self._prd_lefts.cpu()).topk(beta_width)
        top_rights = torch.exp(self._prd_rights.cpu()).topk(beta_width)

        candidates = []

        for ia in range(a_count):
            for il in range(beta_width):
                for ir in range(beta_width):

                    action = top_actions[1][ia].item()
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()

                    if left >= ptra.len() or right >= ptra.len():
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action + len(PREPARE_TOKENS)],
                        ptra.arguments()[left],
                        ptra.arguments()[right],
                    )

                    if ptra.seen(a):
                        continue

                    if not repl.valid(a):
                        continue

                    candidates.append((top_actions[0][ia].item() *
                                       top_lefts[0][il].item() *
                                       top_rights[0][ir].item(),
                                       a))

        return sorted(
            candidates, key=lambda c: c[0], reverse=True
        )[:head_width]


class Beam:
    def __init__(
            self,
            config: Config,
            model: Model,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        self._config = config

        self._width = config.get('prooftrace_beam_width')

        self._model = model
        self._target = target

        index, actions, arguments = self.process_ptra(ptra)
        prd_actions, prd_lefts, prd_rights, prd_values = \
            self._model.infer([index], [actions], [arguments])

        self._ptras = [ptra.copy() for _ in range(self._width)]
        self._repls = [repl.copy() for _ in range(self._width)]
        self._heads = [
            Head(
                prd_actions[0].cpu(),
                prd_lefts[0].cpu(),
                prd_rights[0].cpu(),
                prd_values[0].cpu().item(),
            )
        ] * self._width

    def process_ptra(
            self,
            ptra: ProofTraceActions,
    ) -> typing.Tuple[
        int, typing.List[Action], typing.List[Action],
    ]:
        actions = ptra.actions().copy()
        arguments = ptra.arguments().copy()

        index = len(actions)

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._config.get('prooftrace_sequence_length'):
            actions.append(empty)
        while len(arguments) < self._config.get('prooftrace_sequence_length'):
            arguments.append(empty)

        return index, actions, arguments

    def step(
            self,
            offset: int = 0,
    ) -> typing.Optional[ProofTraceActions]:
        idx = []
        act = []
        arg = []

        candidates = []

        for i in range(self._width):
            for p, action in self._heads[i].apply(
                self._ptras[i],
                self._repls[i],
                self._config.get('prooftrace_beam_beta_width'),
                self._config.get('prooftrace_beam_head_width'),
            ):
                repl = self._repls[i].copy()
                ptra = self._ptras[i].copy()

                thm = repl.apply(action)
                action._index = thm.index()
                argument = ptra.build_argument(
                    thm.concl(), thm.hyp(), thm.index(),
                )
                ptra.append(action, argument)

                if self._target.thm_string(True) == thm.thm_string(True):
                    Log.out("DEMONSTRATED")
                    return ptra

                candidates.append((ptra, repl, action))
                index, actions, arguments = self.process_ptra(ptra)

                idx.append(index)
                act.append(actions)
                arg.append(arguments)

        h = {}
        uniques = []
        for c in candidates:
            if c[2].hash() not in h:
                h[c[2].hash()] = True
                uniques.append(c)

        candidates = uniques

        Log.out("PRE-BEAM", {
            'candidates': len(candidates),
        })

        prd_actions, prd_lefts, prd_rights, prd_values = \
            self._model.infer(idx, act, arg)

        next_heads = []
        for i in range(len(candidates)):
            next_heads.append((
                candidates[i][0],
                candidates[i][1],
                Head(
                    prd_actions[i].cpu(),
                    prd_lefts[i].cpu(),
                    prd_rights[i].cpu(),
                    prd_values[i].cpu().item(),
                ),
                prd_values[i].cpu().item(),
            ))

        next_heads = sorted(
            next_heads, key=lambda v: v[3], reverse=True
        )[0:self._width]

        assert len(next_heads) == self._width

        self._ptras = [v[0] for v in next_heads]
        self._repls = [v[1] for v in next_heads]
        self._heads = [v[2] for v in next_heads]

        for v in next_heads:
            Log.out("BEAM", {
                'value': v[3],
                'summary': v[0].summary(offset),
            })

        return None


def search():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )

    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )

    dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        'train_traces'
    )

    assert os.path.isdir(dataset_dir)
    files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    cases = []

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    for p in files:
        match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
        if match is None:
            continue
        ptra_len = int(match.group(1))
        cases.append((p, ptra_len))

    Log.out(
        "Loaded ProofTraceActions", {
            'cases': len(cases),
        })

    model = Model(config).load()

    cases = sorted(cases, key=lambda c: c[1])

    for i in range(len(cases)):
        c = cases[i][0]
        with gzip.open(c, 'rb') as f:
            ground = pickle.load(f)

        ptra = ProofTraceActions(
            'BEAM-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                ground.actions()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
            [
                ground.arguments()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
        )
        repl = REPL(tokenizer)
        target = repl.prepare(ptra)

        offset = 0
        fixed_gamma = 4
        if fixed_gamma > 0:
            gamma_len = max(ground.action_len() - fixed_gamma, 0)
            offset = ground.prepare_len() + gamma_len

            for i in range(gamma_len):
                assert ground.prepare_len() + i < ground.len() - 1
                pos = ground.prepare_len() + i

                action = ground.actions()[pos]
                argument = ground.arguments()[pos]

                thm = repl.apply(action)

                action._index = thm.index()
                argument._index = thm.index()

                ptra.append(action, argument)

        Log.out("TARGET", {
            'name': ground.name(),
            'prepare_length': ground.prepare_len(),
            'length': ground.action_len(),
            'summary': ground.summary(offset),
        })

        beam = Beam(config, model, ptra, repl, target)

        for i in range(fixed_gamma * 2):
            proof = beam.step(offset)
            if proof is not None:
                break
