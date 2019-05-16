import argparse
import datetime
import gzip
import math
import os
import pickle
import random
import re
import sys
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


C_PUCT = 0.1


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


class Node:
    def __init__(
            self,
            parent,
            p: float,
            repl: REPL,
            ptra: ProofTraceActions,
            theorem: Thm,
    ) -> None:
        self._parent = parent
        self._children = []
        self._expanded = False
        self._theorem = theorem

        self._P = p
        self._N = 0
        self._W = 0.0
        self._Q = 0.0

        self._repl = repl
        self._ptra = ptra

    def is_leaf(
            self,
    ) -> bool:
        return not self._expanded

    def update_value(
            self,
            value: float,
    ) -> None:
        self._W += value
        self._Q = self._W / self._N if self._N > 0 else 0.0

    def update_visit(
            self,
    ) -> None:
        self._N += 1

    def expand(
            self,
            beta_width: int,
            sequence_length: int,
            model: Model,
            target: Thm,
    ) -> float:
        actions = self._ptra.actions().copy()
        arguments = self._ptra.arguments().copy()
        idx = len(actions)

        actions.append(Action.from_action('EXTRACT', None, None))

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < sequence_length:
            actions.append(empty)
        while len(arguments) < sequence_length:
            arguments.append(empty)

        prd_actions, prd_lefts, prd_rights, prd_values = \
            model.infer([idx], [actions], [arguments])

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions[0].cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts[0].cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights[0].cpu()).topk(beta_width)
        value = prd_values[0].item()

        candidates = []

        for ia in range(a_count):
            for il in range(beta_width):
                for ir in range(beta_width):

                    action = top_actions[1][ia].item()
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()

                    if left >= self._ptra.len() or right >= self._ptra.len():
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action + len(PREPARE_TOKENS)],
                        self._ptra.arguments()[left],
                        self._ptra.arguments()[right],
                    )

                    if self._ptra.seen(a):
                        continue

                    if not self._repl.valid(a):
                        continue

                    candidates.append((a,
                                       top_actions[0][ia].item() *
                                       top_lefts[0][il].item() *
                                       top_rights[0][ir].item()))

        for action, p in candidates:
            repl = self._repl.copy()
            thm = repl.apply(action)

            ptra = self._ptra.copy()

            action._index = thm.index()
            argument = ptra.build_argument(
                thm.concl(), thm.hyp(), thm.index(),
            )

            if target.thm_string(True) == thm.thm_string(True):
                Log.out("DEMONSTRATED")
                sys.exit(0)

            ptra.append(action, argument)

            self._children.append(Node(
                self,
                p,
                repl,
                ptra,
                thm,
            ))

        self._expanded = True

        Log.out("EXPAND", {
            'value': value,
            'summary': self._ptra.summary(),
            # 'theorem': self._theorem.thm_string(True),
        })

        return value

    def select(
            self,
    ):
        assert len(self._children) > 0

        total = 0
        for n in self._children:
            total += n._N

        scores = []
        for n in self._children:
            score = n._Q + C_PUCT * n._P * math.sqrt(total) / (1 + n._N)
            scores.append(score)

        m = max(scores)
        for i in range(len(scores)):
            if scores[i] == m:
                return self._children[i]

    @staticmethod
    def run(
            beta_width: int,
            sequence_length: int,
            model: Model,
            target: Thm,
            tree,
    ) -> None:
        node = tree

        while node is not None and node._expanded is True:
            child = node.select()
            node.update_visit()
            node = child

        if node is not None:
            value = node.expand(beta_width, sequence_length, model, target)

            n = node
            while n is not None:
                n.update_value(value)
                n = n._parent
        else:
            Log.out("EMPTY")


def mcts():
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

        if ptra_len <= 64:
            cases.append((p, ptra_len))

    Log.out(
        "Loaded ProofTraceActions", {
            'max_length': 64,
            'cases': len(cases),
        })

    model = Model(config).load()

    # cases = sorted(cases, key=lambda c: c[1])

    for i in range(len(cases)):
        c = cases[i][0]
        with gzip.open(c, 'rb') as f:
            ground = pickle.load(f)

        Log.out("TARGET", {
            'name': ground.name(),
            'prepare_length': ground.prepare_len(),
            'length': ground.action_len(),
            'summary': ground.summary(),
        })

        ptra = ProofTraceActions(
            'MCTS-{}-{}'.format(
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

        fixed_gamma = 8
        if fixed_gamma > 0:
            gamma_len = ground.action_len() - fixed_gamma

            for i in range(gamma_len):
                assert ground.prepare_len() + i < ground.len() - 1
                pos = ground.prepare_len() + i

                action = ground.actions()[pos]
                argument = ground.arguments()[pos]

                thm = repl.apply(action)

                action._index = thm.index()
                argument._index = thm.index()

                ptra.append(action, argument)

        tree = Node(None, 1.0, repl, ptra, target)

        for i in range(64):
            Node.run(
                config.get('prooftrace_tree_search_beta_width'),
                config.get('prooftrace_sequence_length'),
                model,
                target,
                tree,
            )
