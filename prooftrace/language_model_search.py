import argparse
import copy
import datetime
import os
import pickle
import random
import re
import torch
import typing

from dataset.prooftrace import \
    ACTION_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions, ProofTraceTokenizer

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.lstm import H

from prooftrace.repl.repl import REPL

from utils.config import Config
from utils.log import Log


class Beam:
    def __init__(
            self,
            tokenizer: ProofTraceTokenizer,
            ground: ProofTraceActions,
    ):
        self._tokenizer = tokenizer
        self._ground = ground

        self._ptra = ProofTraceActions(
            'BEAM-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                a for a in self._ground.actions()
                if a.value in [
                        ACTION_TOKENS['TARGET'],
                        ACTION_TOKENS['EMPTY'],
                        ACTION_TOKENS['PREMISE'],
                        ACTION_TOKENS['SUBST'],
                        ACTION_TOKENS['SUBST_TYPE'],
                        ACTION_TOKENS['TERM'],
                ]
            ],
        )
        self._repl = REPL(tokenizer)

        self._match_count = 0
        self._value = 0.0

    def match_count(
            self,
    ) -> int:
        return self._match_count

    def ptra(
            self,
    ) -> ProofTraceActions:
        return self._ptra

    def repl(
            self,
    ) -> REPL:
        return self._repl

    def value(
            self,
    ) -> float:
        return self._value

    def apply(
            self,
            action: Action,
    ) -> None:
        if self._ground.seen(action) and not self._ptra.seen(action):
            self._match_count += 1
        self._repl.apply(action)
        self._ptra.append(action)


class BeamSearch:
    def __init__(
            self,
            config: Config,
            ground: ProofTraceActions,
            beam_count: int,
    ):
        self._config = config
        self._beam_count = beam_count

        self._device = torch.device(config.get('device'))
        self._load_dir = config.get('prooftrace_load_dir')
        self._sequence_length = config.get('prooftrace_sequence_length')
        self._beta_width = config.get('prooftrace_lm_search_beta_width')

        self._ground = ground
        Log.out('SEARCH', {
            'ground_length': self._ground.len(),
        })

        with open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._beams = [
            Beam(self._tokenizer, self._ground)
            for _ in range(self._beam_count)
        ]

        self._model_E = E(self._config).to(self._device)
        self._model_H = H(self._config).to(self._device)
        self._model_PH = PH(self._config).to(self._device)
        self._model_VH = VH(self._config).to(self._device)

    def beams(
            self,
    ) -> typing.List[Beam]:
        return self._beams

    def load(
            self,
    ):
        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_H_0.pt"
            ):
                Log.out(
                    "Loading prooftrace", {
                        'load_dir': self._load_dir,
                    })
                self._model_E.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E_0.pt",
                        map_location=self._device,
                    ),
                )
                self._model_H.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_H_0.pt",
                        map_location=self._device,
                    ),
                )
                self._model_PH.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_PH_0.pt",
                        map_location=self._device,
                    ),
                )
                self._model_VH.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_VH_0.pt",
                        map_location=self._device,
                    ),
                )

                self._model_E.eval()
                self._model_H.eval()
                self._model_PH.eval()
                self._model_VH.eval()

        return self

    def beta_explore(
            self,
            beam: Beam,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
    ) -> typing.List[typing.List[int]]:
        top_actions = torch.exp(prd_actions).topk(self._beta_width)
        top_lefts = torch.exp(prd_lefts).topk(self._beta_width)
        top_rights = torch.exp(prd_rights).topk(self._beta_width)

        actions = []

        for ia in range(self._beta_width):
            for il in range(self._beta_width):
                for ir in range(self._beta_width):

                    action = top_actions[1][ia].item()
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()

                    if left >= beam.ptra().len() or right >= beam.ptra().len():
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action],
                        beam.ptra().actions()[left],
                        beam.ptra().actions()[right],
                    )

                    if beam.ptra().seen(a):
                        continue

                    if not beam.repl().valid(a):
                        continue

                    actions.append([action, left, right])

        return actions

    def step_beam(
            self,
            beam: Beam,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
    ) -> typing.List[Beam]:
        out = []

        valids = self.beta_explore(
            beam,
            prd_actions,
            prd_lefts,
            prd_rights,
        )

        trc = [beam.ptra().actions().copy() for _ in valids]
        idx = [len(beam.ptra().actions()) for _ in valids]
        out = [copy.deepcopy(beam) for _ in valids]

        for i in range(len(trc)):
            a = Action.from_action(
                INV_ACTION_TOKENS[valids[i][0]],
                beam.ptra().actions()[valids[i][1]],
                beam.ptra().actions()[valids[i][2]],
            )
            trc[i].append(a)
            out[i].apply(a)
            idx[i] += 1
            trc[i].append(Action.from_action('EXTRACT', None, None))
            while len(trc[i]) < self._sequence_length:
                trc[i].append(Action.from_action('EMPTY', None, None))

        # Log.out("BEGIN VALUES", {'valid_count': len(valids)})
        if len(valids) > 0:
            with torch.no_grad():
                embeds = self._model_E(trc)
                hiddens = self._model_H(embeds)

                head = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    hiddens[i][0].unsqueeze(0) for i in range(len(idx))
                ], dim=0)

                prd_values = self._model_VH(head, targets)
        # Log.out("END VALUES")

        for i in range(len(valids)):
            out[i]._value = prd_values[i].item()

        return out

    def step(
            self,
    ) -> None:
        trc = [b.ptra().actions().copy() for b in self._beams]
        idx = [len(b.ptra().actions()) for b in self._beams]

        empty = Action.from_action('EMPTY', None, None)
        extract = Action.from_action('EXTRACT', None, None)
        for i in range(len(trc)):
            trc[i].append(extract)
            while len(trc[i]) < self._sequence_length:
                trc[i].append(empty)

        # Log.out("BEGIN ACTIONS")
        with torch.no_grad():
            embeds = self._model_E(trc)
            hiddens = self._model_H(embeds)

            head = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                hiddens[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            prd_actions, prd_lefts, prd_rights = \
                self._model_PH(head, targets)
        # Log.out("END ACTIONS")

        beams = []

        for i in range(len(self._beams)):
            beams += self.step_beam(
                self._beams[i],
                prd_actions[i],
                prd_lefts[i],
                prd_rights[i],
            )

        self._beams = sorted(
            beams, key=lambda b: b.value(), reverse=True,
        )[0:self._beam_count]

        if len(self._beams) > 0:
            Log.out('STEP', {
                'best_value': self._beams[0].value(),
                'match_count': self._beams[0].match_count(),
            })
        else:
            Log.out('NO_VALID')

        return len(self._beams)


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

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    dataset_dir = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        'test_traces'
    )

    assert os.path.isdir(dataset_dir)
    files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    cases = []

    for p in files:
        match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
        if match is None:
            continue
        ptra_len = int(match.group(1))

        if ptra_len <= 64:
            cases.append(p)

    Log.out(
        "Loaded ProofTraceActions", {
            'max_length': 64,
            'cases': len(cases),
        })

    for i in range(len(cases)):
        c = cases[i]
        with open(c, 'rb') as f:
            ptra = pickle.load(f)

        bs = BeamSearch(config, ptra, 16).load()
        beam_count = len(bs.beams())
        while beam_count > 0:
            beam_count = bs.step()
