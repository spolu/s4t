import argparse
import datetime
import os
import pickle
import random
import re
import torch
import typing

from dataset.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions, ProofTraceTokenizer

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.lstm import H

from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm

from utils.config import Config
from utils.log import Log

_MAX_VALUE = 10e9


class Model:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))
        self._load_dir = config.get('prooftrace_load_dir')

        with open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._model_E = E(self._config).to(self._device)
        self._model_H = H(self._config).to(self._device)
        self._model_PH = PH(self._config).to(self._device)
        self._model_VH = VH(self._config).to(self._device)

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

    def infer(
            self,
            trc: typing.List[typing.List[Action]],
            idx: typing.List[typing.List[int]],
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            embeds = self._model_E(trc)
            hiddens = self._model_H(embeds)

            head = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                embeds[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            prd_actions, prd_lefts, prd_rights = \
                self._model_PH(head, targets)
            prd_values = self._model_VH(head, targets)

            return (
                prd_actions, prd_lefts, prd_rights,
                prd_values,
            )


class Node:
    def __init__(
            self,
            config: Config,
            parent,
            model: Model,
            ground: ProofTraceActions,
            target: Thm,
            ptra: ProofTraceActions,
            repl: REPL,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            value: float,
    ):
        self._config = config

        self._parent = parent

        self._model = model
        self._ground = ground
        self._target = target
        self._ptra = ptra
        self._repl = repl

        self._sequence_length = config.get('prooftrace_sequence_length')
        self._beta_width = config.get('prooftrace_lm_search_beta_width')

        a_count = min(
            self._beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions).topk(a_count)
        top_lefts = torch.exp(prd_lefts).topk(self._beta_width)
        top_rights = torch.exp(prd_rights).topk(self._beta_width)

        actions = []

        for ia in range(a_count):
            for il in range(self._beta_width):
                for ir in range(self._beta_width):

                    action = top_actions[1][ia].item() + len(PREPARE_TOKENS)
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()

                    if left >= self._ptra.len() or right >= self._ptra.len():
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action],
                        self._ptra.actions()[left],
                        self._ptra.actions()[right],
                    )

                    if self._ptra.seen(a):
                        continue

                    if not self._repl.valid(a):
                        continue

                    actions.append(a)

        if len(actions) > 0:
            trc = []
            idx = []
            for a in actions:
                pre_trc, pre_idx = \
                    Node.prepare(self._ptra, a, self._sequence_length)
                trc.append(pre_trc)
                idx.append(pre_idx)

            prd_actions, prd_lefts, prd_rights, prd_values = \
                self._model.infer(trc, idx)

            self._queue = sorted(
                [(
                    actions[i],
                    prd_actions[i].to(torch.device('cpu')),
                    prd_lefts[i].to(torch.device('cpu')),
                    prd_rights[i].to(torch.device('cpu')),
                    prd_values[i].item(),
                ) for i in range(len(actions))],
                key=lambda t: t[4],
            )
            self._min_value = self.queue_value()
        else:
            self._queue = []
            self._min_value = _MAX_VALUE

        self._children = []

    def min_value(
            self,
    ) -> float:
        return self._min_value

    def child_value(
            self,
            c,
    ) -> float:
        return c.min_value() + 0.2 * len(c._children)

    def queue_value(
            self,
    ) -> float:
        return self._queue[0][4]

    def children_value(
            self,
    ) -> float:
        return self.child_value(self._children[0])

    def update(
            self,
    ) -> None:
        self._children = sorted(
            self._children, key=lambda c: self.child_value(c)
        )

        if len(self._children) > 0 and len(self._queue) > 0:
            self._min_value = min(
                self.children_value(),
                self.queue_value(),
            )
        elif len(self._children) > 0 and len(self._queue) == 0:
            self._min_value = self.children_value()
        elif len(self._children) == 0 and len(self._queue) > 0:
            self._min_value = self.queue_value()
        else:
            self._min_value = _MAX_VALUE

        if self._parent is not None:
            self._parent.update()

    def expand_queue(
            self,
    ) -> Thm:
        candidate = self._queue[0]
        self._queue = self._queue[1:]

        a = candidate[0].copy()

        repl = self._repl.copy()
        thm = repl.apply(a)
        a._index = thm.index()

        ptra = self._ptra.copy()
        ptra.append(a)

        node = Node(
            self._config,
            self,
            self._model,
            self._ground,
            self._target,
            ptra,
            repl,
            candidate[1],
            candidate[2],
            candidate[3],
            candidate[4],
        )

        self._children.append(node)
        node.update()

        Log.out('EXPAND', {
            'ground_length': self._ground.action_len(),
            'ptra_length': ptra.action_len(),
            'value': candidate[4],
            'summary': ptra.summary(),
        })

        if thm.thm_string() == self._target.thm_string():
            return thm
        else:
            return None

    def expand_children(
            self,
    ) -> Thm:
        return self._children[0].expand()

    def expand(
            self,
    ) -> Thm:
        """ Expand the min_value leaf node.

        At this point the tree is up to date so we follow the path of max
        value and go expand that leaf node.
        """
        if len(self._children) > 0 and len(self._queue) > 0:
            if self._children[0].min_value() < self._queue[0][4]:
                return self.expand_children()
            else:
                return self.expand_queue()
        elif len(self._children) > 0 and len(self._queue) == 0:
            return self.expand_children()
        elif len(self._children) == 0 and len(self._queue) > 0:
            return self.expand_queue()
        else:
            assert False

    @staticmethod
    def prepare(
            ptra: ProofTraceActions,
            a: Action,
            sequence_length: int,
    ) -> typing.Tuple[
        typing.List[Action],
        typing.List[int],
    ]:
        trc = ptra.actions().copy()
        idx = len(trc)
        if a is not None:
            trc.append(a)
            idx += 1

        trc.append(Action.from_action('EXTRACT', None, None))
        empty = Action.from_action('EMPTY', None, None)
        while len(trc) < sequence_length:
            trc.append(empty)

        return trc, idx

    @staticmethod
    def bootstrap(
            config: Config,
            tokenizer: ProofTraceTokenizer,
            model: Model,
            ground: ProofTraceActions,
            target: Thm,
    ):
        ptra = ProofTraceActions(
            'TREE-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                a for a in ground.actions()
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
        repl = REPL(tokenizer)
        repl.prepare(ptra)

        pre_trc, pre_idx = \
            Node.prepare(ptra, None, config.get('prooftrace_sequence_length'))
        trc = [pre_trc]
        idx = [pre_idx]

        prd_actions, prd_lefts, prd_rights, prd_values = \
            model.infer(trc, idx)

        return Node(
            config,
            None,
            model,
            ground,
            target,
            ptra,
            repl,
            prd_actions[0].to(torch.device('cpu')),
            prd_lefts[0].to(torch.device('cpu')),
            prd_rights[0].to(torch.device('cpu')),
            prd_values[0].item(),
        )


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

    with open(
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
        with open(c, 'rb') as f:
            ground = pickle.load(f)

        repl = REPL(tokenizer)
        repl.prepare(ground)
        target = repl.replay(ground)

        Log.out("TARGET", {
            'name': ground.name(),
            'prepare_length': ground.prepare_len(),
            'length': ground.action_len(),
            'summary': ground.summary(),
        })

        tree = Node.bootstrap(config, tokenizer, model, ground, target)

        done = False
        while(not done):
            if tree.min_value() is _MAX_VALUE:
                Log.out("FAILED", {
                    'name': ground.name(),
                })
                done = True
            else:
                thm = tree.expand()
                if thm:
                    Log.out("DEMONSTRATED", {
                        'name': ground.name(),
                        'theorem': thm.thm_string(),
                    })
                    done = True
