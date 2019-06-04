import math
import torch
import typing

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search_base import Search, SearchModel

from utils.config import Config
from utils.log import Log


C_PUCT = 10.0


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
        self._N = 1
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
            offset: int,
            model: SearchModel,
            target: Thm,
    ) -> typing.Tuple[
        float, ProofTraceActions, bool,
    ]:
        actions = self._ptra.actions().copy()
        arguments = self._ptra.arguments().copy()

        index = len(actions)

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < sequence_length:
            actions.append(empty)
        while len(arguments) < sequence_length:
            arguments.append(empty)

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights, prd_values = \
                model.infer([index], [actions], [arguments])

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions[0].cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts[0].cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights[0].cpu()).topk(beta_width)
        value = prd_values[0].item() * 2.0 - 1.0

        candidates = []

        # Log.out("EXPAND", {
        #     'value': "{:.3f}".format(value),
        #     'length': self._ptra.len(),
        #     'summary': self._ptra.summary(offset),
        #     # 'theorem': self._theorem.thm_string(True),
        # })

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

                    candidates.append((
                        top_actions[0][ia].item() *
                        top_lefts[0][il].item() *
                        top_rights[0][ir].item(),
                        a
                    ))

        for p, action in candidates:
            repl = self._repl.copy()
            ptra = self._ptra.copy()

            thm = repl.apply(action)
            action._index = thm.index()

            argument = ptra.build_argument(
                thm.concl(), thm.hyp(), thm.index(),
            )
            ptra.append(action, argument)

            if target.thm_string(True) == thm.thm_string(True):
                Log.out("DEMONSTRATED", {
                    'theorem': thm.thm_string(True),
                    'summary': ptra.summary(offset),
                })
                return value, True, ptra

            self._children.append(Node(
                self,
                p,
                repl,
                ptra,
                thm,
            ))

        self._expanded = True

        return value, False, self._ptra

    def select(
            self,
    ):
        if len(self._children) == 0:
            return None

        total = 0
        for n in self._children:
            total += n._N

        scores = []
        for n in self._children:
            score = n._Q + C_PUCT * n._P * math.sqrt(total) / (1 + n._N)
            scores.append(score)

        # if self._parent is None:
        #     import pdb; pdb.set_trace()
        m = max(scores)
        for i in range(len(scores)):
            if scores[i] == m:
                return self._children[i]


class MCTS(Search):
    def __init__(
            self,
            config: Config,
            model: SearchModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(MCTS, self).__init__(config, model, ptra, repl, target)

        self._beta_width = config.get('prooftrace_search_mcts_beta_width')
        self._sequence_length = config.get('prooftrace_sequence_length')

        self._tree = Node(None, 1.0, repl, ptra, target)

    def step(
            self,
            final: bool = False,
            offset: int = 0,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        node = self._tree

        while node is not None and node._expanded is True:
            child = node.select()
            if child is None:
                return True, node._ptra, False
            node.update_visit()
            node = child

        if node is not None:
            value, proved, ptra = node.expand(
                self._beta_width,
                self._sequence_length,
                offset,
                self._model,
                self._target,
            )
            if proved:
                return True, ptra, True

            while node is not None:
                node.update_value(value)
                node = node._parent
        else:
            assert False

        if final:
            return True, ptra, False
        else:
            return False, ptra, False
