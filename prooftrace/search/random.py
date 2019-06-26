import random
import typing
import torch

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.model import Model
from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search.search import Search

from utils.config import Config


NON_PREPARE_TOKENS = {
    k: v
    for k, v in ACTION_TOKENS.items()
    if k not in PREPARE_TOKENS
}


class Random(Search):
    def __init__(
            self,
            config: Config,
            model: Model,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(Random, self).__init__(config, model, ptra, repl, target)

        self._ptra = ptra.copy()
        self._repl = repl.copy()
        self._last_thm = None

        self._term_indices = [0] + [
            i for i in range(self._ptra.len())
            if self._ptra.actions()[i].value == 7
        ]
        self._subst_indices = [0] + [
            i for i in range(self._ptra.len())
            if self._ptra.actions()[i].value == 4
        ]
        self._subst_type_indices = [0] + [
            i for i in range(self._ptra.len())
            if self._ptra.actions()[i].value == 5
        ]
        self._premises_indices = [
            i for i in range(1, self._ptra.len())
            if self._ptra.actions()[i].value == 2
        ]

    def sample_term(
            self,
    ):
        return random.choice(self._term_indices)

    def sample_theorem(
            self,
    ):
        indices = self._premises_indices + \
            list(range(self._ptra.prepare_len(), self._ptra.len()))
        m = torch.distributions.Categorical(
            torch.softmax(torch.tensor(
                list(range(len(indices))),
                dtype=torch.float,
            ), 0),
        )
        return indices[m.sample().item()]

    def sample_subst(
            self,
    ):
        return random.choice(self._subst_indices)

    def sample_subst_type(
            self,
    ):
        return random.choice(self._subst_type_indices)

    def step(
            self,
            offset: int = 0,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        index, actions, arguments = self.preprocess_ptra(self._ptra)

        tries = 32
        candidate = None

        for i in range(tries):
            action = random.choice(list(NON_PREPARE_TOKENS.values()))

            if INV_ACTION_TOKENS[action] == 'REFL':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'TRANS':
                left = self.sample_theorem()
                right = self.sample_theorem()
            if INV_ACTION_TOKENS[action] == 'MK_COMB':
                left = self.sample_theorem()
                right = self.sample_theorem()
            if INV_ACTION_TOKENS[action] == 'ABS':
                left = self.sample_theorem()
                right = self.sample_term()
            if INV_ACTION_TOKENS[action] == 'BETA':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'ASSUME':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'EQ_MP':
                left = self.sample_theorem()
                right = self.sample_theorem()
            if INV_ACTION_TOKENS[action] == 'DEDUCT_ANTISYM_RULE':
                left = self.sample_theorem()
                right = self.sample_theorem()
            if INV_ACTION_TOKENS[action] == 'INST':
                left = self.sample_theorem()
                right = self.sample_subst()
            if INV_ACTION_TOKENS[action] == 'INST_TYPE':
                left = self.sample_theorem()
                right = self.sample_subst_type()

            a = Action.from_action(
                INV_ACTION_TOKENS[action],
                self._ptra.arguments()[left],
                self._ptra.arguments()[right],
            )

            if self._ptra.seen(a):
                continue

            if not self._repl.valid(a):
                continue

            candidate = a
            break

        if candidate is not None:
            thm = self._repl.apply(candidate)
            candidate._index = thm.index()
            argument = self._ptra.build_argument(
                thm.concl(), thm.hyp(), thm.index(),
            )
            self._ptra.append(candidate, argument)

            self._last_thm = thm
        else:
            return True, self._ptra, False

        if self._target.thm_string(True) == thm.thm_string(True):
            return True, self._ptra, True

        return False, self._ptra, False

    def last_thm(
            self,
    ) -> typing.Optional[Thm]:
        return self._last_thm
