import numpy
import random
import typing

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
CONCLUSION_TOKENS = {
    'TRANS': 9,
    'MK_COMB': 10,
    'EQ_MP': 14,
    'DEDUCT_ANTISYM_RULE': 15,
    'INST': 16,
    'INST_TYPE': 17,
}


class RandomSampler:
    def __init__(
            self,
            ptra: ProofTraceActions,
    ) -> None:
        self._term_indices = [0] + [
            i for i in range(ptra.len())
            if ptra.actions()[i].value == 7
        ]
        self._subst_indices = [0] + [
            i for i in range(ptra.len())
            if ptra.actions()[i].value == 4
        ]
        self._subst_type_indices = [0] + [
            i for i in range(ptra.len())
            if ptra.actions()[i].value == 5
        ]
        self._premises_indices = [
            i for i in range(1, ptra.len())
            if ptra.actions()[i].value == 2
        ]

    def sample_term(
            self,
    ):
        return random.choice(self._term_indices)

    def sample_theorem(
            self,
            ptra: ProofTraceActions,
    ):
        indices = self._premises_indices + \
            list(range(ptra.prepare_len(), ptra.len()))

        # If we don't have any theorem to return (no premise, no action yet) we
        # return an invalid position (1 which is the EMPTY action).
        if len(indices) == 0:
            return 1

        probs = [
            float(p) / sum(range(1, len(indices)+1))
            for p in range(1, len(indices)+1)
        ]
        return indices[
            numpy.array(probs).cumsum().searchsorted(numpy.random.sample(1))[0]
        ]

    def sample_subst(
            self,
    ):
        return random.choice(self._subst_indices)

    def sample_subst_type(
            self,
    ):
        return random.choice(self._subst_type_indices)

    def sample(
            self,
            ptra: ProofTraceActions,
            repl: REPL,
            tries: int,
            conclusion: bool = False,
    ) -> Action:
        for i in range(tries):
            if not conclusion:
                action = random.choice(list(NON_PREPARE_TOKENS.values()))
            else:
                action = random.choice(list(CONCLUSION_TOKENS.values()))

            if INV_ACTION_TOKENS[action] == 'REFL':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'TRANS':
                left = self.sample_theorem(ptra)
                right = self.sample_theorem(ptra)
            if INV_ACTION_TOKENS[action] == 'MK_COMB':
                left = self.sample_theorem(ptra)
                right = self.sample_theorem(ptra)
            if INV_ACTION_TOKENS[action] == 'ABS':
                left = self.sample_theorem(ptra)
                right = self.sample_term()
            if INV_ACTION_TOKENS[action] == 'BETA':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'ASSUME':
                left = self.sample_term()
                right = 0
            if INV_ACTION_TOKENS[action] == 'EQ_MP':
                left = self.sample_theorem(ptra)
                right = self.sample_theorem(ptra)
            if INV_ACTION_TOKENS[action] == 'DEDUCT_ANTISYM_RULE':
                left = self.sample_theorem(ptra)
                right = self.sample_theorem(ptra)
            if INV_ACTION_TOKENS[action] == 'INST':
                left = self.sample_theorem(ptra)
                right = self.sample_subst()
            if INV_ACTION_TOKENS[action] == 'INST_TYPE':
                left = self.sample_theorem(ptra)
                right = self.sample_subst_type()

            a = Action.from_action(
                INV_ACTION_TOKENS[action],
                ptra.arguments()[left],
                ptra.arguments()[right],
            )

            if ptra.seen(a):
                continue

            if not repl.valid(a):
                continue

            return a

        return None


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

        self._sampler = RandomSampler(self._ptra)

    def step(
            self,
            offset: int = 0,
            conclusion: bool = False,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        index, actions, arguments = self.preprocess_ptra(self._ptra)

        candidate = self._sampler.sample(
            self._ptra, self._repl, 32, conclusion
        )

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
