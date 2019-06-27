import torch
import typing

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.model import Model
from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search.search import Search

from utils.config import Config


class PolicySample(Search):
    def __init__(
            self,
            config: Config,
            model: Model,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(PolicySample, self).__init__(config, model, ptra, repl, target)

        self._ptra = ptra.copy()
        self._repl = repl.copy()

    def step(
            self,
            offset: int = 0,
            conclusion: bool = False,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        index, actions, arguments = self.preprocess_ptra(self._ptra)

        idx = [index]
        act = [actions]
        arg = [arguments]

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights = \
                self._model.infer(idx, act, arg)

        beta_width = \
            self._config.get('prooftrace_search_policy_sample_beta_width')

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions[0].cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts[0].cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights[0].cpu()).topk(beta_width)

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

                    candidates.append((
                        top_actions[0][ia].item() *
                        top_lefts[0][il].item() *
                        top_rights[0][ir].item(),
                        a
                    ))

        if len(candidates) == 0:
            return True, self._ptra, False

        action = sorted(
            candidates, key=lambda c: c[0], reverse=True
        )[0][1]

        thm = self._repl.apply(action)
        action._index = thm.index()
        argument = self._ptra.build_argument(
            thm.concl(), thm.hyp(), thm.index(),
        )
        self._ptra.append(action, argument)

        if self._target.thm_string(True) == thm.thm_string(True):
            return True, self._ptra, True

        return False, self._ptra, False
