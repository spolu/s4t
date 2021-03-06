import torch
import typing

from prooftrace.prooftrace import \
    PROOFTRACE_TOKENS, PREPARE_TOKENS, INV_PROOFTRACE_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.model import LModel
from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search.search import Search

from utils.config import Config
# from utils.log import Log


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
            len(PROOFTRACE_TOKENS) - len(PREPARE_TOKENS),
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
                        INV_PROOFTRACE_TOKENS[action + len(PREPARE_TOKENS)],
                        ptra.arguments()[left],
                        ptra.arguments()[right],
                    )

                    if ptra.seen(a):
                        continue

                    if not repl.valid(a):
                        continue

                    candidates.append((
                        self._value *  # PROB
                        top_actions[0][ia].item() *
                        top_lefts[0][il].item() *
                        top_rights[0][ir].item(),
                        a
                    ))

        return sorted(
            candidates, key=lambda c: c[0], reverse=True
        )[:head_width]


class Beam(Search):
    def __init__(
            self,
            config: Config,
            l_model: LModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(Beam, self).__init__(config, ptra, repl, target)

        self._l_model = l_model

        index, actions, arguments = self.preprocess_ptra(ptra)

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights = \
                self._l_model.infer([index], [actions], [arguments])

        self._ptras = [ptra.copy()]
        self._repls = [repl.copy()]
        self._heads = [
            Head(
                prd_actions[0][0].cpu(),
                prd_lefts[0][0].cpu(),
                prd_rights[0][0].cpu(),
                1.0,  # PROB
            )
        ]

    def step(
            self,
            offset: int = 0,
            conclusion: bool = False,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        candidates = []

        for i in range(len(self._heads)):
            for p, action in self._heads[i].apply(
                self._ptras[i],
                self._repls[i],
                self._config.get('prooftrace_search_beam_beta_width'),
                self._config.get('prooftrace_search_beam_head_width'),
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
                    return True, ptra, True

                candidates.append((ptra, repl, action, p))

        if len(candidates) == 0 or \
                candidates[0][0].len() == \
                self._config.get('prooftrace_sequence_length'):
            last_ptra = self._ptras[0]

            self._ptras = []
            self._repls = []
            self._heads = []

            return True, last_ptra, False

        h = {}
        uniques = []
        for c in candidates:
            if c[2].hash() not in h:
                h[c[2].hash()] = True
                uniques.append(c)

        candidates = uniques

        idx = None
        act = []
        arg = []

        for c in candidates:
            index, actions, arguments = self.preprocess_ptra(c[0])

            if idx is None:
                idx = index
            else:
                assert idx == index

            act.append(actions)
            arg.append(arguments)

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights = \
                self._l_model.infer([idx], act, arg)

        next_heads = []
        for i in range(len(candidates)):
            next_heads.append((
                candidates[i][0],
                candidates[i][1],
                Head(
                    prd_actions[i][0].cpu(),
                    prd_lefts[i][0].cpu(),
                    prd_rights[i][0].cpu(),
                    candidates[i][3],  # PROB
                ),
                candidates[i][3],  # PROB
            ))

        next_heads = sorted(
            next_heads, key=lambda v: v[3], reverse=True
        )[0:self._config.get('prooftrace_search_beam_width')]

        self._ptras = [v[0] for v in next_heads]
        self._repls = [v[1] for v in next_heads]
        self._heads = [v[2] for v in next_heads]

        # for v in next_heads:
        #     Log.out("BEAM", {
        #         'value': v[3],
        #         'summary': v[0].summary(offset),
        #     })

        return False, self._ptras[0], False
