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
            model: SearchModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(Beam, self).__init__(config, model, ptra, repl, target)

        index, actions, arguments = self.preprocess_ptra(ptra)

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights, prd_values = \
                self._model.infer([index], [actions], [arguments])

        self._ptras = [ptra.copy()]
        self._repls = [repl.copy()]
        self._heads = [
            Head(
                prd_actions[0].cpu(),
                prd_lefts[0].cpu(),
                prd_rights[0].cpu(),
                # prd_values[0].cpu().item(),  # VALUE
                1.0,  # PROB
            )
        ]

    def preprocess_ptra(
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
            final: bool = False,
            offset: int = 0,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        idx = []
        act = []
        arg = []

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
                    Log.out("DEMONSTRATED", {
                        'theorem': thm.thm_string(True),
                        'summary': ptra.summary(offset),
                    })
                    return True, ptra, True

                candidates.append((ptra, repl, action, p))
                index, actions, arguments = self.preprocess_ptra(ptra)

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

        # Log.out("PRE-BEAM", {
        #     'candidates': len(candidates),
        # })
        if len(candidates) == 0:
            last_ptra = self._ptras[0]

            self._ptras = []
            self._repls = []
            self._heads = []

            return True, last_ptra, False

        with torch.no_grad():
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
                    candidates[i][3],  # PROB
                    # prd_values[i].cpu().item(),  # VALUE
                ),
                candidates[i][3],  # PROB
                # prd_values[i].cpu().item(),  # VALUE
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

        if final:
            return True, self._ptras[0], False
        else:
            return False, self._ptras[0], False
