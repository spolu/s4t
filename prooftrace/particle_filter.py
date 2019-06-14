import torch
import typing
import torch.distributions as D
import torch.nn.functional as F

from prooftrace.prooftrace import \
    PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.model import Model
from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search import Search

from utils.config import Config


class ParticleFilter(Search):
    def __init__(
            self,
            config: Config,
            model: Model,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(ParticleFilter, self).__init__(config, model, ptra, repl, target)

        self._filter_size = \
            config.get('prooftrace_search_particle_filter_size')
        self._sample_size = \
            config.get('prooftrace_search_particle_filter_sample_size')

        self._particles = [{
            'ptra': ptra.copy(),
            'repl': repl.copy(),
            'cost': 1.0,
        } for _ in range(self._filter_size)]

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

        for p in self._particles:
            index, actions, arguments = self.preprocess_ptra(p['ptra'])

            idx += [index]
            act += [actions]
            arg += [arguments]

        with torch.no_grad():
            prd_actions, prd_lefts, prd_rights = \
                self._model.infer(idx, act, arg)

        m_actions = D.Categorical(logits=prd_actions)
        m_lefts = D.Categorical(logits=prd_lefts)
        m_rights = D.Categorical(logits=prd_rights)

        samples = {}

        for s in range(self._sample_size):
            actions = m_actions.sample()
            lefts = m_lefts.sample()
            rights = m_rights.sample()

            for i, p in enumerate(self._particles):
                action = actions[i].item()
                left = lefts[i].item()
                right = rights[i].item()

                if left >= p['ptra'].len() or right >= p['ptra'].len():
                    continue

                a = Action.from_action(
                    INV_ACTION_TOKENS[action + len(PREPARE_TOKENS)],
                    p['ptra'].arguments()[left],
                    p['ptra'].arguments()[right],
                )

                if p['ptra'].seen(a):
                    continue

                if not p['repl'].valid(a):
                    continue

                h = p['ptra'].actions()[-1].hash() + a.hash()

                if h not in samples:
                    # print(
                    #     str(len(PREPARE_TOKENS) + action) +
                    #     " " + str(left) +
                    #     " " + str(right)
                    # )
                    samples[h] = (
                        p, a,
                        torch.exp(prd_actions[i][action]).item() *
                        torch.exp(prd_lefts[i][left]).item() *
                        torch.exp(prd_rights[i][right]).item()
                    )

        # Resampling based on cost
        samples = list(samples.values())

        if len(samples) == 0:
            return True, self._particles[0]['ptra'], False

        costs = F.log_softmax(
            torch.tensor(
                [s[0]['cost'] * s[2] for s in samples]
            ),
            dim=0
        )

        m = D.Categorical(logits=costs)
        indices = m.sample((self._filter_size,)).numpy()
        self._particles = []

        for idx in indices:
            s = samples[idx]
            p = s[0]
            action = s[1]

            repl = p['repl'].copy()
            ptra = p['ptra'].copy()

            thm = repl.apply(action)
            action._index = thm.index()
            argument = ptra.build_argument(
                thm.concl(), thm.hyp(), thm.index(),
            )
            ptra.append(action, argument)

            if self._target.thm_string(True) == thm.thm_string(True):
                return True, ptra, True

            self._particles.append({
                'ptra': ptra,
                'repl': repl,
                'cost': p['cost'] * s[2],
            })

        if final:
            return True, self._particles[0]['ptra'], False
        else:
            return False, self._particles[0]['ptra'], False
