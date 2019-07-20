import torch
import typing
import torch.distributions as D
import torch.nn.functional as F

from prooftrace.prooftrace import \
    PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.model import LModel, VModel
from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search.search import Search

from utils.config import Config


class ParticleFilter(Search):
    def __init__(
            self,
            config: Config,
            l_model: LModel,
            v_model: VModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(ParticleFilter, self).__init__(config, ptra, repl, target)

        self._l_model = l_model
        self._v_model = v_model

        self._filter_size = \
            config.get('prooftrace_search_particle_filter_size')
        self._sample_size = \
            config.get('prooftrace_search_particle_filter_sample_size')

        self._particles = [{
            'ptra': ptra.copy(),
            'repl': repl.copy(),
        } for _ in range(self._filter_size)]

    def step(
            self,
            offset: int = 0,
            conclusion: bool = False,
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
                self._l_model.infer(idx, act, arg)

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

                    samples[h] = {
                        'repl': repl,
                        'ptra': ptra,
                    }

                    if len(samples) >= self._sample_size:
                        break

                if len(samples) >= self._sample_size:
                    break

        # Resampling based on value
        samples = list(samples.values())

        if len(samples) == 0:
            return True, self._particles[0]['ptra'], False

        idx = []
        act = []
        arg = []

        for p in samples:
            index, actions, arguments = self.preprocess_ptra(p['ptra'])

            idx += [index]
            act += [actions]
            arg += [arguments]

        with torch.no_grad():
            prd_values = \
                self._v_model.infer(idx, act, arg)

        costs = F.log_softmax(prd_values)

        m = D.Categorical(logits=costs)
        indices = m.sample((self._filter_size,)).numpy()
        self._particles = []

        for idx in indices:
            self._particles.append(samples[idx])

        return False, self._particles[0]['ptra'], False
