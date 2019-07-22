import torch
import typing
import torch.distributions as D
import torch.nn.functional as F

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
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

    def topk(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            beta_width: int,
    ):
        samples = {}

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions.cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts.cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights.cpu()).topk(beta_width)

        for ia in range(a_count):
            for il in range(beta_width):
                for ir in range(beta_width):
                    for i, p in enumerate(self._particles):
                        action = top_actions[1][i][ia].item()
                        left = top_lefts[1][i][il].item()
                        right = top_rights[1][i][ir].item()

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

                            thm = repl.apply(a)
                            a._index = thm.index()

                            argument = ptra.build_argument(
                                thm.concl(), thm.hyp(), thm.index(),
                            )
                            ptra.append(a, argument)

                            if self._target.thm_string(True) == \
                                    thm.thm_string(True):
                                return True, ptra, True

                            print(
                                "STORE {} {} {}  {} {} {}".format(
                                    len(PREPARE_TOKENS) + action,
                                    left,
                                    right,
                                    torch.exp(prd_actions)[i][action],
                                    torch.exp(prd_lefts)[i][left],
                                    torch.exp(prd_rights)[i][right],
                                ),
                            )

                            samples[h] = {
                                'repl': repl,
                                'ptra': ptra,
                            }

        # Resampling based on value
        samples = list(samples.values())
        # import pdb; pdb.set_trace();

        return samples

    def sample(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
    ):
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

                # print(
                #     "TRY {} {} {}  {} {} {}".format(
                #         len(PREPARE_TOKENS) + action,
                #         left,
                #         right,
                #         torch.exp(prd_actions)[i][action],
                #         torch.exp(prd_lefts)[i][left],
                #         torch.exp(prd_rights)[i][right],
                #     ),
                # )

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

                    thm = repl.apply(a)
                    a._index = thm.index()

                    argument = ptra.build_argument(
                        thm.concl(), thm.hyp(), thm.index(),
                    )
                    ptra.append(a, argument)

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
        # import pdb; pdb.set_trace();

        return samples

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

        print("PARTICLES LEN {}".format(len(self._particles)))

        # samples = self.sample(prd_actions, prd_lefts, prd_rights)
        # samples = self.topk(prd_actions, prd_lefts, prd_rights, 3)

        beta_width = 5
        samples = {}

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions.cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts.cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights.cpu()).topk(beta_width)

        for ia in range(a_count):
            for il in range(beta_width):
                for ir in range(beta_width):
                    for i, p in enumerate(self._particles):
                        action = top_actions[1][i][ia].item()
                        left = top_lefts[1][i][il].item()
                        right = top_rights[1][i][ir].item()

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

                            thm = repl.apply(a)
                            a._index = thm.index()

                            argument = ptra.build_argument(
                                thm.concl(), thm.hyp(), thm.index(),
                            )
                            ptra.append(a, argument)

                            if self._target.thm_string(True) == \
                                    thm.thm_string(True):
                                return True, ptra, True

                            print(
                                "STORE {} {} {}  {} {} {}".format(
                                    len(PREPARE_TOKENS) + action,
                                    left,
                                    right,
                                    torch.exp(prd_actions)[i][action],
                                    torch.exp(prd_lefts)[i][left],
                                    torch.exp(prd_rights)[i][right],
                                ),
                            )

                            samples[h] = {
                                'repl': repl,
                                'ptra': ptra,
                            }

        # Resampling based on value
        samples = list(samples.values())
        # import pdb; pdb.set_trace();

        print("SAMPLES LEN {}".format(len(samples)))

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

        costs = F.log_softmax(prd_values, dim=1)

        m = D.Categorical(logits=costs)
        indices = m.sample((self._filter_size,)).cpu().numpy()
        self._particles = []

        for idx in indices:
            self._particles.append(samples[idx[0]])

        return False, self._particles[0]['ptra'], False
