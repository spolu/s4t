import argparse
import concurrent.futures
import datetime
import os
import pickle
import random
import re
import torch
import typing

from dataset.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.fusion import FusionException
from prooftrace.repl.repl import REPL, REPLException

from torch.distributions import Categorical

from utils.config import Config
from utils.log import Log


class Env:
    def __init__(
            self,
            config: Config,
            test: bool,
    ) -> None:
        self._sequence_length = config.get('prooftrace_sequence_length')

        self._device = torch.device(config.get('device'))

        if test:
            dataset_dir = os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'test_traces'
            )
        else:
            dataset_dir = os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'train_traces'
            )
        assert os.path.isdir(dataset_dir)

        self._trace_files = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if (os.path.isfile(os.path.join(dataset_dir, f)) and
                re.search("\\.actions$", f) is not None)
        ]

        with open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._tokenizer = pickle.load(f)

        self._ground = None
        self._run = None
        self._repl = None
        self._target = None

    def reset(
            self,
            gamma: float,
    ) -> typing.Tuple[int, typing.List[Action]]:
        self._ground = None
        self._run = None
        self._repl = None
        self._target = None

        while self._ground is None:
            path = random.choice(self._trace_files)

            match = re.search("_(\\d+)_(\\d+)\\.actions$", path)
            ptra_len = int(match.group(1))

            if ptra_len <= self._sequence_length:
                with open(path, 'rb') as f:
                    self._ground = pickle.load(f)
                # Log.out("Selecting trace", {
                #     "trace": self._ground.name(),
                #     'length': self._ground.len(),
                # })

        self._run = ProofTraceActions(
            'REPL-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                a for a in self._ground.actions()
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

        self._repl = REPL(self._tokenizer)
        self._target = self._repl.prepare(self._run)

        # GAMMA Initialization.
        if gamma > 0.0 and random.random() < gamma:
            gamma_steps = random.randange(0, self._ground.action_len())
            for i in range(gamma_steps):
                a = self._ground.actions()[self._ground.prepare_len() + i]
                self._run.append(a)
                self._repl.apply(a)

        return self.observation()

    def observation(
            self,
    ) -> typing.Tuple[int, typing.List[Action]]:
        actions = self._run.actions().copy()

        # If the len match this is a final observation, so no extract will be
        # appended and that's fine.
        if len(actions) < self._sequence_length:
            actions.append(Action.from_action('EXTRACT', None, None))

        # Finally we always return actions with the same length.
        while len(actions) < self._sequence_length:
            actions.append(Action.from_action('EMPTY', None, None))

        return (self._run.len(), actions)

    def alpha_oracle(
            self,
    ) -> typing.Tuple[torch.Tensor, int]:
        for i in range(self._ground.prepare_len(), self._ground.len()):
            a = self._ground.actions()[i]
            if (not self._run.seen(a)) and \
                    self._run.seen(a.left) and \
                    self._run.seen(a.right):
                assert 0 <= a.value - len(PREPARE_TOKENS)
                assert a.value < len(ACTION_TOKENS)
                actions = torch.tensor([[
                    a.value - len(PREPARE_TOKENS),
                    self._run.hashes()[a.left.hash()],
                    self._run.hashes()[a.right.hash()],
                ]], dtype=torch.int64).to(self._device)
                return actions, 0
        assert False

    def beta_oracle(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            beta_width: int,
            beta_size: int,
    ) -> typing.Tuple[torch.Tensor, int]:
        top_actions = torch.exp(prd_actions).topk(beta_width)
        top_lefts = torch.exp(prd_lefts).topk(beta_width)
        top_rights = torch.exp(prd_rights).topk(beta_width)

        out = []
        frame_count = 0

        for ia in range(beta_width):
            for il in range(beta_width):
                for ir in range(beta_width):
                    action = top_actions[1][ia].item()
                    assert action >= 0
                    assert action < len(ACTION_TOKENS) - len(PREPARE_TOKENS)
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()
                    prob = top_actions[0][ia].item() * \
                        top_lefts[0][il].item() * \
                        top_rights[0][ir].item()

                    if left >= self._run.len() or right >= self._run.len():
                        out.append(([action, left, right], prob))
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action + len(PREPARE_TOKENS)],
                        self._run.actions()[left],
                        self._run.actions()[right],
                    )

                    if self._run.seen(a):
                        out.append(([action, left, right], prob))
                        continue

                    frame_count += 1
                    if not self._repl.valid(a):
                        out.append(([action, left, right], prob - 1.0))
                        continue

                    out.append(([action, left, right], prob + 1.0))

        out = sorted(out, key=lambda o: o[1], reverse=True)

        actions = []
        for i in range(beta_size):
            actions.append(out[i][0])

        return \
            torch.tensor(actions, dtype=torch.int64).to(self._device), \
            frame_count

    def explore(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            alpha: float,
            beta: float,
            beta_width: int,
    ) -> typing.Tuple[torch.Tensor, int]:

        # ALPHA Oracle.
        if alpha > 0.0 and random.random() < alpha:
            return self.alpha_oracle()

        # BETA Oracle.
        if beta > 0.0 and random.random() < beta:
            return self.beta_oracle(
                prd_actions, prd_lefts, prd_rights,
                beta_width, 1,
            )

        # Sampling.
        actions = torch.cat((
            Categorical(
                torch.exp(prd_actions)
            ).sample().unsqueeze(0).unsqueeze(1),
            Categorical(
                torch.exp(prd_lefts)
            ).sample().unsqueeze(0).unsqueeze(1),
            Categorical(
                torch.exp(prd_rights)
            ).sample().unsqueeze(0).unsqueeze(1),
        ), dim=1)

        return actions, 0

    def step(
            self,
            action: typing.Tuple[int, int, int],
            step_reward_prob: float,
            match_reward_prob: float,
    ) -> typing.Tuple[
        typing.Tuple[int, typing.List[Action]],
        typing.Tuple[float, float, float],
        bool,
    ]:
        assert self._ground is not None
        assert self._run is not None

        if action[1] >= self._run.len() or action[2] >= self._run.len():
            return self.observation(), (0.0, 0.0, 0.0), True

        a = Action.from_action(
            INV_ACTION_TOKENS[action[0] + len(PREPARE_TOKENS)],
            self._run.actions()[action[1]],
            self._run.actions()[action[2]],
        )

        if self._run.seen(a):
            return self.observation(), (0.0, 0.0, 0.0), False

        try:
            thm = self._repl.apply(a)
        except (FusionException, REPLException):
            Log.out('DONE ILLEGAL', {
                'ground_length': self._ground.action_len(),
                'run_length': self._run.action_len(),
                'name': self._ground.name(),
            })
            return self.observation(), (-1.0, 0.0, 0.0), True

        self._run.append(a)

        step_reward = 0.0
        match_reward = 0.0
        final_reward = 0.0
        done = False

        if step_reward_prob > 0.0 and random.random() < step_reward_prob:
            step_reward = 1.0

        if self._ground.seen(a):
            if match_reward_prob > 0.0 and random.random() < match_reward_prob:
                match_reward = 2.0
                step_reward = 0.0

        if self._target.thm_string(True) == thm.thm_string(True):
            final_reward = float(self._ground.len())
            Log.out("DEMONSTRATED", {
                'name': self._ground.name(),
                'ground_length': self._ground.action_len(),
                'run_length': self._run.action_len(),
            })
            done = True
        if self._run.len() >= self._sequence_length:
            done = True

        if done:
            Log.out('DONE', {
                'ground_length': self._ground.action_len(),
                'run_length': self._run.action_len(),
                'name': self._ground.name(),
            })

        return self.observation(), \
            (step_reward, match_reward, final_reward), \
            done


class Pool:
    def __init__(
            self,
            config,
            test: bool,
    ):
        self._pool_size = config.get('prooftrace_env_pool_size')
        self._pool = [
            Env(config, test) for _ in range(self._pool_size)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._pool_size,
        )

    def shutdown(
            self,
    ):
        self._executor.shutdown()

    def collate(
            self,
            observations,
    ) -> typing.Tuple[
            typing.List[int],
            typing.List[typing.List[Action]],
    ]:
        indices = []
        traces = []

        for (idx, trc) in observations:
            indices.append(idx)
            traces.append(trc)

        return (indices, traces)

    def reset(
            self,
            gamma: float,
    ) -> typing.Tuple[
            typing.List[int],
            typing.List[typing.List[Action]],
    ]:
        def reset(env):
            return env.reset(gamma)

        observations = []
        for o in self._executor.map(reset, self._pool):
            observations.append(o)

        return self.collate(observations)

    def explore(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            alpha: float,
            beta: float,
            beta_width: int,
    ) -> typing.Tuple[torch.Tensor, int]:
        def explore(a):
            return a[0].explore(
                a[1], a[2], a[3], alpha, beta, beta_width,
            )

        args = []
        assert len(self._pool) == prd_actions.size(0)
        for i in range(len(self._pool)):
            args.append([
                self._pool[i], prd_actions[i], prd_lefts[i], prd_rights[i]
            ])

        frame_count = 0
        actions = []

        for a, f in self._executor.map(explore, args):
            actions.append(a)
            frame_count += f

        return torch.cat(actions, dim=0), frame_count

    def step(
            self,
            actions: typing.List[typing.Tuple[int, int, int]],
            step_reward_prob: float,
            match_reward_prob: float,
    ) -> typing.Tuple[
        typing.Tuple[
            typing.List[int],
            typing.List[typing.List[Action]],
        ],
        typing.List[typing.Tuple[float, float, float]],
        typing.List[bool],
    ]:
        def step(a):
            return a[0].step(
                a[1], step_reward_prob, match_reward_prob,
            )

        args = []
        for i in range(len(actions)):
            args.append([self._pool[i], actions[i]])

        observations = []
        dones = []
        rewards = []

        for o, r, d in self._executor.map(step, args):
            observations.append(o)
            rewards.append(r)
            dones.append(d)

        for i in range(len(dones)):
            if dones[i]:
                self._pool[i].reset()

        return self.collate(observations), rewards, dones


def test():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    sequence_size = config.get('prooftrace_sequence_length')
    action_size = len(ACTION_TOKENS) - len(PREPARE_TOKENS)

    prd_actions = torch.rand(action_size)
    prd_lefts = torch.rand(sequence_size)
    prd_rights = torch.rand(sequence_size)

    prd_actions = torch.log(prd_actions / prd_actions.sum())
    prd_lefts = torch.log(prd_lefts / prd_lefts.sum())
    prd_rights = torch.log(prd_rights / prd_rights.sum())

    env = Env(config, False)
    env.reset()
    env.explore(
        prd_actions,
        prd_lefts,
        prd_rights,
        1.0,
        1.0,
        3,
    )
    print(".")

    # pool = Pool(config, False)
    # pool.reset()

    # observations, rewards, dones = pool.step(
    #     [[8, 12, 13]] * config.get('prooftrace_env_pool_size'),
    # )
    # for i in range(config.get('prooftrace_env_pool_size')):
    #     Log.out("STEP", {
    #         'index': i,
    #         'reward': rewards[i],
    #         'done': dones[i],
    #     })

    # observations, rewards, dones = pool.step(
    #     [[9, 12, 13]] * config.get('prooftrace_env_pool_size'),
    # )
    # for i in range(config.get('prooftrace_env_pool_size')):
    #     Log.out("STEP", {
    #         'index': i,
    #         'reward': rewards[i],
    #         'done': dones[i],
    #     })
