import argparse
import concurrent.futures
import datetime
import os
import pickle
import random
import re
import typing

from dataset.prooftrace import \
    ACTION_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.fusion import FusionException
from prooftrace.repl.repl import REPL, REPLException

from utils.config import Config
from utils.log import Log


class Env:
    def __init__(
            self,
            config: Config,
            test: bool,
    ) -> None:
        self._sequence_length = config.get('prooftrace_sequence_length')
        self._alpha = config.get('prooftrace_env_alpha')

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
                'test_traces'
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
                Log.out("Selecting trace", {
                    "trace": self._ground.name(),
                    'length': self._ground.len(),
                })

        self._run = ProofTraceActions(
            'REPL-{}'.format(
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

    def step(
            self,
            a: typing.Tuple[int, int, int],
    ) -> typing.Tuple[
        typing.Tuple[int, typing.List[Action]],
        float,
        bool,
    ]:
        assert self._ground is not None
        assert self._run is not None

        if a[1] >= self._run.len():
            return self.observation(), (0.0, 0.0), True
        if a[2] >= self._run.len():
            return self.observation(), (0.0, 0.0), True

        action = Action.from_action(
            INV_ACTION_TOKENS[a[0]],
            self._run.actions()[a[1]],
            self._run.actions()[a[2]],
        )

        try:
            thm = self._repl.apply(action)
        except FusionException:
            return self.observation(), (0.0, 0.0), True
        except REPLException:
            return self.observation(), (0.0, 0.0), True

        seen = self._run.seen(action)
        self._run.append(action)

        step_reward = 0.0
        final_reward = 0.0
        done = False

        if not seen:
            step_reward = 1.0
            if self._ground.seen(action):
                step_reward = 2.0

        if self._target.thm_string(True) == thm.thm_string(True):
            # TODO(stan): for now we return the ground ptra length as final
            # reward, hoping that the RL decay will push the agent to minimize
            # sequences length. To investigate.
            final_reward = float(self._ground.len())
            done = True
        if self._run.len() >= self._sequence_length:
            done = True

        if step_reward > 1.0:
            Log.out("MATCH", {
                'name': self._ground.name(),
                'step_reward': step_reward,
                'final_reward': final_reward,
                'ground_length': self._ground.len(),
                'run_length': self._run.len(),
            })
        Log.out("ACTION", {
            'name': self._ground.name(),
            'step_reward': step_reward,
            'final_reward': final_reward,
            'ground_length': self._ground.len(),
            'run_length': self._run.len(),
        })

        return self.observation(), (step_reward, final_reward), done


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
    ) -> typing.Tuple[
            typing.List[int],
            typing.List[typing.List[Action]],
    ]:
        def reset(env):
            return env.reset()

        observations = []
        for o in self._executor.map(reset, self._pool):
            observations.append(o)

        return self.collate(observations)

    def step(
            self,
            actions: typing.List[typing.Tuple[int, int, int]],
    ) -> typing.Tuple[
        typing.Tuple[
            typing.List[int],
            typing.List[typing.List[Action]],
        ],
        typing.List[float],
        typing.List[bool],
    ]:
        def step(a):
            return a[0].step(a[1])

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

    pool = Pool(config, False)
    pool.reset()

    observations, rewards, dones = pool.step(
        [[8, 12, 13]] * config.get('prooftrace_env_pool_size'),
    )
    for i in range(config.get('prooftrace_env_pool_size')):
        Log.out("STEP", {
            'index': i,
            'reward': rewards[i],
            'done': dones[i],
        })

    observations, rewards, dones = pool.step(
        [[9, 12, 13]] * config.get('prooftrace_env_pool_size'),
    )
    for i in range(config.get('prooftrace_env_pool_size')):
        Log.out("STEP", {
            'index': i,
            'reward': rewards[i],
            'done': dones[i],
        })
