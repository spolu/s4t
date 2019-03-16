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
            with open(path, 'rb') as f:
                ptra = pickle.load(f)
            if ptra.len() > self._sequence_length:
                Log.out("Ignoring trace", {
                    'trace': ptra.name(),
                    'length': ptra.len(),
                })
            else:
                self._ground = ptra
                Log.out("Selecting trace", {
                    "trace": ptra.name(),
                    'length': ptra.len(),
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
        pass

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
            return self.observation(), -1.0, False
        if a[2] >= self._run.len():
            return self.observation(), -1.0, False

        action = Action.from_action(
            INV_ACTION_TOKENS[a[0]],
            self._run.actions()[a[1]],
            self._run.actions()[a[2]],
        )

        if self._run.seen(action):
            return self.observation(), -1.0, False

        try:
            thm = self._repl.apply(action)
        except FusionException:
            return self.observation(), -1.0, False
        except REPLException:
            return self.observation(), -1.0, False

        self._run.append(action)

        if self._target.thm_string(True) == thm.thm_string(True):
            return self.observation(), float(
                self.sequence_length - self._run.len()
            ), True

        done = False
        if self._run.len() >= self._sequence_length:
            done = True

        if self._ground.seen(action):
            return self.observation(), 1.0, done
        else:
            return self.observation(), 0.0, done


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

    def reset(
            self,
    ) -> typing.List[
        typing.Tuple[int, typing.List[Action]],
    ]:
        def reset(env):
            return env.reset()

        observations = []
        for o in self._executor.map(reset, self._pool):
            observations.append(o)

        return observations

    def step(
            self,
            actions,
    ) -> typing.Tuple[
        typing.List[typing.Tuple[int, typing.List[Action]]],
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

        return observations, rewards, dones


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

    observations, rewards, dones = pool.step([[8, 12, 13]] * 8)
    for i in range(8):
        Log.out("STEP", {
            'index': i,
            'reward': rewards[i],
            'done': dones[i],
        })

    observations, rewards, dones = pool.step([[9, 12, 13]] * 8)
    for i in range(8):
        Log.out("STEP", {
            'index': i,
            'reward': rewards[i],
            'done': dones[i],
        })
