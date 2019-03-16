import os
import pickle
import random
import re
import typing

from dataset.prooftrace import Action

from prooftrace.repl.repl import REPL

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

        self._ptra = None
        self._repl = None
        self._ground_hashes = []
        self._seen_hashes = []

    def reset(
            self,
    ) -> typing.Tuple[int, typing.List[Action]]:
        self._ptra = None

        while self._ptra is None:
            ptra = pickle.load(random.choice(self._trace_files))
            if ptra.len() > self._sequence_length:
                Log.out("Ignoring trace", {
                    'trace': ptra.name(),
                    'length': ptra.len(),
                })
            else:
                self._ptra = ptra
                Log.out("Selecting trace", {
                    "trace": self._ptra.name(),
                    'length': ptra.len(),
                })

        self._repl = REPL(self._tokenizer)

        self._ground_hashes = [a.hash() for a in self._ptra.actions()]
        self._seen_hashes = []

    def step(
            self,
            action: typing.Tuple[int, int, int],
    ) -> typing.Typle[
        typing.Tuple[int, typing.List[Action]],
        float,
        bool,
    ]:
        assert self._ptra is not None
