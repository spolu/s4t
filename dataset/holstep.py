import os
import random
import torch

from utils.config import Config
from utils.log import Log

from torch.utils.data import Dataset


class HolStepKernel():
    def __init__(
            self,
            config: Config,
    ) -> None:
        self._tokens = {}
        self._token_count = 1

        self._bound_count = 0
        self._free_count = 0
        self._skip_count = 0

        self._theorem_length = config.get('th2vec_theorem_length')

    def process_formula(
            self,
            f: str,
    ) -> None:
        formula = []
        tokens = f.split(' ')

        if len(tokens) > self._theorem_length:
            self._skip_count += 1
            return None

        for t in tokens:
            if t[0] == 'b':
                b = int(t[1:])
                if b > self._bound_count:
                    self._bound_count = b
            elif t[0] == 'f':
                f = int(t[1:])
                if f > self._free_count:
                    self._free_count = f
            if t not in self._tokens:
                self._tokens[t] = self._token_count
                self._token_count += 1
            formula.append(self._tokens[t])

        return formula


class HolStepSet():
    def __init__(
            self,
            config: Config,
            kernel: HolStepKernel,
            dataset_dir: str,
    ) -> None:
        self._kernel = kernel

        self._formulas = []

        self._C = []
        self._D = {}
        self._P = {}
        self._M = {}

        dataset_dir = os.path.abspath(dataset_dir)

        Log.out(
            "Loading HolStep dataset", {
                'dataset_dir': dataset_dir,
            })

        assert os.path.isdir(dataset_dir)
        files = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]

        count = 0
        for p in files:
            count += 1
            self.process_file(p)

            if count % 100 == 0:
                Log.out(
                    "Processing HolStep dataset", {
                        'dataset_dir': dataset_dir,
                        'token_count': self._kernel._token_count,
                        'bound_count': self._kernel._bound_count,
                        'free_count': self._kernel._free_count,
                        'skip_count': self._kernel._skip_count,
                        'formula_count': len(self._formulas),
                        'conjecture_count': len(self._C),
                        'processed': count,
                    })

        Log.out(
            "Loaded HolStep dataset", {
                'dataset_dir': dataset_dir,
                'token_count': self._kernel._token_count,
                'bound_count': self._kernel._bound_count,
                'free_count': self._kernel._free_count,
                'skip_count': self._kernel._skip_count,
                'formula_count': len(self._formulas),
                'conjecture_count': len(self._C),
            })

    def process_file(
            self,
            path,
    ):
        with open(path, 'r') as f:
            lines = f.read().splitlines()

            c_idx = None
            for i in range(len(lines)):
                line = lines[i]
                if line[0] == 'T':
                    f = self._kernel.process_formula(line[2:])
                    if f is not None:
                        f_idx = len(self._formulas)
                        self._fomulas.append(f)

                        if lines[i-1][0] == 'C':
                            self._C.append(f_idx)
                            c_idx = f_idx
                            self._A[c_idx] = []
                            self._P[c_idx] = []
                            self._M[c_idx] = []

                        assert c_idx is not None

                        if lines[i-1][0] == 'A':
                            self._A[c_idx].append(f_idx)
                        if lines[i-1][0] == '+':
                            self._P[c_idx].append(f_idx)
                        if lines[i-1][0] == '-':
                            self._M[c_idx].append(f_idx)


class HolStepRelatedDataset(Dataset):
    def __init__(
            self,
            config: Config,
            hset: HolStepSet,
    ) -> None:
        self._hset = hset

    def __len__(
            self,
    ) -> int:
        return len(self._C)

    def __getitem__(
            self,
            idx: int,
    ):
        inp_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        rel_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        unr_t = torch.zeros(self._theorem_length, dtype=torch.int64)

        rel = random.choice(self._hset._D[idx] + self._hset._P[idx])
        unr = None
        while(unr is None):
            candidate = random.choice(self._hset._formulas)
            if (
                    candidate not in self._hset._D[idx] and
                    candidate not in self._hset._P[idx] and
                    candidate not in self._hset._M[idx]
            ):
                unr = candidate

        for i in range(len(self._hset._formulas[idx])):
            t = self._hset._formulas[idx][i]
            assert t != 0
            inp_t[i] = t
        for i in range(len(self._hset._formulas[rel])):
            t = self._hset._formulas[rel][i]
            assert t != 0
            rel_t[i] = t
        for i in range(len(self._hset._formulas[unr])):
            t = self._hset._formulas[unr][i]
            assert t != 0
            unr_t[i] = t

        return inp_t, rel_t, unr_t
