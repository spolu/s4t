import os
import random
import torch

from utils.config import Config
from utils.log import Log

from torch.utils.data import Dataset


class HolStepKernel():
    def __init__(
            self,
            theorem_length: int,
    ) -> None:
        self._tokens = {}
        self._token_count = 1
        self._theorem_length = theorem_length

        self._bound_count = 0
        self._free_count = 0
        self._skip_count = 0

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
            kernel: HolStepKernel,
            dataset_dir: str,
    ) -> None:
        self._kernel = kernel

        self._formulas = []
        self._theorems = {}

        self._C = []
        self._T = []

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
                        'theorem_count': len(self._theorems),
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
                'theorem_count': len(self._theorems),
                'conjecture_count': len(self._C),
            })

    def process_file(
            self,
            path,
    ):
        with open(path, 'r') as f:
            lines = f.read().splitlines()

            c_idx = None
            has_dep = False
            has_rel = False
            for i in range(len(lines)):
                line = lines[i]
                if line[0] == 'T':
                    f = self._kernel.process_formula(line[2:])
                    if f is not None:
                        f_idx = len(self._formulas)
                        self._formulas.append(f)

                        if lines[i-1][0] == 'C':
                            c_idx = f_idx
                            self._D[c_idx] = []
                            self._P[c_idx] = []
                            self._M[c_idx] = []

                        assert c_idx is not None

                        if lines[i-1][0] == 'A':
                            self._D[c_idx].append(f_idx)
                            has_dep = True
                            if f_idx not in self._theorems:
                                self._theorems[f_idx] = []
                                self._T.append(f_idx)
                            self._theorems[f_idx].append(c_idx)
                        if lines[i-1][0] == '+':
                            self._P[c_idx].append(f_idx)
                            has_rel = True
                        if lines[i-1][0] == '-':
                            self._M[c_idx].append(f_idx)

                    elif lines[i-1][0] == 'C':
                        return

            if has_dep and has_rel:
                self._C.append(c_idx)


class HolStepPremiseDataset(Dataset):
    def __init__(
            self,
            hset: HolStepSet,
    ) -> None:
        self._hset = hset
        self._theorem_length = hset._kernel._theorem_length

    def __len__(
            self,
    ) -> int:
        return 2*len(self._hset._C)

    def __getitem__(
            self,
            idx: int,
    ):
        cnj_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        thr_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        pre_t = torch.ones(1)

        cnj = self._hset._C[int(idx/2)]

        thr = random.choice(self._hset._D[cnj])

        if idx % 2 == 1:
            pre_t = torch.zeros(1)
            unr = None
            while(unr is None):
                candidate = random.choice(self._hset._T)
                if candidate not in self._hset._D[cnj]:
                    unr = candidate
            thr = unr

        for i in range(len(self._hset._formulas[cnj])):
            t = self._hset._formulas[cnj][i]
            assert t != 0
            cnj_t[i] = t
        for i in range(len(self._hset._formulas[thr])):
            t = self._hset._formulas[thr][i]
            assert t != 0
            thr_t[i] = t

        return cnj_t, thr_t, pre_t


class HolStepClassificationDataset(Dataset):
    def __init__(
            self,
            hset: HolStepSet,
    ) -> None:
        self._hset = hset
        self._theorem_length = hset._kernel._theorem_length

    def __len__(
            self,
    ) -> int:
        return 2*len(self._hset._C)

    def __getitem__(
            self,
            idx: int,
    ):
        cnj_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        thr_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        pre_t = torch.ones(1)

        cnj = self._hset._C[int(idx/2)]

        thr = random.choice(self._hset._P[cnj])

        if idx % 2 == 1:
            pre_t = torch.zeros(1)
            thr = random.choice(self._hset._M[cnj])

        for i in range(len(self._hset._formulas[cnj])):
            t = self._hset._formulas[cnj][i]
            assert t != 0
            cnj_t[i] = t
        for i in range(len(self._hset._formulas[thr])):
            t = self._hset._formulas[thr][i]
            assert t != 0
            thr_t[i] = t

        return cnj_t, thr_t, pre_t


# def preprocess():
#     kernel = HolStepKernel(512)
#     test_set = HolStepSet(
#         kernel,
#         os.path.expanduser("./data/th2vec/holstep/test"),
#     )
