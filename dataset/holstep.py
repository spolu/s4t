import os
import random
import torch

from utils.log import Log

from torch.utils.data import Dataset


class HolStepKernel():
    def __init__(
            self,
            theorem_length: int,
    ) -> None:
        self._compression = {}

        self._tokens = {}
        self._token_count = 1
        self._theorem_length = theorem_length

        self._bound_count = 0
        self._free_count = 0

    def process_formula(
            self,
            f: str,
    ):
        formula = []
        tokens = f.split(' ')

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

        for i in range(len(formula)):
            for j in range(1, 3):
                if i + j < len(formula):
                    ngram = str(formula[i:i+j+1])
                    if ngram not in self._compression:
                        self._compression[ngram] = 0
                        self._token_count += 1
                    self._compression[ngram] += j

        return formula

    def postprocess_compression(
            self,
            size: int,
    ) -> None:
        best = sorted(
            self._compression.keys(),
            key=lambda ngram: self._compression[ngram],
            reverse=True,
        )

        for i in range(size):
            # Should not be any collision on `str(formula[i:i+j+1])`
            self._tokens[best[i]] = self._token_count
            self._token_count += 1

    def postprocess_formula(
            self,
            f,
    ):
        formula = []

        i = 0
        while i < len(f):
            done = False
            for j in reversed(range(1, 3)):
                if i + j < len(f):
                    ngram = str(f[i:i+j+1])
                    if ngram in self._tokens:
                        formula.append(self._tokens[ngram])
                        i += j+1
                        done = True
                        break
            if not done:
                formula.append(f[i])
                i += 1

        return formula


class HolStepSet():
    def __init__(
            self,
            kernel: HolStepKernel,
            dataset_dir: str,
    ) -> None:
        self._kernel = kernel

        # The actual tokenized formulas.
        self._formulas = []
        self._max_length = 0

        # All conjectures.
        self._C = []
        # Conjectures with premises (should be all).
        self._C_premise = []
        # Conjectures with proof steps.
        self._C_step = []
        # All premises to conjectures.
        self._T = []
        # Premises for a conjecture in `self._C_premise`.
        self._D = {}
        # Positive proof steps for a conjecture in `self._C_step`.
        self._P = {}
        # Negative proof steps for a conjecture in `self._C_step`.
        self._N = {}

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
                    "PreProcessing HolStep dataset", {
                        'dataset_dir': dataset_dir,
                        'token_count': self._kernel._token_count,
                        'bound_count': self._kernel._bound_count,
                        'free_count': self._kernel._free_count,
                        'max_length': self._max_length,
                        'formula_count': len(self._formulas),
                        'theorem_count': len(self._T),
                        'conjecture_count': len(self._C),
                        'processed': count,
                    })

        self.postprocess()

        Log.out(
            "Loaded HolStep dataset", {
                'dataset_dir': dataset_dir,
                'token_count': self._kernel._token_count,
                'bound_count': self._kernel._bound_count,
                'free_count': self._kernel._free_count,
                'max_length': self._max_length,
                'formula_count': len(self._formulas),
                'theorem_count': len(self._T),
                'conjecture_count': len(self._C),
            })

    def process_file(
            self,
            path,
    ):
        with open(path, 'r') as f:
            lines = f.read().splitlines()

            c_idx = None
            has_premise = False
            has_step = False

            for i in range(len(lines)):
                line = lines[i]
                if line[0] == 'T':
                    f = self._kernel.process_formula(line[2:])
                    assert f is not None

                    f_idx = len(self._formulas)
                    self._formulas.append(f)

                    self._max_length = max(self._max_length, len(f))
                    # if len(f) > self._kernel._theorem_length:
                    #     Log.out("Excessive length formula", {
                    #         'length': len(f),
                    #     })

                    if lines[i-1][0] == 'C':
                        c_idx = f_idx
                        self._C.append(c_idx)
                        self._D[c_idx] = []
                        self._P[c_idx] = []
                        self._N[c_idx] = []

                    assert c_idx is not None

                    if lines[i-1][0] == 'A':
                        self._D[c_idx].append(f_idx)
                        has_premise = True
                        self._T.append(f_idx)
                    if lines[i-1][0] == '+':
                        self._P[c_idx].append(f_idx)
                        has_step = True
                    if lines[i-1][0] == '-':
                        self._N[c_idx].append(f_idx)

            if has_step:
                self._C_step.append(c_idx)
                assert len(self._P[c_idx]) > 0
                assert len(self._N[c_idx]) > 0
            if has_premise:
                self._C_premise.append(c_idx)

            assert has_premise

    def postprocess(
            self,
    ) -> None:
        Log.out("Postprocessing HolStep dataset", {})

        self._kernel.postprocess_compression(4096)

        self._max_length = 0
        for i in range(len(self._formulas)):
            self._formulas[i] = self._kernel.postprocess_formula(
                self._formulas[i],
            )
            if len(self._formulas[i]) > self._max_length:
                self._max_length = len(self._formulas[i])
            if len(self._formulas[i]) > self._kernel._theorem_length:
                Log.out("Excessive length formula", {
                    'length': len(self._formulas[i]),
                })

            if i % 100000 == 0:
                Log.out("Postprocessing HolStep dataset", {
                    'token_count': self._kernel._token_count,
                    'max_length': self._max_length,
                    'formula_count': len(self._formulas),
                    'postprocessed': i,
                })

        Log.out("Postprocessed HolStep dataset", {
            'max_length': self._max_length,
        })


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
        return 2*len(self._hset._C_premise)

    def __getitem__(
            self,
            idx: int,
    ):
        cnj_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        thr_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        pre_t = torch.ones(1)

        cnj = self._hset._C_premise[int(idx/2)]

        thr = random.choice(self._hset._D[cnj])

        if idx % 2 == 1:
            pre_t = torch.zeros(1)
            unr = None
            while(unr is None):
                candidate = random.choice(self._hset._T)
                if candidate not in self._hset._D[cnj]:
                    unr = candidate
            thr = unr

        for i in range(
                min(self._theorem_length, len(self._hset._formulas[cnj]))
        ):
            t = self._hset._formulas[cnj][i]
            assert t != 0
            cnj_t[i] = t
        for i in range(
                min(self._theorem_length, len(self._hset._formulas[thr]))
        ):
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
        return 2*len(self._hset._C_step)

    def __getitem__(
            self,
            idx: int,
    ):
        cnj_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        thr_t = torch.zeros(self._theorem_length, dtype=torch.int64)
        pre_t = torch.ones(1)

        cnj = self._hset._C_step[int(idx/2)]

        thr = random.choice(self._hset._P[cnj])

        if idx % 2 == 1:
            pre_t = torch.zeros(1)
            thr = random.choice(self._hset._N[cnj])

        for i in range(
                min(self._theorem_length, len(self._hset._formulas[cnj]))
        ):
            t = self._hset._formulas[cnj][i]
            assert t != 0
            cnj_t[i] = t
        for i in range(
                min(self._theorem_length, len(self._hset._formulas[thr]))
        ):
            t = self._hset._formulas[thr][i]
            assert t != 0
            thr_t[i] = t

        return cnj_t, thr_t, pre_t


def preprocess():

    kernel = HolStepKernel(512)

    HolStepSet(
        kernel,
        os.path.expanduser("./data/th2vec/holstep/train"),
    )
