import os
import random
import torch

from utils.config import Config
from utils.log import Log

from torch.utils.data import Dataset


class HolStepRelationsDataset(Dataset):
    def __init__(
            self,
            config: Config,
            dataset_dir: str,
    ) -> None:
        self._theorems = []
        self._relations = []

        self._tokens = {}
        self._token_count = 1

        self._bound_count = 0
        self._free_count = 0
        self._skip_count = 0

        self._theorem_length = config.get('th2vec_theorem_length')

        dataset_dir = os.path.expanduser(dataset_dir)
        dataset_dir = os.path.abspath(dataset_dir)

        Log.out(
            "Loading HolStep dataset", {
                'dataset_dir': dataset_dir,
            })

        assert os.path.isdir(dataset_dir)
        files = [
            os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]

        count = 0
        for p in files:
            count += 1
            with open(p, 'r') as f:
                lines = f.read().splitlines()
                theorems = []
                for i in range(len(lines)):
                    line = lines[i]
                    if lines[i-1][0] == '-':
                        continue
                    if line[0] == 'T':
                        th = self.process_tokenized(line[2:])
                        if th is not None:
                            th_idx = len(self._theorems)
                            self._theorems.append(th)
                            theorems.append(th_idx)
                for th_idx in theorems:
                    assert len(self._relations) == th_idx
                    self._relations.append(theorems)

            if count % 100 == 0:
                Log.out(
                    "Processing HolStep dataset", {
                        'dataset_dir': dataset_dir,
                        'token_count': self._token_count,
                        'bound_count': self._bound_count,
                        'free_count': self._free_count,
                        'skip_count': self._skip_count,
                        'theorem_count': len(self._theorems),
                        'processed': count,
                    })

        Log.out(
            "Loaded HolStep dataset", {
                'dataset_dir': dataset_dir,
                'token_count': self._token_count,
                'bound_count': self._bound_count,
                'free_count': self._free_count,
                'skip_count': self._skip_count,
                'theorem_count': len(self._theorems),
            })

    def process_tokenized(
            self,
            th: str,
    ) -> None:
        theorem = []

        tokens = th.split(' ')

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
            theorem.append(self._tokens[t])

        return theorem

    def __len__(
            self,
    ) -> int:
        return len(self._relations)

    def __getitem__(
            self,
            idx: int,
    ):
        inp = torch.zeros(self._theorem_length, dtype=torch.int64)
        out = torch.zeros(self._theorem_length, dtype=torch.int64)
        rnd = torch.zeros(self._theorem_length, dtype=torch.int64)

        rel = random.choice(self._relations[idx])
        unr = random.choice(range(len(self._theorems)))

        for i in range(len(self._theorems[idx])):
            t = self._theorems[idx][i]
            assert t != 0
            inp[i] = t
        for i in range(len(self._theorems[rel])):
            t = self._theorems[rel][i]
            assert t != 0
            out[i] = t
        for i in range(len(self._theorems[unr])):
            t = self._theorems[unr][i]
            assert t != 0
            rnd[i] = t

        return inp, out, rnd


# class HolStepStepDataset(Dataset):
#     def __init__(
#             self,
#             config: Config,
#             dataset_dir: str,
#     ) -> None:
#         self._steps = []
#
#         self._tokens = {}
#
#         self._token_count = 1
#         self._bound_count = 0
#         self._free_count = 0
#         self._skip_count = 0
#
#         self._theorem_length = config.get('th2vec_theorem_length')
#
#         dataset_dir = os.path.expanduser(dataset_dir)
#         dataset_dir = os.path.abspath(dataset_dir)
#
#         Log.out(
#             "Loading HolStep dataset", {
#                 'dataset_dir': dataset_dir,
#             })
#
#         assert os.path.isdir(dataset_dir)
#         files = [
#             os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
#             if os.path.isfile(os.path.join(dataset_dir, f))
#         ]
#
#         count = 0
#         for p in files:
#             count += 1
#             with open(p, 'r') as f:
#                 lines = f.read().splitlines()
#                 for l in lines:
#                     if l[0] == 'T':
#                         self.process_tokenized(l[2:])
#             if count % 100 == 0:
#                 Log.out(
#                     "Processing HolStep dataset", {
#                         'dataset_dir': dataset_dir,
#                         'token_count': self._token_count,
#                         'bound_count': self._bound_count,
#                         'free_count': self._free_count,
#                         'step_count': len(self._steps),
#                         'skip_count': self._skip_count,
#                         'processed': count,
#                     })
#
#         Log.out(
#             "Loaded HolStep dataset", {
#                 'dataset_dir': dataset_dir,
#                 'token_count': self._token_count,
#                 'bound_count': self._bound_count,
#                 'free_count': self._free_count,
#                 'step_count': len(self._steps),
#                 'skip_count': self._skip_count,
#             })
#
#     def process_tokenized(
#             self,
#             th: str,
#     ) -> None:
#         theorem = []
#
#         tokens = th.split(' ')
#
#         if len(tokens) > self._theorem_length:
#             self._skip_count += 1
#             return
#
#         for t in tokens:
#             if t[0] == 'b':
#                 b = int(t[1:])
#                 if b > self._bound_count:
#                     self._bound_count = b
#             elif t[0] == 'f':
#                 f = int(t[1:])
#                 if f > self._free_count:
#                     self._free_count = f
#             if t not in self._tokens:
#                 self._tokens[t] = self._token_count
#                 self._token_count += 1
#             theorem.append(self._tokens[t])
#
#         self._steps.append(theorem)
#
#     def __len__(
#             self,
#     ) -> int:
#         return len(self._steps)
#
#     def __getitem__(
#             self,
#             idx: int,
#     ):
#         step = torch.zeros(self._theorem_length_max, dtype=torch.int64)
#
#         for t in self._steps[idx]:
#             assert t != 0
#             step[idx] = t
#
#         return self._steps[idx]
