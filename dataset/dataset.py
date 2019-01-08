import concurrent.futures
import os
import random
import threading
import torch

from utils.config import Config
from utils.log import Log

from torch.utils.data import Dataset


class CNF:
    def __init__(
            self,
            config: Config,
            cnf: str,
    ) -> None:
        self._device = torch.device(config.get('device'))

        lines = cnf.splitlines()

        assert lines[0] == 'c SAT' or lines[0] == 'c UNSAT'

        self._sat = (lines[0] == 'c SAT')

        self._clauses = []
        for l in lines:
            if l[0] == 'c':
                pass
            elif l[0] == 'p':
                tokens = l.split(' ')
                assert len(tokens) == 4

                self._variable_count = int(tokens[2])
                self._clause_count = int(tokens[3])
            else:
                clause = [int(v) for v in l.split(' ')[:-1]]
                self._clauses.append(clause)

        assert self._clause_count == len(self._clauses)

    def get_clause_as_word(
            self,
    ):
        clauses = []

        for c in range(self._clause_count):
            cc = []
            for a in self._clauses[c]:
                v = a
                assert v != 0
                if v > 0:
                    v -= 1
                v += self._variable_count
                assert v >= 0 and v < 2*self._variable_count
                cc.append(v)
            clauses.append(cc)

        sat = torch.zeros(1)
        if self._sat:
            sat[0] = 1.0

        return (
            torch.LongTensor(clauses).to(self._device),
            sat.to(self._device),
        )

    def get_variable_as_word(
            self,
            variable_count,
            clause_count,
    ):
        variables_map = random.sample(
            range(0, variable_count),
            self._variable_count,
        )
        clauses_map = random.sample(
            range(0, clause_count),
            self._clause_count,
        )

        variables = torch.zeros(variable_count, clause_count)

        for c in range(self._clause_count):
            for a in self._clauses[c]:
                v = a
                assert v != 0
                truth = True
                if v < 0:
                    v = -v
                    truth = False
                v -= 1

                assert v >= 0
                assert v < self._variable_count
                assert v < variable_count

                if truth:
                    variables[variables_map[v]][clauses_map[c]] = 1.0
                else:
                    variables[variables_map[v]][clauses_map[c]] = -1.0

        sat = torch.zeros(1)
        if self._sat:
            sat[0] = 1.0

        return (
            variables.to(self._device),
            sat.to(self._device),
        )


class SATDataset(Dataset):
    def __init__(
            self,
            config: Config,
            dataset_dir: str,
    ) -> None:
        self._config = config

        Log.out(
            "Loading dataset", {
                'dataset_dir': dataset_dir,
            })

        assert os.path.isdir(dataset_dir)
        files = [
            os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]

        self._variable_count = 0
        self._clause_count = 0

        self._cnfs = []

        def build_cnf(path):
            with open(path, 'r') as f:
                cnf = CNF(config, f.read())
            return cnf

        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for cnf in executor.map(build_cnf, files):
                with lock:
                    if cnf._variable_count > self._variable_count:
                        self._variable_count = cnf._variable_count
                    if cnf._clause_count > self._clause_count:
                        self._clause_count = cnf._clause_count
                    self._cnfs.append(cnf)

        # for p in files:
        #     with open(p, 'r') as f:
        #         cnf = CNF(config, f.read())
        #         if cnf._variable_count > self._variable_count:
        #             self._variable_count = cnf._variable_count
        #         if cnf._clause_count > self._clause_count:
        #             self._clause_count = cnf._clause_count
        #         self._cnfs.append(cnf)

        assert len(self._cnfs) > 0

    def variable_count(
            self,
    ) -> int:
        return self._variable_count

    def clause_count(
            self,
    ) -> int:
        return self._clause_count

    def __len__(
            self,
    ) -> int:
        return len(self._cnfs)

    def __getitem__(
            self,
            idx: int,
    ):
        return self._cnfs[idx].get_variable_as_word(
            self._variable_count,
            self._clause_count,
        )
