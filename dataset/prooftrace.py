import os
import json
import sys

from utils.log import Log


class ProofTraceKernel():
    def __init__(
            self,
            dataset_dir: str,
    ) -> None:
        self._proofs = {}
        self._theorems = {}
        self._names = {}

        # Rewrites to remove INST_TYPE from proofs.
        self._rewrites = {}

        # Proof steps that are re-used >1 time.
        self._shared = {}
        # Terms that are used is proof traces.
        self._terms = {}

        self._dataset_dir = os.path.abspath(dataset_dir)

        Log.out(
            "Loading ProofTrace dataset", {
                'dataset_dir': dataset_dir,
            })

        assert os.path.isdir(dataset_dir)

        self.process_theorems()
        Log.out(
            "Processed ProofTrace theorems", {
            })

        self.process_proofs()
        Log.out(
            "Processed ProofTrace proofs", {
            })

        self.process_names()
        Log.out(
            "Processed ProofTrace names", {
            })

    def process_theorems(
            self,
    ):
        with open(os.path.join(
                self._dataset_dir,
                "prooftrace.theorems",
        ), 'r') as f:
            for line in f:
                data = json.loads(line)
                self._theorems[data['id']] = data['th']

    def process_proofs(
            self,
    ):
        with open(os.path.join(
                self._dataset_dir,
                "prooftrace.proofs",
        ), 'r') as f:
            for line in f:
                data = json.loads(line)

                # Store type instantiations rewrites.
                if data['pr'][0] == 'INST_TYPE':
                    ptr = data['pr'][1]
                    while self._proofs[ptr][0] == 'INST_TYPE':
                        ptr = self._rewrites[ptr]
                    self._rewrites[data['id']] = ptr

                # Apply rewrites to binary operators.
                if data['pr'][0] in [
                        'TRANS', 'MK_COMB', 'EQ_MP', 'DEDUCT_ANTISYM_RULE'
                ]:
                    if data['pr'][1] in self._rewrites:
                        data['pr'][1] = self._rewrites[data['pr'][1]]
                    if data['pr'][2] in self._rewrites:
                        data['pr'][2] = self._rewrites[data['pr'][2]]

                # Apply rewrites to unary operators.
                if data['pr'][0] in [
                        'ABS', 'INST',
                ]:
                    if data['pr'][1] in self._rewrites:
                        data['pr'][1] = self._rewrites[data['pr'][1]]

                self._proofs[data['id']] = data['pr']

    def process_names(
            self,
    ):
        with open(os.path.join(
                self._dataset_dir,
                "prooftrace.names",
        ), 'r') as f:
            for line in f:
                data = json.loads(line)
                self._names[data['id']] = data['nm']

    def add_shared(
            self,
            index,
            theorems,
    ):
        self._shared[index] = theorems

    def add_term(
            self,
            term,
            theorems,
    ):
        self._terms[term] = theorems


class ProofTrace():
    def __init__(
            self,
            kernel: ProofTraceKernel,
            proof_index: int
    ):
        self._kernel = kernel
        self._index = proof_index

        self._terms = {}
        self._steps = {}
        self._premises = {}

        self.walk(self._index)

        Log.out(
            "Constructed ProofTrace", {
                'index': self._index,
                'name': self._kernel._names[self._index],
                'step_count': len(self._steps),
                'terms_count': len(self._terms),
                'premises_count': len(self._premises),
            })

    def record_term(
            self,
            term,
    ):
        if term not in self._terms:
            self._terms[term] = 0
        self._terms[term] += 1

    def record_premise(
            self,
            index,
    ):
        if index not in self._premises:
            self._premises[index] = self._kernel._theorems[index]

    def walk(
            self,
            index,
    ):
        if index in self._steps:
            return

        if index != self._index:
            if (
                    index in self._kernel._names or
                    index in self._kernel._shared
            ):
                self.record_premise(index)
                return

        step = self._kernel._proofs[index]

        if step[0] == 'REFL':
            self.record_term(step[1])

        elif step[0] == 'TRANS':
            self.walk(step[1])
            self.walk(step[2])

        elif step[0] == 'MK_COMB':
            self.walk(step[1])
            self.walk(step[2])

        elif step[0] == 'ABS':
            self.walk(step[1])
            self.record_term(step[2])

        elif step[0] == 'BETA':
            self.record_term(step[1])

        elif step[0] == 'ASSUME':
            self.record_term(step[1])

        elif step[0] == 'EQ_MP':
            self.walk(step[1])
            self.walk(step[2])

        elif step[0] == 'DEDUCT_ANTISYM_RULE':
            self.walk(step[1])
            self.walk(step[2])

        elif step[0] == 'INST':
            self.walk(step[1])
            for inst in step[2]:
                self.record_term(inst[0])
                self.record_term(inst[1])

        elif step[0] == 'INST_TYPE':
            assert False

        elif step[0] == 'AXIOM':
            self.record_premise(index)
            return

        elif step[0] == 'DEFINITION':
            self.record_premise(index)
            return

        elif step[0] == 'TYPE_DEFINITION':
            self.record_premise(index)
            return

        else:
            assert False

        self._steps[index] = step


def extract():
    sys.setrecursionlimit(4096)
    kernel = ProofTraceKernel(
        os.path.expanduser("./data/prooftrace/medium"),
    )

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    cross_steps = {}
    for tr in traces:
        for th in tr._steps.keys():
            if th not in cross_steps:
                cross_steps[th] = []
            if tr._index not in cross_steps[th]:
                cross_steps[th].append(tr._index)

    cross_step_count = 0
    for th in cross_steps:
        if len(cross_steps[th]) > 1:
            cross_step_count += 1
            kernel.add_shared(th, cross_steps[th])

    Log.out("Cross steps detection", {
        "cross_step_count": cross_step_count,
    })

    terms = {}
    for tr in traces:
        for tm in tr._terms.keys():
            if tm not in terms:
                terms[tm] = []
            if tr._index not in terms[tm]:
                terms[tm].append(tr._index)

    for tm in terms:
        kernel.add_term(tm, terms[tm])

    Log.out("Terms aggregation", {
        "term_count": len(terms),
    })

    Log.histogram(
        "ProofTraces Steps",
        [len(pr._steps) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )
    Log.histogram(
        "ProofTraces Terms",
        [len(pr._terms) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )
    Log.histogram(
        "ProofTraces Premises",
        [len(pr._premises) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    Log.histogram(
        "ProofTraces Steps",
        [len(pr._steps) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )
    Log.histogram(
        "ProofTraces Terms",
        [len(pr._terms) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )
    Log.histogram(
        "ProofTraces Premises",
        [len(pr._premises) for pr in traces],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )
