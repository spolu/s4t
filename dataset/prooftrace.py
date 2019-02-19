import os
import json

from utils.log import Log

class ProofTraceKernel():
    def __init__(
            self,
            dataset_dir: str,
    ) -> None:
        self._proofs = {}
        self._theorems = {}
        self._names = {}

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

    def record_term(
            self,
            term,
    ):
        if term not in self._terms:
            self._terms[term] = 0
        self._terms[term] += 1

    def walk(
            self,
            index,
    ):
        step = self._kernel.proofs[index]

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
            self.record_term(step[1])

        elif step[0] == 'BETA':
            pass
        elif step[0] == 'ASSUME':
            pass
        elif step[0] == 'EQ_MP':
            pass
        elif step[0] == 'DEDUCT_ANTISYM_RULE':
            pass
        elif step[0] == 'INST':
            pass
        elif step[0] == 'INST_TYPE':
            pass
        elif step[0] == 'AXIOM':
            pass
        elif step[0] == 'DEFINITION':
            pass
        elif step[0] == 'TYPE_DEFINITION':
            pass
        else:
            assert False





        self._steps[index] = step



def extract():
    kernel = ProofTraceKernel(
        os.path.expanduser("./data/prooftrace"),
    )
    import pdb; pdb.set_trace()
