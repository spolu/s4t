import argparse
import os
import json
# import shutil
import sys
import typing
import xxhash

from generic.tree_lstm import BVT

from utils.config import Config
from utils.log import Log

ACTION_TOKENS = {
    'EMPTY': 0,
    'PREMISE': 1,
    'HYPOTHESIS': 2,
    'CONCLUSION': 3,
    'TERM': 4,
    'REFL': 5,
    'TRANS': 6,
    'MK_COMB': 7,
    'ABS': 8,
    'BETA': 9,
    'ASSUME': 10,
    'EQ_MP': 11,
    'DEDUCT_ANTISYM_RULE': 12,
    'INST': 13,
}


class Term(BVT):
    pass


class Action():
    def __init__(
            self,
            action: str,
            args,
    ):
        self.action = ACTION_TOKENS[action]
        self.args = args

        self._hash = None
        self._depth = None

    def hash(
            self,
    ):
        if self._hash is None:
            h = xxhash.xxh64()
            h.update(str(self.action))
            for arg in self.args:
                h.update(arg.hash())
            self._hash = h.digest()

        return self._hash

    def depth(
            self,
    ):
        if self._depth is None:
            self._depth = 1 + max(
                [arg.depth() for arg in self.args] + [0]
            )

        return self._depth


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

        self._term_tokens = {
            '__C': 0,
            '__A': 1,
            '__c': 2,
            '__v': 3,
        }

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

    def term(
            self,
            tm: str,
    ) -> Term:
        """ Construct a term BVT from a term string.

        Tokenizes constants appearing in terms using self._term_tokens.
        """
        def split(t):
            stack = []
            for i, c in enumerate(t):
                if c == '(':
                    stack.append(i)
                elif c == ',':
                    start = stack.pop()
                    if len(stack) == 0:
                        yield t[start + 1: i]
                    stack.append(i)
                elif c == ')' and stack:
                    start = stack.pop()
                    if len(stack) == 0:
                        yield t[start + 1: i]

        def construct(t):
            if t[0] == 'C':
                chld = list(split(t))
                return BVT(
                    self._term_tokens['__C'],
                    construct(chld[0]),
                    construct(chld[1]),
                )
            if t[0] == 'A':
                chld = list(split(t))
                return BVT(
                    self._term_tokens['__A'],
                    construct(chld[0]),
                    construct(chld[1]),
                )
            if t[0] == 'c':
                if t not in self._term_tokens:
                    self._term_tokens[t] = len(self._term_tokens)
                return BVT(self._term_tokens['__c'], BVT(self._term_tokens[t]))
            if t[0] == 'v':
                if t not in self._term_tokens:
                    self._term_tokens[t] = len(self._term_tokens)
                return BVT(self._term_tokens['__v'], BVT(self._term_tokens[t]))

        return construct(tm)


class ProofTrace():
    def __init__(
            self,
            kernel: ProofTraceKernel,
            proof_index: int
    ):
        self._kernel = kernel
        self._index = proof_index

        self._premises = {}
        self._terms = {}

        self._steps = {}
        self._sequence = []

        self.walk(self._index)

        Log.out(
            "Constructed ProofTrace", {
                'index': self._index,
                'name': self.name(),
                'premises_count': len(self._premises),
                'step_count': len(self._steps),
                'terms_count': len(self._terms),
            })

    def name(
            self,
    ):
        return str(self._index) + '_' + self._kernel._names[self._index]

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
        self._sequence.append(index)

    def __iter__(
            self,
    ):
        yield 'index', self._index
        yield 'terms', list(self._terms.keys())
        yield 'premises', self._premises
        yield 'steps', self._steps

    def actions(
            self,
    ) -> typing.List[Action]:
        sequence = []

        cache = {
            'terms': {},
            'indices': {},
        }

        # We start by terms as they are generally deeper than premises but
        # include very similar terms (optimize TreeLSTM cache hit).
        for tm in self._terms:
            action = Action('TERM', [self._kernel.term(tm)])
            cache['terms'][tm] = action
            sequence.append(action)

        # Terms are unordered so we order them by depth to optimize cache hit
        # again.
        sequence = sorted(
            sequence, key=lambda action: action.depth(), reverse=True,
        )

        for idx in self._premises:
            p = self._premises[idx]
            action = Action(
                'PREMISE',
                [Action('CONCLUSION', [self._kernel.term(p['cc'])])] +
                [Action('HYPOTHESIS', [self._kernel.term(h)]) for h in p['hy']]
            )
            cache['indices'][idx] = action
            sequence.append(action)

        for idx in self._sequence:
            step = self._steps[idx]
            actions = []

            if step[0] == 'REFL':
                actions = [
                    Action('REFL', [
                        cache['terms'][step[1]],
                    ])
                ]
            elif step[0] == 'TRANS':
                actions = [
                    Action('TRANS', [
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ])
                ]
            elif step[0] == 'MK_COMB':
                actions = [
                    Action('MK_COMB', [
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ])
                ]
            elif step[0] == 'ABS':
                actions = [
                    Action('ABS', [
                        cache['indices'][step[1]],
                        cache['terms'][step[2]],
                    ])
                ]
            elif step[0] == 'BETA':
                actions = [
                    Action('BETA', [
                        cache['terms'][step[1]],
                    ])
                ]
            elif step[0] == 'ASSUME':
                actions = [
                    Action('ASSUME', [
                        cache['terms'][step[1]],
                    ])
                ]
            elif step[0] == 'EQ_MP':
                actions = [
                    Action('EQ_MP', [
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ])
                ]
            elif step[0] == 'DEDUCT_ANTISYM_RULE':
                actions = [
                    Action('DEDUCT_ANTISYM_RULE', [
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ])
                ]
            elif step[0] == 'INST':
                pred = cache['indices'][step[1]]
                actions = []
                for inst in step[2]:
                    action = Action('INST', [
                        pred,
                        cache['terms'][inst[0]],
                        cache['terms'][inst[1]],
                    ])
                    pred = action
                    actions.append(action)

            elif step[0] == 'INST_TYPE':
                assert False

            elif step[0] == 'AXIOM':
                assert False

            elif step[0] == 'DEFINITION':
                assert False

            elif step[0] == 'TYPE_DEFINITION':
                assert False

            cache['indices'][idx] = actions[-1]
            sequence = sequence + actions

        return sequence


def extract():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_dir',
        type=str, help="prooftrace dataset directory",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_dir is not None:
        config.override(
            'prooftrace_dataset_dir',
            os.path.expanduser(args.dataset_dir),
        )

    sys.setrecursionlimit(4096)
    kernel = ProofTraceKernel(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
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

    # terms = {}
    # for tr in traces:
    #     for tm in tr._terms.keys():
    #         if tm not in terms:
    #             terms[tm] = []
    #         if tr._index not in terms[tm]:
    #             terms[tm].append(tr._index)

    # for tm in terms:
    #     kernel.add_term(tm, terms[tm])

    # Log.out("Terms aggregation", {
    #     "term_count": len(terms),
    # })

    actions = traces[0].actions()
    import pdb; pdb.set_trace()

    # traces_path = os.path.join(
    #     os.path.expanduser(config.get('prooftrace_dataset_dir')),
    #     "traces",
    # )

    # if os.path.isdir(traces_path):
    #     shutil.rmtree(traces_path)
    # os.mkdir(traces_path)

    # for tr in traces:
    #     trace_path = os.path.join(traces_path, tr.name())
    #     with open(trace_path, 'w') as f:
    #         json.dump(dict(tr), f, sort_keys=False, indent=2)

    # Log.out("Dumped all traces", {
    #     "traces_path": traces_path,
    #     "trace_count": len(traces),
    # })
