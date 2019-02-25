import argparse
import os
import json
import pickle
import re
import shutil
import sys
import typing

from generic.tree_lstm import BVT

from torch.utils.data import Dataset

from utils.config import Config
from utils.log import Log

ACTION_TOKENS = {
    'EMPTY': 0,
    'EXTRACT': 1,
    'PREMISE': 2,
    'HYPOTHESIS': 3,
    'CONCLUSION': 4,
    'TERM': 5,
    'REFL': 6,
    'TRANS': 7,
    'MK_COMB': 8,
    'ABS': 9,
    'BETA': 10,
    'ASSUME': 11,
    'EQ_MP': 12,
    'DEDUCT_ANTISYM_RULE': 13,
    'INST': 14,
    'INST_PAIR': 15,
}


class Term(BVT):
    pass


class Action(BVT):
    @staticmethod
    def from_action(
            action: str,
            left=None,
            right=None,
    ):
        value = ACTION_TOKENS[action]
        return Action(value, left, right)

    @staticmethod
    def from_term(
            term: Term,
    ):
        return Action(term)


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

                # Store type instantiations (or empty instantiations) rewrites.
                if data['pr'][0] == 'INST_TYPE' or (
                        data['pr'][0] == 'INST' and len(data['pr'][2]) == 0
                ):
                    ptr = data['pr'][1]
                    while ptr in self._rewrites:
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
            proof_index: int,
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
            action = Action.from_action(
                'TERM',
                Action.from_term(self._kernel.term(tm)),
            )
            cache['terms'][tm] = action
            sequence.append(action)

        # Terms are unordered so we order them by depth to optimize cache hit
        # again.
        sequence = sorted(
            sequence,
            key=lambda action: action.left.value.depth(),
            reverse=True,
        )

        for idx in self._premises:
            p = self._premises[idx]

            def build_hypothesis(hypotheses):
                if len(hypotheses) == 0:
                    return None
                else:
                    return Action.from_action(
                        'HYPOTHESIS',
                        Action.from_term(self._kernel.term(hypotheses[0])),
                        build_hypothesis(hypotheses[1:]),
                    )

            action = Action.from_action(
                'PREMISE',
                Action.from_action(
                    'CONCLUSION',
                    Action.from_term(self._kernel.term(p['cc'])),
                    build_hypothesis(p['hy']),
                ),
            )
            cache['indices'][idx] = action
            sequence.append(action)

        for idx in self._sequence:
            step = self._steps[idx]
            actions = []

            if step[0] == 'REFL':
                actions = [
                    Action.from_action(
                        'REFL',
                        cache['terms'][step[1]],
                    ),
                ]
            elif step[0] == 'TRANS':
                actions = [
                    Action.from_action(
                        'TRANS',
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ),
                ]
            elif step[0] == 'MK_COMB':
                actions = [
                    Action.from_action(
                        'MK_COMB',
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ),
                ]
            elif step[0] == 'ABS':
                actions = [
                    Action.from_action(
                        'ABS',
                        cache['indices'][step[1]],
                        cache['terms'][step[2]],
                    ),
                ]
            elif step[0] == 'BETA':
                actions = [
                    Action.from_action(
                        'BETA',
                        cache['terms'][step[1]],
                    ),
                ]
            elif step[0] == 'ASSUME':
                actions = [
                    Action.from_action(
                        'ASSUME',
                        cache['terms'][step[1]],
                    ),
                ]
            elif step[0] == 'EQ_MP':
                actions = [
                    Action.from_action(
                        'EQ_MP',
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ),
                ]
            elif step[0] == 'DEDUCT_ANTISYM_RULE':
                actions = [
                    Action.from_action(
                        'DEDUCT_ANTISYM_RULE',
                        cache['indices'][step[1]],
                        cache['indices'][step[2]],
                    ),
                ]
            elif step[0] == 'INST':
                pred = cache['indices'][step[1]]
                actions = []
                for inst in step[2]:
                    action = Action.from_action(
                        'INST',
                        pred,
                        Action.from_action(
                            'INST_PAIR',
                            cache['terms'][inst[0]],
                            cache['terms'][inst[1]],
                        ),
                    )
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

        # sequence.append(Action.from_action('EXTRACT'))

        return sequence


class ProofTraceDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            trace_max_length=-1,
    ) -> None:
        self._traces = []

        assert os.path.isdir(dataset_dir)
        files = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]

        for p in files:
            if re.search("\\.actions$", p) is None:
                continue
            with open(p, 'rb') as f:
                trace = pickle.load(f)
                if trace_max_length <= -1 or len(trace) <= trace_max_length:
                    self._traces.append(trace)

        Log.out(
            "Loaded extracted ProofTraces", {
                'dataset_dir': dataset_dir,
                'trace_max_length': trace_max_length,
                'traces_count': len(self._traces),
            })

    def __len__(
            self,
    ) -> int:
        return len(self._traces)

    def __getitem__(
            self,
            idx: int,
    ):
        return self._traces[idx]


class ProofTraceLMDataset(ProofTraceDataset):
    def __init__(
            self,
            dataset_dir: str,
            sequence_length: int,
            trace_max_length=-1,
    ) -> None:
        self._sequence_length = sequence_length
        self._cases = []

        super(ProofTraceLMDataset, self).__init__(
            dataset_dir,
            trace_max_length,
        )

        for idx, tr in enumerate(self._traces):
            for pos in range(len(tr)):
                if pos < (self._sequence_length - 1):
                    if tr[pos].value not in \
                            [ACTION_TOKENS['TERM'], ACTION_TOKENS['PREMISE']]:
                        self._cases.append((idx, pos))

        Log.out(
            "Loaded extracted ProofTraces LM Dataset", {
                'dataset_dir': dataset_dir,
                'cases': len(self._cases),
            })

    def __len__(
            self,
    ) -> int:
        return len(self._cases)

    def __getitem__(
            self,
            idx: int,
    ):
        trace = self._traces[self._cases[idx][0]][:self._cases[idx][1]]

        trace.append(Action.from_action('EXTRACT'))

        while len(trace) < self._sequence_length:
            trace.append(Action.from_action('EMPTY'))

        return trace


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
    traces = sorted(traces, key=lambda tr: tr._index)

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

    actions = [tr.actions() for tr in traces]

    Log.histogram(
        "ProofTraces Length",
        [len(a) for a in actions],
        buckets=[1e1, 1e2, 1e3, 2e3, 4e3, 1e4],
        labels=["1e1", "1e2", "1e3", "2e3", "4e3", "1e4"]
    )

    traces_path_train = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        "train_traces",
    )
    traces_path_test = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        "test_traces",
    )

    if os.path.isdir(traces_path_train):
        shutil.rmtree(traces_path_train)
    os.mkdir(traces_path_train)
    if os.path.isdir(traces_path_test):
        shutil.rmtree(traces_path_test)
    os.mkdir(traces_path_test)

    train_size = int(len(traces) * 90 / 100)

    for i, tr in enumerate(traces):
        if i < train_size:
            path = traces_path_train
        else:
            path = traces_path_test
        actions_path = os.path.join(path, tr.name() + '.actions')
        with open(actions_path, 'wb') as f:
            pickle.dump(actions[i], f)

        trace_path = os.path.join(path, tr.name() + '.trace')
        with open(trace_path, 'w') as f:
            json.dump(dict(tr), f, sort_keys=False, indent=2)

    Log.out("Dumped all traces", {
        "traces_path_train": traces_path_train,
        "traces_path_test": traces_path_test,
        "trace_count": len(traces),
        "train_size": train_size,
        "term_token_count": len(kernel._term_tokens),
    })

    # small term_token_count: 400
    # medium term_token_count: 13404
