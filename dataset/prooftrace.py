import argparse
import os
import json
import pickle
import re
import shutil
import sys
import typing
import xxhash

from generic.tree_lstm import BVT

from torch.utils.data import Dataset

from utils.config import Config
from utils.log import Log

ACTION_TOKENS = {
    'EMPTY': 0,
    'TARGET': 1,
    'EXTRACT': 2,
    'PREMISE': 3,
    'HYPOTHESIS': 4,
    'CONCLUSION': 5,
    'SUBST': 6,
    'SUBST_PAIR': 7,
    'TERM': 8,
    'REFL': 9,
    'TRANS': 10,
    'MK_COMB': 11,
    'ABS': 12,
    'BETA': 13,
    'ASSUME': 14,
    'EQ_MP': 15,
    'DEDUCT_ANTISYM_RULE': 16,
    'INST': 17,
}


class Term(BVT):
    def __init__(
            self,
            value,
            left,
            right,
            token: str,
    ):
        super(Term, self).__init__(
            value, left, right
        )
        # `self._token` stores the associated string token so that we can
        # reconstruct term strings directly from their BVT.
        self._token = token

    def token(
            self,
    ) -> str:
        return self._token

    def term_string(
            self,
    ) -> str:
        """ `term_string` formats the Term BVT as a HOL Light term string
        """
        def dump(term, args):
            if term.token() == '__C':
                right = dump(term.right, [])
                return dump(term.left, [right] + args)
            if term.token() == '__A':
                assert len(args) <= 1
                right = dump(term.right, [])
                left = dump(term.left, [])
                if len(args) == 0:
                    return '\\' + left + '. ' + right
                else:
                    return '(\\' + left + '. ' + right + ') (' + args[0] + ')'
            if term.token() == '__c' or term.token() == '__v':
                assert term.right is None
                token = term.left.token()[1:]
                if token == '_comma_':
                    token = ','
                if len(args) == 0:
                    return token
                else:
                    tm = '((' + token + ')'
                    for a in args:
                        tm += ' (' + a + ')'
                    tm += ')'
                    return tm

        return dump(self, [])


class Action(BVT):
    def __init__(
            self,
            value,
            left=None,
            right=None,
            index: int = None,
    ):
        super(Action, self).__init__(
            value, left, right
        )
        # `self._index` stores the original index of the associated action.
        # It's used solely for PREMISE to store their index in order to be
        # able to retrieve the associated theorem (through the proof) in the
        # HOL Light environment as we create a prooftrace REPL environment.
        self._index = index

    def index(
            self,
    ) -> int:
        return self._index

    @staticmethod
    def from_action(
            action: str,
            left,
            right,
            origin=None,
    ):
        value = ACTION_TOKENS[action]
        return Action(value, left, right, origin)

    @staticmethod
    def from_term(
            term: Term,
    ):
        return Action(term)


class ProofTraceKernel():
    def __init__(
            self,
            dataset_dir: str,
            dataset_size: str,
    ) -> None:
        self._proofs = {}
        self._theorems = {}
        self._names = {}

        # Rewrites to remove INST_TYPE from proofs.
        self._rewrites = {}

        # Proof steps that are re-used >1 time.
        self._shared = {}

        self._term_tokens = {
            '__C': 0,
            '__A': 1,
            '__c': 2,
            '__v': 3,
        }
        self._term_cache = {}

        self._dataset_dir = os.path.abspath(
            os.path.join(dataset_dir, dataset_size),
        )

        Log.out(
            "Loading ProofTrace dataset", {
                'dataset_dir': self._dataset_dir,
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

    def name_shared_premise(
            self,
            index,
    ) -> bool:
        if index not in self._names:
            # Log.out("SHARED", {
            #     "index": index,
            # })
            self._names[index] = "SHARED_" + str(index)

            return True

        return False

    def term(
            self,
            tm: str,
    ) -> Term:
        """ Construct a Term BVT from a term string.

        Tokenizes constants appearing in terms using self._term_tokens.
        """
        if tm in self._term_cache:
            return self._term_cache[tm]

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
                return Term(
                    self._term_tokens['__C'],
                    construct(chld[0]),
                    construct(chld[1]),
                    '__C',
                )
            if t[0] == 'A':
                chld = list(split(t))
                return Term(
                    self._term_tokens['__A'],
                    construct(chld[0]),
                    construct(chld[1]),
                    '__A',
                )
            if t[0] == 'c':
                if t not in self._term_tokens:
                    self._term_tokens[t] = len(self._term_tokens)
                return Term(
                    self._term_tokens['__c'],
                    Term(self._term_tokens[t], None, None, t),
                    None,
                    '__c',
                )
            if t[0] == 'v':
                if t not in self._term_tokens:
                    self._term_tokens[t] = len(self._term_tokens)
                return Term(
                    self._term_tokens['__v'],
                    Term(self._term_tokens[t], None, None, t),
                    None,
                    '__v',
                )

        term = construct(tm)
        self._term_cache[tm] = term

        return term


class ProofTraceActions():
    def __init__(
            self,
            name: str,
            actions: typing.List[Action],
    ) -> None:
        self._name = name
        self._actions = actions

    def dump(
            self,
            path,
    ) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def len(
            self,
    ) -> int:
        return len(self._actions)

    def name(
            self,
    ) -> str:
        return self._name

    def actions(
            self,
    ) -> typing.List[Action]:
        return self._actions


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
        self._substs = {}

        self._steps = {}
        self._sequence = []

        self.walk(self._index)

        # Log.out(
        #     "Constructed ProofTrace", {
        #         'index': self._index,
        #         'name': self.name(),
        #         'premises_count': len(self._premises),
        #         'step_count': len(self._steps),
        #         'terms_count': len(self._terms),
        #         'substs_count': len(self._substs),
        #     })

    def name(
            self,
    ):
        return str(self._index) + '_' + self._kernel._names[self._index]

    def term_hash(
            self,
            term,
    ):
        h = xxhash.xxh64()
        h.update(term)
        return h.digest()

    def subst_hash(
            self,
            subst,
    ):
        h = xxhash.xxh64()
        for s in subst:
            assert len(s) == 2
            h.update(s[0])
            h.update(s[1])
        return h.digest()

    def record_term(
            self,
            term,
    ):
        h = self.term_hash(term)
        if h in self._terms:
            assert term == self._terms[h]
        else:
            self._terms[h] = term

    def record_subst(
            self,
            subst,
    ):
        h = self.subst_hash(subst)
        if h in self._substs:
            assert len(subst) == len(self._substs[h])
            for i, s in enumerate(subst):
                assert s[0] == self._substs[h][i][0]
                assert s[1] == self._substs[h][i][1]
        else:
            self._substs[h] = subst

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
            self.record_subst(step[2])

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
        yield 'target', self._kernel._theorems[self._index]
        yield 'terms', list(self._terms.values())
        yield 'substs', list(self._substs.values())
        yield 'premises', self._premises
        yield 'steps', self._steps

    def actions(
            self,
    ) -> ProofTraceActions:
        sequence = []

        cache = {
            'terms': {},
            'substs': {},
            'indices': {},
        }

        # Empty is used by unary actions as right argument, it lives at the
        # start of the sequence after the target (index 1). We can't use None
        # since the language model loss needs an index to use as right
        # arguments even for right arguments of unary actions.
        empty = Action.from_action('EMPTY', None, None)

        # Recursive function used to build theorems hypotheses used for TARGET
        # and PREMISE actions.
        def build_hypothesis(hypotheses):
            if len(hypotheses) == 0:
                return None
            else:
                return Action.from_action(
                    'HYPOTHESIS',
                    Action.from_term(self._kernel.term(hypotheses[0])),
                    build_hypothesis(hypotheses[1:]),
                )

        # Recursive function used to build instantiations substitutions
        def build_subst(subst):
            if len(subst) == 0:
                return None
            else:
                return Action.from_action(
                    'SUBST',
                    Action.from_action(
                        'SUBST_PAIR',
                        Action.from_term(self._kernel.term(subst[0][0])),
                        Action.from_term(self._kernel.term(subst[0][1])),
                    ),
                    build_subst(subst[1:]),
                )

        # Start by recording the target theorem (TARGET action).
        t = self._kernel._theorems[self._index]

        target = Action.from_action(
            'TARGET',
            Action.from_action(
                'CONCLUSION',
                Action.from_term(self._kernel.term(t['cc'])),
                build_hypothesis(t['hy']),
            ),
            None,
        )

        substs = []
        # We first record subst as they are generally deeper than terms but
        # include very similar terms (optimize TreeLSTM cache hit).
        for subst in self._substs.values():
            action = build_subst(subst)
            cache['substs'][self.subst_hash(subst)] = action
            substs.append(action)

        terms = []
        # We then record terms as they are generally deeper than premises but
        # include very similar terms (optimize TreeLSTM cache hit).
        for term in self._terms.values():
            action = Action.from_action(
                'TERM',
                Action.from_term(self._kernel.term(term)),
                None,
            )
            cache['terms'][self.term_hash(term)] = action
            terms.append(action)

        # Terms are unordered so we order them by depth to optimize cache hit
        # again.
        terms = sorted(
            terms,
            key=lambda action: action.left.value.depth(),
            reverse=True,
        )

        sequence = [target, empty] + substs + terms

        for idx in self._premises:
            p = self._premises[idx]

            action = Action.from_action(
                'PREMISE',
                Action.from_action(
                    'CONCLUSION',
                    Action.from_term(self._kernel.term(p['cc'])),
                    build_hypothesis(p['hy']),
                ),
                None,
                idx,
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
                        cache['terms'][self.term_hash(step[1])],
                        empty,
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
                        cache['terms'][self.term_hash(step[2])],
                    ),
                ]
            elif step[0] == 'BETA':
                actions = [
                    Action.from_action(
                        'BETA',
                        cache['terms'][self.term_hash(step[1])],
                        empty,
                    ),
                ]
            elif step[0] == 'ASSUME':
                actions = [
                    Action.from_action(
                        'ASSUME',
                        cache['terms'][self.term_hash(step[1])],
                        empty,
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
                actions = [
                    Action.from_action(
                        'INST',
                        cache['indices'][step[1]],
                        cache['substs'][self.subst_hash(step[2])]
                    ),
                ]

            elif step[0] == 'INST_TYPE':
                assert False

            elif step[0] == 'AXIOM':
                assert False

            elif step[0] == 'DEFINITION':
                assert False

            elif step[0] == 'TYPE_DEFINITION':
                assert False

            assert len(actions) == 1
            cache['indices'][idx] = actions[-1]

            sequence = sequence + actions

        return ProofTraceActions(
            self.name(),
            sequence
        )


class ProofTraceDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            dataset_size: str,
            test: bool,
            trace_max_length=-1,
    ) -> None:
        self._traces = []

        if test:
            dataset_dir = os.path.join(
                dataset_dir, dataset_size, 'test_traces'
            )
        else:
            dataset_dir = os.path.join(
                dataset_dir, dataset_size, 'train_traces'
            )

        assert os.path.isdir(dataset_dir)
        files = [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if os.path.isfile(os.path.join(dataset_dir, f))
        ]

        processed = 0
        for p in files:
            if re.search("\\.actions$", p) is None:
                continue
            with open(p, 'rb') as f:
                trace = pickle.load(f)
                if trace_max_length <= -1 or trace.len() <= trace_max_length:
                    self._traces.append(trace)

            processed += 1

            if processed % 100 == 0:
                Log.out(
                    "Loading extracted ProofTraces", {
                        'dataset_dir': dataset_dir,
                        'total': len(files),
                        'processed': processed,
                    })

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
        return self._traces[idx].actions()


class ProofTraceLMDataset(ProofTraceDataset):
    def __init__(
            self,
            dataset_dir: str,
            dataset_size: str,
            sequence_length: int,
            trace_max_length=-1,
    ) -> None:
        self._sequence_length = sequence_length
        self._cases = []

        super(ProofTraceLMDataset, self).__init__(
            dataset_dir,
            dataset_size,
            trace_max_length,
        )

        for idx, tr in enumerate(self._traces):
            actions = tr.actions()
            for pos in range(len(actions)):
                if pos < self._sequence_length:
                    if actions[pos].value not in \
                            [
                                ACTION_TOKENS['TARGET'],
                                ACTION_TOKENS['EMPTY'],
                                ACTION_TOKENS['SUBST'],
                                ACTION_TOKENS['TERM'],
                                ACTION_TOKENS['PREMISE'],
                            ]:
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
        trace = self._traces[self._cases[idx][0]].actions()[
            :self._cases[idx][1]+1
        ]

        while len(trace) < self._sequence_length:
            trace.append(Action.from_action('EMPTY', None, None))

        return (self._cases[idx][1], trace)


def lm_collate(batch):
    indices = []
    traces = []

    for (idx, trc) in batch:
        indices.append(idx)
        traces.append(trc)

    return (indices, traces)


def extract():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    sys.setrecursionlimit(4096)
    kernel = ProofTraceKernel(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
    )

    Log.out("Starting cross steps detection")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    Log.out("Prooftraces computed", {
        "traces_count": len(traces),
    })

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

    Log.out("Starting shared premises detection")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    Log.out("Prooftraces computed", {
        "traces_count": len(traces),
    })

    shared_premise_count = 0
    for tr in traces:
        for th in tr._premises.keys():
            if kernel.name_shared_premise(th):
                shared_premise_count += 1

    Log.out("Shared premises detection", {
        "shared_premise_count": shared_premise_count,
    })

    Log.out("Starting final prooftraces generation")

    kernel._shared = {}
    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]
    traces = [tr for tr in traces if len(tr._steps) > 0]
    traces = sorted(traces, key=lambda tr: tr._index)

    Log.out("Prooftraces computed, filtered and sorted", {
        "traces_count": len(traces),
    })

    Log.histogram(
        "ProofTraces Premises",
        [len(pr._premises) for pr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Substs",
        [len(pr._substs) for pr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Terms",
        [len(pr._terms) for pr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Steps",
        [len(pr._steps) for pr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.out("Starting action generation")

    trace_actions = [tr.actions() for tr in traces]

    Log.histogram(
        "ProofTraces Length",
        [a.len() for a in trace_actions],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )

    traces_path_train = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "train_traces",
    )
    traces_path_test = os.path.join(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        "test_traces",
    )

    if os.path.isdir(traces_path_train):
        shutil.rmtree(traces_path_train)
    os.mkdir(traces_path_train)
    if os.path.isdir(traces_path_test):
        shutil.rmtree(traces_path_test)
    os.mkdir(traces_path_test)

    train_size = int(len(traces) * 90 / 100)

    for i, ptra in enumerate(trace_actions):
        tr = traces[i]
        if i < train_size:
            path = traces_path_train
        else:
            path = traces_path_test
        actions_path = os.path.join(path, tr.name() + '.actions')
        with open(actions_path, 'wb') as f:
            pickle.dump(ptra, f)

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


def dump_shared():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="congif override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    sys.setrecursionlimit(4096)
    kernel = ProofTraceKernel(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
    )

    Log.out("Starting cross steps detection")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    Log.out("Prooftraces computed", {
        "traces_count": len(traces),
    })

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

    Log.out("Starting shared premises detection")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]

    Log.out("Prooftraces computed", {
        "traces_count": len(traces),
    })

    shared = {}

    for tr in traces:
        for th in tr._premises.keys():
            if kernel.name_shared_premise(th):
                shared[th] = 1
            elif th in shared:
                shared[th] += 1

    keys = sorted(shared.keys(), key=lambda k: shared[k], reverse=True)

    for idx in keys:
        dump = "=========\n"
        dump += str(shared[idx]) + " [" + str(idx) + "]\n"

        th = kernel._theorems[idx]

        for h in th['hy']:
            dump += kernel.term(h).term_string() + '\n'
        dump += '|-\n'
        dump += kernel.term(th['cc']).term_string() + '\n'
        # dump += kernel._proofs[idx][0] + '\n'
        dump += "---------"
        print(dump)
