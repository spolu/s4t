import argparse
import base64
import concurrent.futures
import copy
import gzip
import json
import os
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

TEST_FILTER = [
    'IRRATIONAL_SQRT_NONSQUARE',
    'IRRATIONAL_SQRT_PRIME',
    'IRRATIONAL_SQRT_2',
    'PAIR_EXISTS_THM',
]

ACTION_TOKENS = {
    'EMPTY': 0,
    'EXTRACT': 1,
    'THEOREM': 2,
    'HYPOTHESIS': 3,
    'SUBST': 4,
    'SUBST_TYPE': 5,
    'SUBST_PAIR': 6,
    'TERM': 7,
    'REFL': 8,
    'TRANS': 9,
    'MK_COMB': 10,
    'ABS': 11,
    'BETA': 12,
    'ASSUME': 13,
    'EQ_MP': 14,
    'DEDUCT_ANTISYM_RULE': 15,
    'INST': 16,
    'INST_TYPE': 17,
}
PREPARE_TOKENS = {
    'EMPTY': 0,
    'EXTRACT': 1,
    'THEOREM': 2,
    'HYPOTHESIS': 3,
    'SUBST': 4,
    'SUBST_TYPE': 5,
    'SUBST_PAIR': 6,
    'TERM': 7,
}


INV_ACTION_TOKENS = {v: k for k, v in ACTION_TOKENS.items()}
INV_PREPARE_TOKENS = {v: k for k, v in PREPARE_TOKENS.items()}


class TypeException(Exception):
    pass


class Type(BVT):
    def __init__(
            self,
            value: typing.Any,
            left: BVT,
            right: BVT,
            token: str,
    ):
        super(Type, self).__init__(
            value, left, right
        )
        # `self._token` stores the associated string token so that we can
        # reconstruct type strings directly from their BVT.
        self._token = token

    def token(
            self,
    ) -> str:
        return self._token

    def type_string(
            self,
    ) -> str:
        """ `type_string` formats the Type BVT as a HOL Light type string
        """
        def dump(typ, ifx):
            assert typ.left is not None
            if typ.token() == '__v':
                token = typ.left.token()
                if token[0] == '?':
                    return '(_?_'+token[1:] + ')'
                else:
                    return token
            if typ.token() == '__a':
                if typ.right is None:
                    assert ifx is None
                    return dump(typ.left, None)
                elif ifx is not None:
                    return dump(typ.left, None) + ifx + dump(typ.right, None)
                else:
                    return dump(typ.left, None) + "," + dump(typ.right, None)
            if typ.token() == '__c':
                token = typ.left.token()
                if token == 'fun':
                    if typ.right is None:
                        raise TypeException()
                    return "(" + dump(typ.right, "->") + ")"
                if token == 'prod':
                    if typ.right is None:
                        raise TypeException()
                    return "(" + dump(typ.right, "#") + ")"
                if typ.right is None:
                    return token
                else:
                    return "(" + dump(typ.right, None) + ")" + token

        return ':' + dump(self, None)


class Term(BVT):
    def __init__(
            self,
            value: typing.Any,
            left: BVT,
            right: BVT,
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
            de_bruijn: bool = False,
    ) -> str:
        """ `term_string` formats the Term BVT as a HOL Light term string
        """
        def v_term(term, bounded=[]):
            assert term.token() == '__v'
            typ = term.right.value.type_string()
            term = '(' + term.left.token() + typ + ')'

            if not de_bruijn:
                return term

            if term in bounded:
                for i in reversed(range(len(bounded))):
                    if term == bounded[i]:
                        return '(b' + str(i) + typ + ')'
            return term

        def dump(term, args, bounded):
            if term.token() == '__C':
                right = dump(term.right, [], bounded)
                return dump(term.left, [right] + args, bounded)
            if term.token() == '__A':
                assert term.left.token() == '__v'
                left = dump(term.left, [], [])
                right = dump(term.right, [], bounded+[left])
                left = dump(term.left, [], bounded+[left])
                if len(args) == 0:
                    return '(\\' + left + '. ' + right + ')'
                else:
                    tm = '((\\' + left + '. ' + right + ')'
                    for a in args:
                        tm += ' ' + a
                    tm += ')'
                    return tm
            if term.token() == '__c':
                assert type(term.right.value) is Type

                if len(args) == 0:
                    return '((' + term.left.token() + ')' + \
                        term.right.value.type_string() + ')'
                else:
                    # This is an attempt at simplyfing terms as much as
                    # possible to make them readable.
                    if term.left.token() in [
                            "=", "==>", "/\\", "\\/",
                    ] and len(args) == 2:
                        tm = '(' + args[0] + ' ' + \
                            term.left.token() + \
                            ' ' + args[1] + ')'
                        return tm
                    else:
                        tm = '(((' + term.left.token() + ')' + \
                            term.right.value.type_string() + ')'
                        for a in args:
                            tm += ' ' + a
                        tm += ')'
                        return tm
            if term.token() == '__v':
                assert type(term.right.value) is Type

                if len(args) == 0:
                    return v_term(term, bounded)
                else:
                    tm = '(' + v_term(term, bounded)
                    for a in args:
                        tm += ' ' + a
                    tm += ')'
                    return tm

        return dump(self, [], [])


class Action(BVT):
    def __init__(
            self,
            value: typing.Any,
            left: BVT = None,
            right: BVT = None,
            index: int = None,
    ):
        super(Action, self).__init__(
            value, left, right
        )
        # `self._index` stores the original index of the associated action. It
        # is used to store the index of the theorem assosicated with this
        # action for REPL/Fusion.
        self._index = index

    def hash(
            self,
    ):
        # Compute a hash that is not order dependent for HYPOTHESIS.
        if self.value == ACTION_TOKENS['HYPOTHESIS'] and self._hash is None:
            hashes = [b'HYPOTHESIS']

            def walk(h):
                if h is None:
                    return
                if type(h.value) is Term:
                    hashes.append(h.hash())
                walk(h.left)
                walk(h.right)

            walk(self)

            h = xxhash.xxh64()
            for hh in sorted(hashes):
                h.update(hh)
            self._hash = h.digest()

        return super(Action, self).hash()

    def index(
            self,
    ) -> int:
        return self._index

    def copy(
            self,
    ):
        return Action(
            self.value,
            self.left,
            self.right,
            self._index,
        )

    def __iter__(
            self,
    ):
        yield 'type', INV_ACTION_TOKENS[self.value]
        yield 'hash', base64.b64encode(self.hash()).decode('utf-8')

        if self.left is not None and self.left.value != 0:
            yield 'left', base64.b64encode(self.left.hash()).decode('utf-8')
        else:
            yield 'left', None
        if self.right is not None and self.right.value != 0:
            yield 'right', base64.b64encode(self.right.hash()).decode('utf-8')
        else:
            yield 'right', None

        def hypothesis(a):
            if a is None:
                return []
            else:
                return [a.left.value.term_string()] + hypothesis(a.right)

        def term(a):
            return a.value.term_string()

        def subst(a):
            if a is None:
                return []
            else:
                if a.left is not None:
                    return [
                        [a.left.left.value.term_string(),
                         a.left.right.value.term_string()],
                    ] + subst(a.right)
                else:
                    return [] + subst(a.right)

        def subst_type(a):
            if a is None:
                return []
            else:
                if a.left is not None:
                    return [
                        [a.left.left.value.type_string(),
                         a.left.right.value.type_string()],
                    ] + subst_type(a.right)
                else:
                    return [] + subst_type(a.right)

        if INV_ACTION_TOKENS[self.value] == 'SUBST':
            yield 'subst', subst(self)
        if INV_ACTION_TOKENS[self.value] == 'SUBST_TYPE':
            yield 'subst_type', subst_type(self)
        if INV_ACTION_TOKENS[self.value] == 'TERM':
            yield 'term', term(self.left)
        if INV_ACTION_TOKENS[self.value] == 'THEOREM':
            yield 'hyp', hypothesis(self.left)
            yield 'ccl', term(self.right)

        yield 'index', self._index

    @staticmethod
    def from_action(
            action: str,
            left: BVT,
            right: BVT,
            origin: int = None,
    ):
        value = ACTION_TOKENS[action]
        return Action(value, left, right, origin)

    @staticmethod
    def from_term(
            term: Term,
    ):
        return Action(term)


class ProofTraceTokenizer():
    def __init__(
            self,
    ) -> None:
        self._type_tokens = {
            '__c': 0,
            '__v': 1,
            '__a': 2,
            'bool': 3,
            'fun': 4,
        }
        self._term_tokens = {
            '__C': 0,
            '__A': 1,
            '__c': 2,
            '__v': 3,
            'T': 4,
            '=': 5,
        }

    def split(
            self,
            t,
            seps=['(', ')'],
    ):
        stack = []
        for i, c in enumerate(t):
            if c == seps[0]:
                stack.append(i)
            elif c == seps[1] and stack:
                start = stack.pop()
                if len(stack) == 0:
                    yield t[start + 1: i]

    def type(
            self,
            ty: str,
    ) -> Type:
        """ Construct a Type BVT from a type string.

        Tokenizes constants appearing in types using self._type_tokens.
        """
        def build_args(args):
            if len(args) == 0:
                return None
            else:
                return Type(
                    self._type_tokens['__a'],
                    args[0],
                    build_args(args[1:]),
                    '__a',
                )

        def construct(t):
            if t[0] == 'v':
                chld = list(self.split(t, ['[', ']']))
                assert len(chld) == 1
                assert chld[0] in self._type_tokens
                # if chld[0] not in self._type_tokens:
                #     self._type_tokens[chld[0]] = len(self._type_tokens)
                return Type(
                    self._type_tokens['__v'],
                    Type(self._type_tokens[chld[0]], None, None, chld[0]),
                    None,
                    '__v',
                )
            if t[0] == 'c':
                chld = list(self.split(t, ['[', ']']))
                assert len(chld) == 2
                assert chld[0] in self._type_tokens
                # if chld[0] not in self._type_tokens:
                #     self._type_tokens[chld[0]] = len(self._type_tokens)
                args = [
                    self.type(ty)
                    for ty in list(self.split(chld[1], ['[', ']']))
                ]
                return Type(
                    self._type_tokens['__c'],
                    Type(self._type_tokens[chld[0]], None, None, chld[0]),
                    build_args(args),
                    '__c',
                )

        return construct(ty)

    def term(
            self,
            tm: str,
    ) -> Term:
        """ Construct a Term BVT from a term string.

        Tokenizes constants appearing in terms using self._term_tokens.
        """
        def construct(t):
            if t[0] == 'C':
                chld = list(self.split(t, ['(', ')']))
                assert len(chld) == 2
                return Term(
                    self._term_tokens['__C'],
                    construct(chld[0]),
                    construct(chld[1]),
                    '__C',
                )
            if t[0] == 'A':
                chld = list(self.split(t, ['(', ')']))
                assert len(chld) == 2
                return Term(
                    self._term_tokens['__A'],
                    construct(chld[0]),
                    construct(chld[1]),
                    '__A',
                )
            if t[0] == 'c':
                chld = list(self.split(t, ['(', ')']))
                assert len(chld) == 2
                assert chld[0] in self._term_tokens
                # if chld[0] not in self._term_tokens:
                #     self._term_tokens[chld[0]] = len(self._term_tokens)
                return Term(
                    self._term_tokens['__c'],
                    Term(self._term_tokens[chld[0]], None, None, chld[0]),
                    Term(self.type(chld[1]), None, None, None),
                    '__c',
                )
            if t[0] == 'v':
                chld = list(self.split(t, ['(', ')']))
                assert len(chld) == 2
                assert chld[0] in self._term_tokens
                # if chld[0] not in self._term_tokens:
                #     self._term_tokens[chld[0]] = len(self._term_tokens)
                return Term(
                    self._term_tokens['__v'],
                    Term(self._term_tokens[chld[0]], None, None, chld[0]),
                    Term(self.type(chld[1]), None, None, None),
                    '__v',
                )

        return construct(tm)


class ProofTraceKernel():
    def __init__(
            self,
            dataset_dir: str,
            dataset_size: str,
    ) -> None:
        self._proofs = {}
        self._theorems = {}
        self._names = {}

        # Proof steps that are re-used >1 time.
        self._shared = {}

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

    def name_cut_premise(
            self,
            index,
    ) -> bool:
        if index not in self._names:
            # Log.out("CUT", {
            #     "index": index,
            # })
            self._names[index] = "CUT_" + str(index)
            return True
        return False

    def remove_premise(
            self,
            index,
    ) -> bool:
        if index in self._names:
            del self._names[index]
        if index in self._shared:
            del self._shared[index]


class ProofTraceActions():
    def __init__(
            self,
            name: str,
            actions: typing.List[Action],
            arguments: typing.List[Action],
    ) -> None:
        self._name = name
        self._actions = actions
        self._arguments = arguments
        self._hashes = None

    def dump(
            self,
            path,
    ) -> None:
        with gzip.open(path, 'wb') as f:
            pickle.dump(
                self, f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    def len(
            self,
    ) -> int:
        assert len(self._arguments) == len(self._actions)
        return len(self._actions)

    def prepare_len(
            self,
    ) -> int:
        prepare_len = 0
        for a in self._actions:
            if a.value in INV_PREPARE_TOKENS:
                prepare_len += 1
            else:
                break
        return prepare_len

    def action_len(
            self,
    ) -> int:
        return self.len() - self.prepare_len()

    def name(
            self,
    ) -> str:
        return self._name

    def path(
            self,
    ) -> str:
        return self.name() + \
            '_' + str(self.len()) + '_' + str(self.prepare_len()) + \
            '.actions'

    def actions(
            self,
    ) -> typing.List[Action]:
        return self._actions

    def arguments(
            self,
    ) -> typing.List[Action]:
        return self._arguments

    def hashes(
            self,
    ) -> typing.Dict[bytes, bool]:
        if self._hashes is None:
            self._hashes = {}
            for i in range(self.len()):
                action = self._actions[i]
                argument = self._arguments[i]
                self._hashes[action.hash()] = i
                self._hashes[argument.hash()] = i
        return self._hashes

    def append(
            self,
            action: Action,
            argument: Action,
    ) -> None:
        self._actions.append(action)
        self._arguments.append(argument)

        self.hashes()

        self._hashes[action.hash()] = len(self._actions) - 1
        self._hashes[argument.hash()] = len(self._arguments) - 1

    def build_argument(
            self,
            conclusion: Term,
            hypotheses: typing.List[Term],
            index: int,
    ) -> Action:
        def build_hypothesis(hypotheses):
            if len(hypotheses) == 0:
                return None
            else:
                return Action.from_action(
                    'HYPOTHESIS',
                    Action.from_term(hypotheses[0]),
                    build_hypothesis(hypotheses[1:]),
                )

        return Action.from_action(
            'THEOREM',
            build_hypothesis(hypotheses),
            Action.from_term(conclusion),
            index,
        )

    def seen(
            self,
            a: Action,
    ) -> bool:
        return a.hash() in self.hashes()

    def copy(
            self,
    ):
        ptra = ProofTraceActions(
            self._name,
            self._actions.copy(),
            self._arguments.copy(),
        )
        ptra._hashes = dict(self._hashes)

        return ptra

    def summary(
            self,
    ):
        summary = "["
        for a in self._actions:
            if a.value not in INV_PREPARE_TOKENS:
                left = self._arguments.index(a.left)
                right = self._arguments.index(a.right)
                summary += \
                    "(" + \
                    str(a.value) + "," + str(left) + "," + str(right) + \
                    ")"
        summary += "]"
        return summary


class ProofTrace():
    def __init__(
            self,
            kernel: ProofTraceKernel,
            proof_index: int,
    ):
        self._index = proof_index

        self._target = None
        self._premises = {}
        self._terms = {}
        self._substs = {}
        self._subst_types = {}

        self._steps = {}
        self._theorems = {}
        self._sequence = []

        self._name = str(self._index) + '_' + kernel._names[self._index]

        self.walk(self._index, kernel)

    def name(
            self,
    ):
        return self._name

    def len(
            self,
    ):
        return \
            len(self._sequence) + \
            len(self._premises) + \
            len(self._terms) + \
            len(self._substs) + len(self._subst_types)

    def record_term(
            self,
            term,
    ):
        def term_hash(
                term,
        ):
            h = xxhash.xxh64()
            h.update(term)
            return str(h.digest())

        h = term_hash(term)
        if h in self._terms:
            assert term == self._terms[h]
        else:
            self._terms[h] = term
        return h

    def record_subst(
            self,
            subst,
    ):
        def subst_hash(
                subst,
        ):
            h = xxhash.xxh64()
            for s in subst:
                assert len(s) == 2
                h.update(s[0])
                h.update(s[1])
            return str(h.digest())

        h = subst_hash(subst)
        if h in self._substs:
            assert len(subst) == len(self._substs[h])
            for i, s in enumerate(subst):
                assert s[0] == self._substs[h][i][0]
                assert s[1] == self._substs[h][i][1]
        else:
            self._substs[h] = subst
        return h

    def record_subst_type(
            self,
            subst_type,
    ):
        def subst_type_hash(
                subst_type,
        ):
            h = xxhash.xxh64()
            for s in subst_type:
                assert len(s) == 2
                h.update(s[0])
                h.update(s[1])
            return str(h.digest())

        h = subst_type_hash(subst_type)
        if h in self._subst_types:
            assert len(subst_type) == len(self._subst_types[h])
            for i, s in enumerate(subst_type):
                assert s[0] == self._subst_types[h][i][0]
                assert s[1] == self._subst_types[h][i][1]
        else:
            self._subst_types[h] = subst_type
        return h

    def record_premise(
            self,
            index,
            theorem,
    ):
        if index not in self._premises:
            self._premises[index] = theorem

    def walk(
            self,
            index,
            kernel: ProofTraceKernel,
    ):
        if index in self._steps:
            return

        if index != self._index:
            if (
                    index in kernel._names or
                    index in kernel._shared
            ):
                self.record_premise(index, kernel._theorems[index])
                return
        else:
            self._target = kernel._theorems[index]

        step = kernel._proofs[index].copy()

        if step[0] == 'REFL':
            step[1] = self.record_term(step[1])

        elif step[0] == 'TRANS':
            self.walk(step[1], kernel)
            self.walk(step[2], kernel)

        elif step[0] == 'MK_COMB':
            self.walk(step[1], kernel)
            self.walk(step[2], kernel)

        elif step[0] == 'ABS':
            self.walk(step[1], kernel)
            step[2] = self.record_term(step[2])

        elif step[0] == 'BETA':
            step[1] = self.record_term(step[1])

        elif step[0] == 'ASSUME':
            step[1] = self.record_term(step[1])

        elif step[0] == 'EQ_MP':
            self.walk(step[1], kernel)
            self.walk(step[2], kernel)

        elif step[0] == 'DEDUCT_ANTISYM_RULE':
            self.walk(step[1], kernel)
            self.walk(step[2], kernel)

        elif step[0] == 'INST':
            self.walk(step[1], kernel)
            step[2] = self.record_subst(step[2])

        elif step[0] == 'INST_TYPE':
            self.walk(step[1], kernel)
            step[2] = self.record_subst_type(step[2])

        elif step[0] == 'AXIOM':
            self.record_premise(index, kernel._theorems[index])
            return

        elif step[0] == 'DEFINITION':
            self.record_premise(index, kernel._theorems[index])
            return

        elif step[0] == 'TYPE_DEFINITION':
            self.record_premise(index, kernel._theorems[index])
            return

        else:
            assert False

        self._steps[index] = step
        self._theorems[index] = kernel._theorems[index]

        self._sequence.append(index)

    def actions(
            self,
            t: ProofTraceTokenizer,
    ) -> ProofTraceActions:
        """ Concretize the ProofTraceActions from the ProofTrace

        ProofTraceActions are composed of two sequences, an `actions` sequence
        which are Action passed as input to our models and an `arguments`
        sequence which are Actions representing theorems of previous actions
        and used as "argument" to later actions.
        """
        actions = []
        arguments = []

        cache = {
            'substs': {},
            'subst_types': {},
            'terms': {},
            'indices': {},
        }

        # Empty is used by unary actions as right argument, it lives at the
        # start of the actions sequence after the target (index 1). We can't
        # use None since the language model loss needs an index to use as right
        # arguments even for right arguments of unary actions.
        empty = Action.from_action('EMPTY', None, None)

        # Recursive function used to build theorems hypotheses used for THEOREM
        # actions.
        def build_hypothesis(hypotheses):
            if len(hypotheses) == 0:
                return None
            else:
                return Action.from_action(
                    'HYPOTHESIS',
                    Action.from_term(t.term(hypotheses[0])),
                    build_hypothesis(hypotheses[1:]),
                )

        # Recursive function used to build instantiations substitutions
        def build_subst(subst):
            if len(subst) == 0:
                return Action.from_action('SUBST', None, None)
            else:
                return Action.from_action(
                    'SUBST',
                    Action.from_action(
                        'SUBST_PAIR',
                        Action.from_term(t.term(subst[0][0])),
                        Action.from_term(t.term(subst[0][1])),
                    ),
                    build_subst(subst[1:]),
                )

        # Recursive function used to build type instantiations substitutions
        def build_subst_type(subst_type):
            if len(subst_type) == 0:
                return Action.from_action('SUBST_TYPE', None, None)
            else:
                return Action.from_action(
                    'SUBST_TYPE',
                    Action.from_action(
                        'SUBST_PAIR',
                        Action.from_term(t.type(subst_type[0][0])),
                        Action.from_term(t.type(subst_type[0][1])),
                    ),
                    build_subst_type(subst_type[1:]),
                )

        # Start by recording the target theorem (TARGET action).
        target = Action.from_action(
            'THEOREM',
            build_hypothesis(self._target['hy']),
            Action.from_term(t.term(self._target['cc'])),
        )

        substs = []
        # We first record subst as they are generally deeper than terms but
        # include very similar terms (optimize TreeLSTM cache hit).
        for h in self._substs:
            subst = self._substs[h]
            action = build_subst(subst)
            cache['substs'][h] = action
            substs.append(action)

        subst_types = []
        # We then record subst_type.
        for h in self._subst_types:
            subst_type = self._subst_types[h]
            action = build_subst_type(subst_type)
            cache['subst_types'][h] = action
            subst_types.append(action)

        terms = []
        # We then record terms as they are generally deeper than premises but
        # include very similar terms (optimize TreeLSTM cache hit).
        for h in self._terms:
            term = self._terms[h]
            action = Action.from_action(
                'TERM',
                Action.from_term(t.term(term)),
                None,
            )
            cache['terms'][h] = action
            terms.append(action)

        # Terms are unordered so we order them by depth to optimize cache hit
        # again.
        terms = sorted(
            terms,
            key=lambda action: action.left.value.depth(),
            reverse=True,
        )

        actions = [target, empty] + substs + subst_types + terms
        arguments = [empty, empty] + substs + subst_types + terms

        for idx in self._premises:
            p = self._premises[idx]

            action = Action.from_action(
                'THEOREM',
                build_hypothesis(p['hy']),
                Action.from_term(t.term(p['cc'])),
                idx,
            )
            theorem = action

            cache['indices'][idx] = theorem

            actions.append(action)
            arguments.append(theorem)

        for idx in self._sequence:
            step = self._steps[idx]
            action = None

            if step[0] == 'REFL':
                action = Action.from_action(
                    'REFL',
                    cache['terms'][step[1]],
                    empty,
                    idx,
                )
            elif step[0] == 'TRANS':
                action = Action.from_action(
                    'TRANS',
                    cache['indices'][step[1]],
                    cache['indices'][step[2]],
                    idx,
                )
            elif step[0] == 'MK_COMB':
                action = Action.from_action(
                    'MK_COMB',
                    cache['indices'][step[1]],
                    cache['indices'][step[2]],
                    idx,
                )
            elif step[0] == 'ABS':
                action = Action.from_action(
                    'ABS',
                    cache['indices'][step[1]],
                    cache['terms'][step[2]],
                    idx,
                )
            elif step[0] == 'BETA':
                action = Action.from_action(
                    'BETA',
                    cache['terms'][step[1]],
                    empty,
                    idx,
                )
            elif step[0] == 'ASSUME':
                action = Action.from_action(
                    'ASSUME',
                    cache['terms'][step[1]],
                    empty,
                    idx,
                )
            elif step[0] == 'EQ_MP':
                action = Action.from_action(
                    'EQ_MP',
                    cache['indices'][step[1]],
                    cache['indices'][step[2]],
                    idx,
                )
            elif step[0] == 'DEDUCT_ANTISYM_RULE':
                action = Action.from_action(
                    'DEDUCT_ANTISYM_RULE',
                    cache['indices'][step[1]],
                    cache['indices'][step[2]],
                    idx,
                )
            elif step[0] == 'INST':
                action = Action.from_action(
                    'INST',
                    cache['indices'][step[1]],
                    cache['substs'][step[2]],
                    idx,
                )

            elif step[0] == 'INST_TYPE':
                action = Action.from_action(
                    'INST_TYPE',
                    cache['indices'][step[1]],
                    cache['subst_types'][step[2]],
                    idx,
                )

            elif step[0] == 'AXIOM':
                assert False

            elif step[0] == 'DEFINITION':
                assert False

            elif step[0] == 'TYPE_DEFINITION':
                assert False

            theorem = Action.from_action(
                'THEOREM',
                build_hypothesis(self._theorems[idx]['hy']),
                Action.from_term(t.term(
                    self._theorems[idx]['cc']
                )),
                idx,
            )
            cache['indices'][idx] = theorem

            actions.append(action)
            arguments.append(theorem)

        return ProofTraceActions(
            self.name(),
            actions,
            arguments,
        )

    def min_cut(
            self,
            min_size: int,
            max_size: int,
    ) -> typing.List[int]:
        candidates = []
        seen, queue = set(), [self._sequence[-1]]

        def add_step(idx):
            if idx not in self._premises and idx not in seen:
                queue.append(idx)

        while len(queue) > 0:
            idx = queue.pop(0)

            if idx in seen:
                continue
            seen.add(idx)
            step = self._steps[idx]

            if step[0] == 'REFL':
                pass
            elif step[0] == 'TRANS':
                add_step(step[1])
                add_step(step[2])
            elif step[0] == 'MK_COMB':
                add_step(step[1])
                add_step(step[2])
            elif step[0] == 'ABS':
                add_step(step[1])
            elif step[0] == 'BETA':
                pass
            elif step[0] == 'ASSUME':
                pass
            elif step[0] == 'EQ_MP':
                add_step(step[1])
                add_step(step[2])
            elif step[0] == 'DEDUCT_ANTISYM_RULE':
                add_step(step[1])
                add_step(step[2])
            elif step[0] == 'INST':
                add_step(step[1])
            elif step[0] == 'INST_TYPE':
                add_step(step[1])

            elif step[0] == 'AXIOM':
                assert False
            elif step[0] == 'DEFINITION':
                assert False
            elif step[0] == 'TYPE_DEFINITION':
                assert False

            if len(seen) >= min_size and len(seen) <= max_size:
                candidates.append((set(seen), set(queue)))

        candidates = sorted(candidates, key=lambda c: len(c[1]))
        candidates = [
            c for c in candidates if len(c[1]) <= len(candidates[0][1]) + 2
        ]
        candidates = sorted(candidates, key=lambda c: len(c[0]))

        assert len(candidates) > 0

        return sorted(list(candidates[0][1]))

    def localize(
            self,
    ) -> None:
        cache = {
            '_term_index': 0,
            '_type_index': 0,
        }

        term_pattern = re.compile(r"v\(_[0-9]+\)")
        type_pattern = re.compile(r"v\[\?[0-9]+\]")

        def localize_blob(blob):
            replacements = {}

            for m in re.findall(term_pattern, blob):
                if m not in cache:
                    cache[m] = "v(_" + str(cache['_term_index']) + ")"
                    cache['_term_index'] += 1
                if m not in replacements:
                    replacements[m] = cache[m]
            for m in re.findall(type_pattern, blob):
                if m not in cache:
                    cache[m] = "v[?" + str(cache['_type_index']) + "]"
                    cache['_type_index'] += 1
                if m not in replacements:
                    replacements[m] = cache[m]

            for old in replacements:
                blob = blob.replace(old, replacements[old])

            return blob

        def localize_term(term):
            return localize_blob(term)

        def localize_subst(subst):
            new = copy.deepcopy(subst)
            for i in range(len(subst)):
                assert len(subst[i]) == 2
                new[i][0] = localize_blob(subst[i][0])
                new[i][1] = localize_blob(subst[i][1])
            return new

        def localize_subst_type(subst_type):
            new = copy.deepcopy(subst_type)
            for i in range(len(subst_type)):
                assert len(subst_type[i]) == 2
                new[i][0] = localize_blob(subst_type[i][0])
                new[i][1] = localize_blob(subst_type[i][1])
            return new

        def localize_theorem(th):
            new = copy.deepcopy(th)
            new['cc'] = localize_blob(th['cc'])
            for i in range(len(th['hy'])):
                new['hy'][i] = localize_blob(th['hy'][i])
            return new

        self._target = localize_theorem(self._target)
        for idx in self._premises:
            self._premises[idx] = localize_theorem(self._premises[idx])
        for idx in self._theorems:
            self._theorems[idx] = localize_theorem(self._theorems[idx])
        for h in self._terms:
            self._terms[h] = localize_term(self._terms[h])
        for h in self._substs:
            self._substs[h] = localize_subst(self._substs[h])
        for h in self._subst_types:
            self._subst_types[h] = localize_subst_type(self._subst_types[h])

    def tokenize(
            self,
            tokenizer: ProofTraceTokenizer,
    ):
        token_pattern = re.compile(r"[vc]\([^\(\)]+\)|[vc]\[[^\[\]]+\]")

        def tokenize_blob(blob):
            for m in re.findall(token_pattern, blob):
                token = m[2:-1]
                if m[1] == '[':
                    if token not in tokenizer._type_tokens:
                        tokenizer._type_tokens[token] = \
                            len(tokenizer._type_tokens)
                if m[1] == '(':
                    if token not in tokenizer._term_tokens:
                        tokenizer._term_tokens[token] = \
                            len(tokenizer._term_tokens)

        def tokenize_theorem(th):
            tokenize_blob(th['cc'])
            for i in range(len(th['hy'])):
                tokenize_blob(th['hy'][i])

        for idx in self._theorems:
            tokenize_theorem(self._theorems[idx])
        for idx in self._premises:
            tokenize_theorem(self._premises[idx])


class ProofTraceLMDataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            dataset_size: str,
            test: bool,
            sequence_length: int,
            trace_max_length=-1,
    ) -> None:
        self._sequence_length = sequence_length

        self._cases = []
        self._ptra_files = []

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
            match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
            if match is None:
                continue
            ptra_len = int(match.group(1))
            prepare_len = int(match.group(2))

            if trace_max_length <= -1 or ptra_len <= trace_max_length:
                self._ptra_files.append(p)
                for pos in range(prepare_len, ptra_len):
                    if pos < self._sequence_length:
                        self._cases.append((processed, pos))
                processed += 1

        Log.out(
            "Loaded extracted ProofTraces LM Dataset", {
                'cases': len(self._cases),
                'processed': processed,
            })

    def __len__(
            self,
    ) -> int:
        return len(self._cases)

    def __getitem__(
            self,
            idx: int,
    ):
        with gzip.open(self._ptra_files[self._cases[idx][0]], 'rb') as f:
            ptra = pickle.load(f)

        truth = ptra.actions()[self._cases[idx][1]]
        actions = ptra.actions()[:self._cases[idx][1]]
        arguments = ptra.arguments()[:self._cases[idx][1]]

        # value = 0.0
        # for i in range(ptra.len() - len(trace)):
        #     value = 1.0 + 0.99 * value
        # value = ptra.action_len() * 0.99 ** (ptra.len() - len(trace))
        value = float(ptra.len() - len(actions))

        actions.append(Action.from_action('EXTRACT', None, None))

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._sequence_length:
            actions.append(empty)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)

        return (self._cases[idx][1], actions, arguments, truth, value)


def lm_collate(
        batch
) -> typing.Tuple[
    typing.List[int],
    typing.List[typing.List[Action]],
    typing.List[Action],
    typing.List[float],
]:
    indices = []
    actions = []
    arguments = []
    truths = []
    values = []

    for (idx, act, arg, trh, val) in batch:
        indices.append(idx)
        actions.append(act)
        arguments.append(arg)
        truths.append(trh)
        values.append(val)

    return (indices, actions, arguments, truths, values)


def dump_trace(args):
    config, tokenizer, tr, idx, total = args
    ptra = tr.actions(tokenizer)

    test = False
    for nm in TEST_FILTER:
        if re.search(nm, tr.name()) is not None:
            test = True

    if test:
        path = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            "test_traces",
        )
    else:
        path = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            "train_traces",
        )

    ptra_path = os.path.join(path, ptra.path())
    Log.out("Writing ProofTraceActions", {
        'path': ptra_path,
        'index': idx,
        'total': total,
    })
    ptra.dump(ptra_path)

    length = ptra.len()
    del ptra

    return length


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
    tokenizer = ProofTraceTokenizer()

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

    Log.out("Starting min_cut operations")

    kernel._shared = {}
    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]
    traces = [tr for tr in traces if len(tr._steps) > 0]

    excess = [
        tr for tr in traces
        if tr.len() > config.get('prooftrace_max_demo_length') * 4/5
    ]
    Log.out("Min-cut initialization", {
        'excess': len(excess),
    })

    while len(excess) > 0:
        orig = []
        cut = []

        for tr in excess:
            orig.append(tr._index)
            cut += tr.min_cut(
                config.get('prooftrace_max_demo_length') * 1/8,
                config.get('prooftrace_max_demo_length') * 1/2,
            )

        for idx in cut:
            kernel.name_cut_premise(idx)

        refresh = orig + cut
        traces = [ProofTrace(kernel, k) for k in refresh]
        excess = [
            tr for tr in traces
            if tr.len() > config.get('prooftrace_max_demo_length') * 4/5
        ]

        Log.out("Min-cut processing loop", {
            'excess': len(excess),
            'orig': len(orig),
            'cut': len(cut),
        })

    Log.out("Stitching small prooftraces")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]
    traces = [tr for tr in traces if len(tr._steps) > 0]

    for tr in traces:
        if tr.len() < 32:
            # Log.out("Remove small prooftrace", {
            #     'name': tr.name(),
            #     'index': tr._index,
            # })
            kernel.remove_premise(tr._index)

    Log.out("Starting final prooftraces generation")

    traces = [ProofTrace(kernel, k) for k in kernel._names.keys()]
    traces = [tr for tr in traces if len(tr._steps) > 0]
    traces = sorted(traces, key=lambda tr: tr._index)

    # Finally we localize the resulting traces.
    for tr in traces:
        tr.localize()

    Log.out("Prooftraces computed, filtered, localized and sorted", {
        "traces_count": len(traces),
    })

    for tr in traces:
        tr.tokenize(tokenizer)

    Log.out("Pre-tokenized prooftraces", {
        "term_token_count": len(tokenizer._term_tokens),
        "type_token_count": len(tokenizer._type_tokens),
    })

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'wb') as f:
        pickle.dump(
            tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL
        )

    Log.out("Dumped tokenizer", {
        "term_token_count": len(tokenizer._term_tokens),
        "type_token_count": len(tokenizer._type_tokens),
    })

    Log.histogram(
        "ProofTraces Premises",
        [len(tr._premises) for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Substs",
        [len(tr._substs) for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces SubstTypes",
        [len(tr._subst_types) for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Terms",
        [len(tr._terms) for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Steps",
        [len(tr._steps) for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.histogram(
        "ProofTraces Length",
        [tr.len() for tr in traces],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )
    Log.out("Starting action generation")

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

    executor = concurrent.futures.ProcessPoolExecutor()

    map_args = []
    for i, tr in enumerate(traces):
        map_args.append([config, tokenizer, tr, i, len(traces)])

    trace_lengths = [
        l for l in executor.map(dump_trace, map_args, chunksize=32)
    ]

    Log.histogram(
        "ProofTraces Length",
        trace_lengths,
        buckets=[64, 128, 256, 512, 1024, 2048, 4096],
        labels=["0064", "0128", "0256", "0512", "1024", "2048", "4096"]
    )

    Log.out("Dumped all traces", {
        "traces_path_train": traces_path_train,
        "traces_path_test": traces_path_test,
        "trace_count": len(traces),
    })

    # small: term_token_count=427 type_token_count=70
    # small[1024]: term_token_count=338 type_token_count=70
    # small[1024 min_cut]: term_token_count=427 type_token_count=70
    # small[1024 min_cut local]: term_token_count=114 type_token_count=28

    # medium term_token_count=14227 type_token_count=983
    # medium[1024]: term_token_count=2247 type_token_count=564
    # medium[1024 min_cut]: term_token_count=18756 type_token_count=1017
    # medium[1024 min_cut local]: term_token_count=1124 type_token_count=69


# def extract():
#     import cProfile
#     cProfile.runctx(
#         'extract_profile()', globals(), locals(), 'extract.profile'
#     )
