import argparse
import typing
import os
import pickle
import re

from dataset.prooftrace import \
    INV_ACTION_TOKENS, \
    Type, Term

from utils.config import Config
from utils.log import Log


class Thm():
    def __init__(
            self,
            index: int,
            hypotheses: typing.List[Term],
            conclusion: Term,
    ):
        self._index = index
        self._hypotheses = hypotheses
        self._conclusion = conclusion

    def index(
            self,
    ) -> int:
        return self._index

    def concl(
            self,
    ) -> Term:
        return self._conclusion

    def hyp(
            self,
    ) -> typing.List[Term]:
        return self._hypotheses


class KernelException(Exception):
    pass


def assume(
        condition: bool,
):
    if not condition:
        raise KernelException()


class Kernel():
    def __init__(
            self,
    ):
        self._theorems = {}
        self._next_thm_index = 9999999

    def token_term(
            self,
            token,
    ) -> Type:
        if token == '=':
            return Term(5, None, None, '=')
        assert False

    def token_type(
            self,
            token,
    ) -> Type:
        if token == 'fun':
            return Type(4, None, None, 'fun')
        if token == 'bool':
            return Type(3, None, None, 'fun')
        assert False

    def fun_type(
            self,
            left: Type,
            right: Type,
    ) -> Type:
        return Type(
            0,
            self.token_type('fun'),
            Type(2, left, Type(2, right, None, '__a'), '__a'),
            '__c',
        )

    def bool_type(
            self,
    ) -> Type:
        return Type(
            0,
            self.token_type('bool'),
            None,
            '__c',
        )

    def type_of(
            self,
            term: Term,
    ) -> Type:
        if term.token() == '__v':
            assert type(term.right.value) is Type
            return term.right.value

        if term.token() == '__c':
            assert type(term.right.value) is Type
            return term.right.value

        if term.token() == '__C':
            ty = self.type_of(term.left)
            assert ty.token() == '__c'
            assert ty.left.token() == 'fun'
            assert ty.right.token() == '__a'
            assert ty.right.right.token() == '__a'
            assert ty.right.right.left is not None
            return ty.right.right.left

        if term.token() == '__A':
            return self.fun_type(
                self.type_of(term.left),
                self.type_of(term.right),
            )

    def safe_mk_eq(
            self,
            left: Term,
            right: Term,
    ) -> Term:
        ty = self.type_of(left)

        return Term(
            0,
            Term(
                0, Term(
                    2, self.token_term('='),
                    Term(
                        self.fun_type(
                            ty,
                            self.fun_type(ty, self.bool_type()),
                        ), None, None, None,
                    ), '__c'
                ), left, '__C',
            ), right, '__C',
        )

    def term_union(
            self,
            s1: typing.List[Term],
            s2: typing.List[Term],
    ) -> typing.List[Term]:
        u = [t for t in s1]
        hh = [t.term_string(True) for t in s1]

        for t in s2:
            if t.term_string(True) not in hh:
                u.append(t)

        return u

    def PREMISE(
            self,
            thm: Thm,
    ) -> Thm:
        assert thm.index() not in self._theorems
        self._theorems[thm.index()] = thm

        return thm

    def REFL(
            self,
            term: Term,
    ) -> Thm:
        self._next_thm_index += 1
        index = self._next_thm_index

        thm = Thm(
            index,
            [],
            self.safe_mk_eq(term, term),
        )

        self._theorems[index] = thm

        return thm

    def TRANS(
            self,
            idx1: int,
            idx2: int,
    ) -> Thm:
        assume(idx1 in self._theorems)
        assume(idx2 in self._theorems)

        thm1 = self._theorems[idx1]
        thm2 = self._theorems[idx2]

        c1 = thm1.concl()
        c2 = thm2.concl()

        assume(c1.token() == '__C')
        assume(c1.left.token() == '__C')
        assume(c1.left.left.token() == '__c')
        assume(c1.left.left.left.token() == '=')

        eql = c1.left
        m1 = c1.right

        assume(c2.token() == '__C')
        assume(c2.left.token() == '__C')
        assume(c2.left.left.token() == '__c')
        assume(c2.left.left.left.token() == '=')

        r = c2.right
        m2 = c2.left.right

        assume(m1.term_string(True) == m2.term_string(True))

        self._next_thm_index += 1
        index = self._next_thm_index

        thm = Thm(
            index,
            self.term_union(thm1.hyp(), thm2.hyp()),
            Term(0, eql, r, '__C'),
        )

        self._theorems[index] = thm

        return thm

    def MK_COMB(
            self,
            idx1: int,
            idx2: int,
    ) -> Thm:
        assume(idx1 in self._theorems)
        assume(idx2 in self._theorems)

        thm1 = self._theorems[idx1]
        thm2 = self._theorems[idx2]

        c1 = thm1.concl()
        c2 = thm2.concl()

        assume(c1.token() == '__C')
        assume(c1.left.token() == '__C')
        assume(c1.left.left.token() == '__c')
        assume(c1.left.left.left.token() == '=')

        l1 = c1.left.right
        r1 = c1.right

        assume(c2.token() == '__C')
        assume(c2.left.token() == '__C')
        assume(c2.left.left.token() == '__c')
        assume(c2.left.left.left.token() == '=')

        l2 = c2.left.right
        r2 = c2.right

        tyr1 = self.type_of(r1)

        assume(tyr1.token() == '__c')
        assume(tyr1.left.token() == 'fun')

        ty = tyr1.right.left
        tyr2 = self.type_of(r2)

        assume(ty.type_string() == tyr2.type_string())

        self._next_thm_index += 1
        index = self._next_thm_index

        thm = Thm(
            index,
            self.term_union(thm1.hyp(), thm2.hyp()),
            self.safe_mk_eq(
                Term(0, l1, l2, '__C'),
                Term(0, r1, r2, '__C'),
            ),
        )

        self._theorems[index] = thm

        return thm

    def ABS(
            self,
            idx1: int,
            term: Term,
    ) -> Thm:
        pass

    def BETA(
            self,
            term: Term,
    ) -> Thm:
        pass

    def ASSUME(
            self,
            term: Term,
    ) -> Thm:
        pass

    def EQ_MP(
            self,
            idx1: int,
            idx2: int,
    ) -> Thm:
        pass

    def DEDUCT_ANTISYM_RULE(
            self,
            idx1: int,
            idx2: int,
    ) -> Thm:
        pass

    def subst(
            self,
            tm,
            subst,
    ):
        for s in subst:
            assume(s[0].token() == '__v')

        def vsubst(tm, subst):
            if len(subst) == 0:
                return tm

            if tm.token() == '__v':
                for s in subst:
                    if s[0].term_string() == tm.term_string():
                        return s[1]
                return tm
            if tm.token() == '__c':
                return tm
            if tm.token() == '__C':
                ltm = vsubst(tm.left, subst)
                rtm = vsubst(tm.right, subst)
                if ltm.hash() == tm.left.hash() and \
                        rtm.hash() == tm.right.hash():
                    return tm
                else:
                    return Term(0, ltm, rtm, '__C')
            if tm.token() == '__A':
                v = tm.left
                s = vsubst(
                    tm.right,
                    filter(lambda s: s[0].hash() != v.hash(), subst),
                )
                if s.hash() == tm.right.hash():
                    return tm
                # TOOD(stan) variant if bounded
                return Term(1, tm.left, s, '__A')

        return vsubst(tm, subst)

    def INST(
            self,
            idx1: int,
            subst: typing.List[typing.List[Term]],
    ) -> Thm:
        assume(idx1 in self._theorems)
        thm1 = self._theorems[idx1]

    def INST_TYPE(
            self,
            idx1: int,
            subst_type: typing.List[typing.List[Type]],
    ) -> Thm:
        pass


def test():
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

    print("==============================")
    print("ProofTrace Kernel testing \\o/")
    print("------------------------------")

    kernel = Kernel()

    dataset_dir = "./data/prooftrace/{}/test_traces".format(
        config.get("prooftrace_dataset_size"),
    )
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    for p in files:
        if re.search("\\.actions$", p) is None:
            continue
        with open(p, 'rb') as f:
            ptra = pickle.load(f)

        Log.out("Replaying ProofTraceActions", {
            "path": p,
            "actions_count": len(ptra.actions()),
        })

        for a in ptra.actions():
            action_token = INV_ACTION_TOKENS[a.value]
            if action_token == 'PREMISE':
                def hypotheses(action):
                    if action is None:
                        return []
                    else:
                        action_token = INV_ACTION_TOKENS[action.value]
                        assert action_token == 'HYPOTHESIS'
                        return hypotheses(action.right) + [action.left.value]
                thm = kernel.PREMISE(
                    Thm(
                        a.index(),
                        hypotheses(a.left.right),
                        a.left.left.value,
                    ),
                )

                a._index = thm.index()
                Log.out("PREMISE", {
                    'index': thm.index(),
                })

            if action_token == 'REFL':
                thm = kernel.REFL(a.left.left.value)

                a._index = thm.index()
                Log.out("REFL", {
                    'index': thm.index(),
                })

            if action_token == 'TRANS':
                thm = kernel.TRANS(
                    a.left.index(),
                    a.right.index(),
                )

                a._index = thm.index()
                Log.out("TRANS", {
                    'index': thm.index(),
                })

            if action_token == 'MK_COMB':
                thm = kernel.MK_COMB(
                    a.left.index(),
                    a.right.index(),
                )

                a._index = thm.index()
                Log.out("MK_COMB", {
                    'index': thm.index(),
                })

            print(action_token)
