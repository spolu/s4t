import argparse
import typing
import os
import pickle

from dataset.prooftrace import \
    ProofTraceKernel, ProofTraceTokenizer, \
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

    def thm_string(
            self,
            de_bruijn: bool = False,
    ) -> str:
        return "{} |- {}".format(
            ", ".join(sorted([
                h.term_string(de_bruijn) for h in self._hypotheses
            ])),
            self._conclusion.term_string(de_bruijn),
        )


class FusionException(Exception):
    pass


def assume(
        condition: bool,
):
    if not condition:
        raise FusionException()


class Fusion():
    def __init__(
            self,
            tokenizer: ProofTraceTokenizer,
    ):
        self._theorems = {}
        self._next_thm_index = 99999999

        self._t = tokenizer

    def copy(
            self,
    ):
        f = Fusion(self._t)
        f._theorems = dict(self._theorems)
        f._next_thm_index = self._next_thm_index

        return f

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
        hh = [t.term_string() for t in s1]

        for t in s2:
            if t.term_string() not in hh:
                u.append(t)

        return u

    def term_unique(
            self,
            s: typing.List[Term],
    ) -> typing.List[Term]:
        out = []
        dmp = []
        for t in s:
            tt = t.term_string()
            if tt not in dmp:
                out.append(t)
                dmp.append(tt)
        return out

    def _theorem(
            self,
            hyp: typing.List[Term],
            concl: Term,
            fake,
    ) -> Thm:
        if fake:
            return Thm(-1, hyp, concl)

        self._next_thm_index += 1
        index = self._next_thm_index

        thm = Thm(index, hyp, concl)
        self._theorems[index] = thm

        return thm

    def PREMISE(
            self,
            thm: Thm,
            fake: bool = False,
    ) -> Thm:
        if fake:
            return thm

        assert thm.index() not in self._theorems
        self._theorems[thm.index()] = thm

        return thm

    def REFL(
            self,
            term: Term,
            fake: bool = False,
    ) -> Thm:
        return self._theorem(
            [],
            self.safe_mk_eq(term, term),
            fake,
        )

    def TRANS(
            self,
            idx1: int,
            idx2: int,
            fake: bool = False,
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

        return self._theorem(
            self.term_union(thm1.hyp(), thm2.hyp()),
            Term(0, eql, r, '__C'),
            fake,
        )

    def MK_COMB(
            self,
            idx1: int,
            idx2: int,
            fake: bool = False,
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

        return self._theorem(
            self.term_union(thm1.hyp(), thm2.hyp()),
            self.safe_mk_eq(
                Term(0, l1, l2, '__C'),
                Term(0, r1, r2, '__C'),
            ),
            fake,
        )

    def ABS(
            self,
            idx1: int,
            v: Term,
            fake: bool = False,
    ) -> Thm:
        assume(v.token() == '__v')

        assume(idx1 in self._theorems)
        thm1 = self._theorems[idx1]
        c1 = thm1.concl()

        assume(c1.token() == '__C')
        assume(c1.left.token() == '__C')
        assume(c1.left.left.token() == '__c')
        assume(c1.left.left.left.token() == '=')

        l1 = c1.left.right
        r1 = c1.right

        assume(len(list(filter(
            lambda t: self.free_in(v, t),
            thm1.hyp(),
        ))) == 0)

        return self._theorem(
            thm1.hyp(),
            self.safe_mk_eq(
                Term(1, v, l1, '__A'),
                Term(1, v, r1, '__A'),
            ),
            fake,
        )

    def BETA(
            self,
            term: Term,
            fake: bool = False,
    ) -> Thm:
        assume(term.token() == '__C')
        assume(term.left.token() == '__A')

        v = term.left.left
        bod = term.left.right
        arg = term.right

        assume(v.term_string() == arg.term_string())

        return self._theorem(
            [],
            self.safe_mk_eq(term, bod),
            fake,
        )

    def ASSUME(
            self,
            term: Term,
            fake: bool = False,
    ) -> Thm:
        assume(self.type_of(term).type_string() == ':bool')

        return self._theorem(
            [term],
            term,
            fake,
        )

    def EQ_MP(
            self,
            idx1: int,
            idx2: int,
            fake: bool = False,
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

        assume(l1.term_string(True) == c2.term_string(True))

        return self._theorem(
            self.term_union(thm1.hyp(), thm2.hyp()),
            r1,
            fake,
        )

    def DEDUCT_ANTISYM_RULE(
            self,
            idx1: int,
            idx2: int,
            fake: bool = False,
    ) -> Thm:
        assume(idx1 in self._theorems)
        assume(idx2 in self._theorems)

        thm1 = self._theorems[idx1]
        thm2 = self._theorems[idx2]

        c1 = thm1.concl()
        c2 = thm2.concl()

        return self._theorem(
            self.term_union(
                list(filter(
                    lambda h1: h1.term_string(True) != c2.term_string(True),
                    thm1.hyp(),
                )),
                list(filter(
                    lambda h2: h2.term_string(True) != c1.term_string(True),
                    thm2.hyp(),
                )),
            ),
            self.safe_mk_eq(c1, c2),
            fake,
        )

    def free_in(
            self,
            v: Term,
            term: Term,
    ) -> bool:
        assume(v.token() == '__v')

        def vfree_in(v, tm):
            if tm.token() == '__A':
                return v.term_string() != tm.left.term_string() and \
                    vfree_in(v, tm.right)
            if tm.token() == '__C':
                return vfree_in(v, tm.left) or vfree_in(v, tm.right)
            return v.term_string() == tm.term_string()

        return vfree_in(v, term)

    def frees(
            self,
            term: Term,
    ) -> typing.List[Term]:
        def vfrees(tm):
            if tm.token() == '__v':
                return [tm]
            if tm.token() == '__c':
                return []
            if tm.token() == '__A':
                bfs = vfrees(tm.right)
                vfs = []
                for v in bfs:
                    if v.term_string() != tm.left.term_string():
                        vfs.append(v)
                return vfs
            if tm.token() == '__C':
                return self.term_union(
                    vfrees(tm.left),
                    vfrees(tm.right),
                )

        return vfrees(term)

    def variant(
            self,
            avoid: typing.List[Term],
            v: Term,
    ):
        assume(v.token() == '__v')

        if len(list(filter(
            lambda t: self.free_in(v, t),
            avoid,
        ))) == 0:
            return v

        token = v.left.token() + '\''

        if token not in self._t._term_tokens:
            self._t._term_tokens[token] = len(self._t._term_tokens)

        return self.variant(
            avoid,
            Term(3, Term(
                self._t._term_tokens[token], None, None, token
            ), v.right, '__v'),
        )

    def subst(
            self,
            tm: Term,
            subst: typing.List[typing.List[Term]],
    ):
        for s in subst:
            assume(s[0].token() == '__v')

        def vsubst(tm, subst):
            if len(subst) == 0:
                return tm
            if tm.token() == '__v':
                for s in subst:
                    if s[0].term_string() == tm.term_string():
                        assume(self.type_of(tm).type_string() ==
                               self.type_of(s[1]).type_string())
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
                fsubst = list(filter(
                    lambda s: s[0].term_string() != v.term_string(),
                    subst,
                ))
                b = vsubst(tm.right, fsubst)
                if b.hash() == tm.right.hash():
                    return tm
                if len(list(filter(
                        lambda s: (self.free_in(v, s[1]) and
                                   self.free_in(s[0], tm.right)),
                        fsubst,
                ))) > 0:
                    vv = self.variant([b], v)
                    return Term(1, vv, vsubst(
                        tm.right, fsubst + [[v, vv]]
                    ), '__A')
                return Term(1, v, b, '__A')

        return vsubst(tm, subst)

    def INST(
            self,
            idx1: int,
            subst: typing.List[typing.List[Term]],
            fake: bool = False,
    ) -> Thm:
        assume(idx1 in self._theorems)
        thm1 = self._theorems[idx1]

        return self._theorem(
            self.term_unique(
                [self.subst(h, subst) for h in thm1.hyp()],
            ),
            self.subst(thm1.concl(), subst),
            fake,
        )

    def subst_type(
            self,
            tm: Term,
            subst_type: typing.List[typing.List[Type]],
    ):
        for s in subst_type:
            assume(s[0].token() == '__v')

        def tsubst(ty, subst_type):
            if len(subst_type) == 0:
                return ty
            if ty is None:
                return None
            if ty.token() == '__v':
                for s in subst_type:
                    if s[0].type_string() == ty.type_string():
                        return s[1]
                return ty
            if ty.token() == '__a':
                lty = tsubst(ty.left, subst_type)
                rty = tsubst(ty.right, subst_type)
                if lty.hash() == ty.left.hash() and \
                        (ty.right is None or rty.hash() == ty.right.hash()):
                    return ty
                return Type(2, lty, rty, '__a')
            if ty.token() == '__c':
                rty = tsubst(ty.right, subst_type)
                if ty.right is None or rty.hash() == ty.right.hash():
                    return ty
                return Type(0, ty.left, rty, '__c')

        class Clash(Exception):
            def __init__(
                    self,
                    term: Term,
            ) -> None:
                self.term = term

        def inst(tm, subst_type, env):
            if tm.token() == '__v':
                ty = tsubst(tm.right.value, subst_type)
                stm = Term(3, tm.left, Term(ty, None, None, None), '__v')
                if ty.hash() == tm.right.value.hash():
                    stm = tm
                ttm = tm
                for s in env:
                    if s[0].term_string() == stm.term_string():
                        ttm = s[1]
                if ttm.term_string() != tm.term_string():
                    raise Clash(stm)
                return stm
            if tm.token() == '__c':
                ty = tsubst(tm.right.value, subst_type)
                if ty.hash() == tm.right.value.hash():
                    return tm
                return Term(2, tm.left, Term(ty, None, None, None), '__c')
            if tm.token() == '__C':
                ltm = inst(tm.left, subst_type, env)
                rtm = inst(tm.right, subst_type, env)
                if ltm.hash() == tm.left.hash() and \
                        rtm.hash() == tm.right.hash():
                    return tm
                else:
                    return Term(0, ltm, rtm, '__C')
            if tm.token() == '__A':
                v = inst(tm.left, subst_type, [])
                try:
                    b = inst(tm.right, subst_type, env + [[v, tm.left]])
                    if v.hash() == tm.left.hash() and \
                            b.hash() == tm.right.hash():
                        return tm
                    else:
                        return Term(1, v, b, '__A')
                except Clash as e:
                    assume(e.term.term_string() == v.term_string())
                    frees = [inst(v, subst_type, [])
                             for v in self.frees(tm.right)]
                    vv = self.variant(frees, v)
                    assume(vv.token() == '__v')
                    assume(v.token() == '__v')
                    z = Term(3, vv.left, tm.left.right, '__v')
                    return inst(
                        Term(1, z, self.subst(
                            tm.right, [[tm.left, z]]
                        ), '__A'),
                        subst_type,
                        env,
                    )

        if len(subst_type) == 0:
            return tm

        return inst(tm, subst_type, [])

    def INST_TYPE(
            self,
            idx1: int,
            subst_type: typing.List[typing.List[Type]],
            fake: bool = False,
    ) -> Thm:
        assume(idx1 in self._theorems)
        thm1 = self._theorems[idx1]

        return self._theorem(
            self.term_unique(
                [self.subst_type(h, subst_type) for h in thm1.hyp()],
            ),
            self.subst_type(thm1.concl(), subst_type),
            fake,
        )


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

    with open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    k = ProofTraceKernel(
        os.path.expanduser(config.get('prooftrace_dataset_dir')),
        config.get('prooftrace_dataset_size'),
        tokenizer,
    )

    print("==============================")
    print("ProofTrace Fusion testing \\o/")
    print("------------------------------")

    fusion = Fusion(tokenizer)

    for i in range(len(k._proofs)):
        step = k._proofs[i]

        thm = None

        if step[0] == 'DEFINITION' or \
                step[0] == 'TYPE_DEFINITION' or \
                step[0] == 'AXIOM':
            thm = Thm(
                i,
                [k.term(hy) for hy in k._theorems[i]['hy']],
                k.term(k._theorems[i]['cc']),
            )

        if step[0] == 'REFL':
            thm = fusion.REFL(k.term(step[1]))

        if step[0] == 'TRANS':
            thm = fusion.TRANS(
                step[1],
                step[2],
            )

        if step[0] == 'MK_COMB':
            thm = fusion.MK_COMB(
                step[1],
                step[2],
            )

        if step[0] == 'ABS':
            thm = fusion.ABS(step[1], k.term(step[2]))

        if step[0] == 'BETA':
            thm = fusion.BETA(k.term(step[1]))

        if step[0] == 'ASSUME':
            thm = fusion.ASSUME(k.term(step[1]))

        if step[0] == 'EQ_MP':
            thm = fusion.EQ_MP(
                step[1],
                step[2],
            )

        if step[0] == 'DEDUCT_ANTISYM_RULE':
            thm = fusion.DEDUCT_ANTISYM_RULE(
                step[1],
                step[2],
            )

        if step[0] == 'INST':
            thm = fusion.INST(
                step[1],
                [[k.term(s[0]), k.term(s[1])] for s in step[2]],
            )

        if step[0] == 'INST_TYPE':
            thm = fusion.INST_TYPE(
                step[1],
                [[k.type(s[0]), k.type(s[1])] for s in step[2]],
            )

        if thm is None:
            Log.out("NOT IMPLEMENTED", {
                'action': step[0],
            })
            return

        # Reinsert the theorem where it belongs in the fusion kernel
        thm._index = i
        fusion.PREMISE(thm)

        org = Thm(
            i,
            [k.term(hy) for hy in k._theorems[i]['hy']],
            k.term(k._theorems[i]['cc']),
        )

        Log.out("STEP", {
            'index': i,
            'rule': step[0],
        })

        if thm.thm_string() != org.thm_string():
            Log.out("DIVERGENCE", {
                'org': org.thm_string(),
            })
            Log.out("DIVERGENCE", {
                'thm': org.thm_string(),
            })
            return
