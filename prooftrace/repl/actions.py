import typing
import re


class Argument():
    """ Argument are attached to Actions sent to the REPL

    Arguments can either be:
    - ProofIndex: the index of a proof we rely on to generate a new proof.
    - Term: a term string we rely on to generate a new proof.
    """
    pass


class ProofIndex(Argument):
    def __init__(
            self,
            index: int,
    ) -> None:
        self._index = index

    def index(
            self,
    ) -> int:
        return self._index


class Term(Argument):
    def __init__(
            self,
            term: str,
    ) -> None:
        self._term = term

    def term(
            self,
    ) -> str:
        return self._term


class Subst(Argument):
    def __init__(
            self,
            subst: typing.List[typing.List[str]],
    ) -> None:
        self._subst = subst

    def subst(
            self,
    ) -> typing.List[typing.List[str]]:
        return self._subst


class SubstType(Argument):
    def __init__(
            self,
            subst_type: typing.List[typing.List[str]],
    ) -> None:
        self._subst_type = subst_type

    def subst_type(
            self,
    ) -> typing.List[typing.List[str]]:
        return self._subst_type


class Action():
    """ Actions are sent to the REPL to generate a new proof
    """
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        self._args = args

    def run(
            self,
            repl,
    ) -> int:
        raise Exception("Not implemented.")


class REFL(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(REFL, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let th = REFL `{}` in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ");;").format(
                 self._args[0].term(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class TRANS(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(TRANS, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let Proof(_,th0,_) = proof_at {} in ("
             "let Proof(_,th1,_) = proof_at {} in ("
             "let th = TRANS th0 th1 in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ")));;").format(
                 self._args[0].index(),
                 self._args[1].index(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class MK_COMB(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(MK_COMB, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let Proof(_,th0,_) = proof_at {} in ("
             "let Proof(_,th1,_) = proof_at {} in ("
             "let th = MK_COMB (th0, th1) in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ")));;").format(
                 self._args[0].index(),
                 self._args[1].index(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class ABS(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(ABS, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is Term

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let Proof(_,th0,_) = proof_at {} in ("
             "let th = ABS `{}` th0 in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             "));;").format(
                 self._args[0].index(),
                 self._args[1].term(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class BETA(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(BETA, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let th = BETA `{}` in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ");;").format(
                 self._args[0].term(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class ASSUME(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(ASSUME, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let th = ASSUME `{}` in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ");;").format(
                 self._args[0].term(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class EQ_MP(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(EQ_MP, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let Proof(_,th0,_) = proof_at {} in ("
             "let Proof(_,th1,_) = proof_at {} in ("
             "let th = EQ_MP th0 th1 in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ")));;").format(
                 self._args[0].index(),
                 self._args[1].index(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class DEDUCT_ANTISYM_RULE(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(DEDUCT_ANTISYM_RULE, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            ("let Proof(_,th0,_) = proof_at {} in ("
             "let Proof(_,th1,_) = proof_at {} in ("
             "let th = DEDUCT_ANTISYM_RULE th0 th1 in (" +
             "let Proof(idx,_, _) = (proof_of th) in idx" +
             ")));;").format(
                 self._args[0].index(),
                 self._args[1].index(),
            ),
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class INST(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(INST, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is Subst

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            "let Proof(_,th0,_) = proof_at {} in (".format(
                self._args[0].index(),
            ) +
            "let th = INST (" +
            "::".join([
                "(`" + s[1] + "`,`" + s[0] + "`)"
                for s in self._args[1].subst()
            ] + ['[]']) +
            ") th0 in (" +
            "let Proof(idx,_, _) = (proof_of th) in idx" +
            "));;",
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))


class INST_TYPE(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(INST_TYPE, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is SubstType

    def run(
            self,
            repl,
    ) -> int:
        out = repl.run(
            "let Proof(_,th0,_) = proof_at {} in (".format(
                self._args[0].index(),
            ) +
            "let th = INST_TYPE (" +
            "::".join([
                "(`" + s[1] + "`,`" + s[0] + "`)"
                for s in self._args[1].subst_type()
            ] + ['[]']) +
            ") th0 in (" +
            "let Proof(idx,_, _) = (proof_of th) in idx" +
            "));;",
        )
        match = re.search('val it : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))
