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

    def proof_index(
            self,
            repl,
            thm_var,
    ) -> int:
        proof_var = repl.next_var()

        out = repl.run("let Proof({}, _, _) = proof_of {};;".format(
            proof_var,
            thm_var,
        ))

        match = re.search('^val ' + proof_var + ' : int = (\\d+)\r\n$', out)
        assert match

        return int(match.group(1))

    def var_for_theorem(
            self,
            repl,
            proof_index,
    ) -> str:
        thm_var = repl.next_var()

        out = repl.run("let Proof(_, {}, _) = proof_at {};;".format(
            thm_var,
            proof_index,
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return thm_var


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
        thm_var = repl.next_var()

        out = repl.run("let {} = REFL `{}`;;".format(
            thm_var,
            self._args[0].term(),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = TRANS {} {};;".format(
            thm_var,
            self.var_for_theorem(repl, self._args[0].index()),
            self.var_for_theorem(repl, self._args[1].index()),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = MK_COMB ({}, {});;".format(
            thm_var,
            self.var_for_theorem(repl, self._args[0].index()),
            self.var_for_theorem(repl, self._args[1].index()),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = ABS `{}` {};;".format(
            thm_var,
            self._args[1].term(),
            self.var_for_theorem(repl, self._args[0].index()),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = BETA `{}`;;".format(
            thm_var,
            self._args[0].term(),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = ASSUME `{}`;;".format(
            thm_var,
            self._args[0].term(),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = EQ_MP {} {};;".format(
            thm_var,
            self.var_for_theorem(repl, self._args[0].index()),
            self.var_for_theorem(repl, self._args[1].index()),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        out = repl.run("let {} = DEDUCT_ANTISYM_RULE {} {};;".format(
            thm_var,
            self.var_for_theorem(repl, self._args[0].index()),
            self.var_for_theorem(repl, self._args[1].index()),
        ))
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)


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
        thm_var = repl.next_var()

        cmd = "let {} = INST ({}::[]) {};;".format(
            thm_var,
            "::".join(
                [
                    "(`" + s[0] + "`, `" + s[1] + "`)"
                    for s in self._args[1].subst()
                ],
            ),
            self.var_for_theorem(repl, self._args[0].index()),
        )
        print(cmd)
        out = repl.run(cmd)
        print(out)
        assert re.search(
            'val ' + thm_var + ' : thm =.*\r\n$', out, flags=re.DOTALL
        )

        return self.proof_index(repl, thm_var)
