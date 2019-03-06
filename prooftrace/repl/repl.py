import argparse
import os
import pexpect
import pexpect.replwrap
import pickle
import typing

from utils.config import Config


class Argument():
    pass


class ProofIndex(Argument):
    def __init__(
            self,
            index: int,
    ) -> None:
        self._index = index


class Term(Argument):
    def __init__(
            self,
            term: str,
    ) -> None:
        self._term = term


class Action():
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        self._args = args


class REFL(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(REFL, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term


class TRANS(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(TRANS, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex


class MK_COMB(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(MK_COMB, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex


class ABS(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(ABS, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is Term
        assert type(args[1]) is ProofIndex


class BETA(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(BETA, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term


class ASSUME(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(BETA, self).__init__(args)
        assert len(args) == 1
        assert type(args[0]) is Term


class EQ_MP(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(EQ_MP, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex


class DEDUCT_ANTISYM(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(DEDUCT_ANTISYM, self).__init__(args)
        assert len(args) == 2
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is ProofIndex


class INST(Action):
    def __init__(
            self,
            args: typing.List[Argument],
    ) -> None:
        super(INST, self).__init__(args)
        assert len(args) == 3
        assert type(args[0]) is ProofIndex
        assert type(args[1]) is Term
        assert type(args[2]) is Term


class REPL():
    def __init__(
            self,
            config: Config,
    ) -> None:
        self._config = config

        ocaml_path = os.path.expanduser(
            config.get("prooftrace_repl_ocaml_path"),
        )
        camlp5_path = os.path.expanduser(
            config.get("prooftrace_repl_camlp5_path"),
        )

        self._ocaml = pexpect.replwrap.REPLWrapper(
            ocaml_path + " -I " + camlp5_path + " camlp5o.cma",
            "# ",
            None,
        )

    def prepare(
            self,
    ):
        hol_ml_path = os.path.expanduser(
            os.path.join(
                self._config.get("prooftrace_repl_hol_light_path"),
                "hol.ml",
            )
        )

        self._ocaml.run_command(
            "#use \"{}\";;".format(hol_ml_path),
            timeout=None,
        )


class Pool():
    pass


def test():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    print("============================")
    print("ProofTrace REPL testing \\o/")
    print("----------------------------")

    # child = pexpect.spawn(
    #     ocaml_path,
    #     args=['-I', camlp5_path, 'camlp5o.cma'],
    #     echo=False,
    # )
    # child.expect('\r\n# ')
    # child.sendline ('2+3;;')

    # repl = REPL(config)

    # Log.out("Preparing REPL")
    # repl.prepare()
    # Log.out("Done")

    # out = repl._ocaml.run_command("2+3;;")
    # print(out)

    p = 'data/prooftrace/small/train_traces/33892_LEFT_EXISTS_IMP_THM.actions'
    ptra = None
    with open(p, 'rb') as f: 
        ptra = pickle.load(f)

    print(ptra.actions()[0].left.left.value.term_string())
    i = 2
    while ptra.actions()[i].value == 6:
        print(ptra.actions()[i].left.value.term_string())
        i += 1
