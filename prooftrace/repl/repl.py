import argparse
import os
import pexpect
import pexpect.replwrap
# import pickle

from prooftrace.repl.actions import \
    ProofIndex, Term, \
    REFL, TRANS, MK_COMB, ABS, BETA, ASSUME, EQ_MP, DEDUCT_ANTISYM_RULE

from utils.config import Config
from utils.log import Log


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

        self._var_index = 0

    def prepare(
            self,
    ):
        hol_ml_path = os.path.expanduser(
            os.path.join(
                self._config.get("prooftrace_repl_hol_light_path"),
                "hol.ml",
            )
        )

        self.run(
            "#use \"{}\";;".format(hol_ml_path),
            timeout=None,
        )

    def next_var(
            self,
    ) -> str:
        self._var_index += 1
        return "___" + str(self._var_index)

    def run(
            self,
            cmd: str,
            timeout=-1,
    ) -> str:
        return self._ocaml.run_command(cmd, timeout)


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

    # p = \
    #   'data/prooftrace/small/train_traces/33892_LEFT_EXISTS_IMP_THM.actions'
    # ptra = None
    # with open(p, 'rb') as f:
    #     ptra = pickle.load(f)

    # print(ptra.actions()[0].left.left.value.term_string())
    # i = 2
    # while ptra.actions()[i].value == 6:
    #     print(ptra.actions()[i].left.value.term_string())
    #     i += 1

    repl = REPL(config)

    repl.prepare()
    Log.out("PREPARED")

    try:
        proof_index = REFL([Term('q')]).run(repl)
        Log.out("REFL `q` = [64]", {
            "proof_index": proof_index,
        })

        proof_index = TRANS([ProofIndex(79), ProofIndex(80)]).run(repl)
        Log.out("TRANS 79 80 = [81]", {
            "proof_index": proof_index,
        })

        proof_index = MK_COMB([ProofIndex(74), ProofIndex(75)]).run(repl)
        Log.out("MK_COMB 74 75 = [76]", {
            "proof_index": proof_index,
        })

        proof_index = ABS([ProofIndex(498), Term('Q')]).run(repl)
        Log.out("ABS 498 `Q` = [499]", {
            "proof_index": proof_index,
        })

        proof_index = BETA([Term('(\\x.T) x')]).run(repl)
        Log.out("BETA `(\\x.T) x` = [430]", {
            "proof_index": proof_index,
        })

        proof_index = ASSUME([Term('(!) P')]).run(repl)
        Log.out("BETA `(!) P` = [422]", {
            "proof_index": proof_index,
        })

        proof_index = EQ_MP([ProofIndex(432), ProofIndex(429)]).run(repl)
        Log.out("EQ_MP 432 429 = [433]", {
            "proof_index": proof_index,
        })

        proof_index = DEDUCT_ANTISYM_RULE([
            ProofIndex(413), ProofIndex(417)
        ]).run(repl)
        Log.out("DEDUCT_ANTISYM_RULE 413 417 = [418]", {
            "proof_index": proof_index,
        })

    except Exception as e:
        print(e)
        import pdb
        pdb.set_trace()
