import argparse
import os
import pexpect
import pexpect.replwrap
import pickle

from dataset.prooftrace import ProofTraceActions

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
