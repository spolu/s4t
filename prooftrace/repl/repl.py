import argparse
import os
import pexpect
import pexpect.replwrap

from utils.config import Config


class REPL():
    def init(
            self,
    ) -> None:
        pass


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

    ocaml_path = os.path.expanduser(
        config.get("prooftrace_repl_ocaml_path"),
    )
    camlp5_path = os.path.expanduser(
        config.get("prooftrace_repl_camlp5_path"),
    )
    hol_ml_path = os.path.expanduser(
        os.path.join(
            config.get("prooftrace_repl_hol_light_path"),
            "hol.ml",
        )
    )

    print("============================")
    print("ProofTrace REPL testing \\o/")
    print("----------------------------")

    ocaml = pexpect.replwrap.REPLWrapper(
        ocaml_path,
        "# ",
        None,
        extra_init_cmd="#use \"{}\"".format(hol_ml_path),
    )
    # child = pexpect.spawn(
    #     ocaml_path,
    #     args=['-I', camlp5_path, 'camlp5o.cma'],
    #     echo=False,
    # )
    # child.expect('\r\n# ')
    # child.sendline ('2+3;;')
    # child.expect('\r\n# ')
    # print(child.before)
    out = ocaml.run_command('2+3;;')
    print(out)
    out = ocaml.run_command('5+3;;')
    print(out)
