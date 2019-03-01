import os
import pexpect


class REPL():
    def init(
            self,
    ) -> None:
        pass


class Pool():
    pass


def test():
    print("ProofTrace REPL testing \o/")

    child = pexpect.spawn(
        'ocaml',
        args=['-I', '/home/stan/.opam/4.06.1/lib/camlp5', 'camlp5o.cma'],
        echo=False,
    )
    child.expect('# ')
    child.sendline ('2+3;;')
    child.expect('# ')
    print(child.before)
