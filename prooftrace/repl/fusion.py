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
            hypotheses: typing.List[Term],
            conclusion: Term,
    ):
        self._hypotheses = hypotheses
        self._conclusion = conclusion


class Kernel():
    def __init__(
            self,
    ):
        self._theorems = []

    def token_term(
            self,
            token,
    ) -> Type:
        if token == '=':
            return Term(4, None, None, '=')
        assert False

    def token_type(
            self,
            token,
    ) -> Type:
        if token == 'fun':
            return Type(3, None, None, 'fun')
        if token == 'bool':
            return Type(4, None, None, 'fun')
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
                0,
                Term(2,
                     self.token_term('='),
                     Term(
                         self.fun_type(
                             ty,
                             self.fun_type(ty, self.bool_type())
                         ),
                         None, None, None,
                     ),
                     '__c'),
                left,
                '__C',
            ),
            right,
            '__C',
        )

    def REFL(
            self,
            term: Term,
    ):
        thm = Thm(
            [],
            self.safe_mk_eq(term, term),
        )
        self._theorems.append(thm)
        return thm


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
            if action_token == 'REFL':
                thm = kernel.REFL(a.left.left.value)
                import pdb; pdb.set_trace()
