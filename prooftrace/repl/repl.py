import argparse
import gzip
import os
import pickle
import re

from prooftrace.prooftrace import \
    PROOFTRACE_TOKENS, INV_PROOFTRACE_TOKENS, INV_ACTION_TOKENS, \
    ProofTraceTokenizer, Action, ProofTraceActions, TypeException

from prooftrace.repl.fusion import Fusion, Thm, FusionException

from utils.config import Config
from utils.log import Log


class REPLException(Exception):
    pass


class REPL():
    def __init__(
            self,
            tokenizer: ProofTraceTokenizer,
    ) -> None:
        self._fusion = Fusion(tokenizer)

    def build_hypothesis(
            self,
            hyp: Action,
    ):
        if hyp is None:
            return []
        if INV_PROOFTRACE_TOKENS[hyp.value] == 'HYPOTHESIS':
            return [hyp.left.value] + self.build_hypothesis(hyp.right)
        raise REPLException()

    def valid(
            self,
            action: Action,
    ) -> bool:
        try:
            self.apply(action, True)
        except (FusionException, REPLException, TypeException):
            return False
        return True

    def apply(
            self,
            action: Action,
            fake: bool = False,
    ) -> Thm:
        action_token = INV_PROOFTRACE_TOKENS[action.value]

        thm = None

        if action_token == 'PREMISE':
            thm = Thm(
                action.index(),
                self.build_hypothesis(action.left),
                action.right.value,
            )
            thm = self._fusion.PREMISE(thm, fake)
        elif action_token == 'REFL':
            if action.left.value != PROOFTRACE_TOKENS['TERM']:
                raise REPLException
            if action.right.value != PROOFTRACE_TOKENS['EMPTY']:
                raise REPLException
            thm = self._fusion.REFL(
                action.left.left.value,
                fake,
            )
        elif action_token == 'TRANS':
            thm = self._fusion.TRANS(
                action.left.index(),
                action.right.index(),
                fake,
            )
        elif action_token == 'MK_COMB':
            thm = self._fusion.MK_COMB(
                action.left.index(),
                action.right.index(),
                fake,
            )
        elif action_token == 'ABS':
            if action.right.value != PROOFTRACE_TOKENS['TERM']:
                raise REPLException
            thm = self._fusion.ABS(
                action.left.index(),
                action.right.left.value,
                fake,
            )
        elif action_token == 'BETA':
            if action.left.value != PROOFTRACE_TOKENS['TERM']:
                raise REPLException
            if action.right.value != PROOFTRACE_TOKENS['EMPTY']:
                raise REPLException
            thm = self._fusion.BETA(
                action.left.left.value,
                fake,
            )
        elif action_token == 'ASSUME':
            if action.left.value != PROOFTRACE_TOKENS['TERM']:
                raise REPLException
            if action.right.value != PROOFTRACE_TOKENS['EMPTY']:
                raise REPLException
            thm = self._fusion.ASSUME(
                action.left.left.value,
                fake,
            )
        elif action_token == 'EQ_MP':
            thm = self._fusion.EQ_MP(
                action.left.index(),
                action.right.index(),
                fake,
            )
        elif action_token == 'DEDUCT_ANTISYM_RULE':
            thm = self._fusion.DEDUCT_ANTISYM_RULE(
                action.left.index(),
                action.right.index(),
                fake,
            )
        elif action_token == 'INST':
            def build_subst(subst):
                if subst is None:
                    return []
                if INV_PROOFTRACE_TOKENS[subst.value] == 'SUBST_PAIR':
                    return [[
                        subst.left.value,
                        subst.right.value,
                    ]]
                if INV_PROOFTRACE_TOKENS[subst.value] == 'SUBST':
                    return (
                        build_subst(subst.left) +
                        build_subst(subst.right)
                    )
                raise REPLException()

            if action.right.value != PROOFTRACE_TOKENS['SUBST']:
                raise REPLException

            thm = self._fusion.INST(
                action.left.index(),
                build_subst(action.right),
                fake,
            )
        elif action_token == 'INST_TYPE':
            def build_subst_type(subst_type):
                if subst_type is None:
                    return []
                if INV_PROOFTRACE_TOKENS[subst_type.value] == 'SUBST_PAIR':
                    return [[
                        subst_type.left.value,
                        subst_type.right.value,
                    ]]
                if INV_PROOFTRACE_TOKENS[subst_type.value] == 'SUBST_TYPE':
                    return (
                        build_subst_type(subst_type.left) +
                        build_subst_type(subst_type.right)
                    )
                raise REPLException()

            if action.right.value != PROOFTRACE_TOKENS['SUBST_TYPE']:
                raise REPLException

            thm = self._fusion.INST_TYPE(
                action.left.index(),
                build_subst_type(action.right),
                fake,
            )
        else:
            raise REPLException()

        try:
            return thm
        except AssertionError:
            Log.out("Action replay failure", {
                'action_token': action_token,
            })
            raise

    def replay(
            self,
            ptra: ProofTraceActions,
    ) -> Thm:
        for i, a in enumerate(ptra.actions()):
            if i == 0:
                target = Thm(
                    a.index(),
                    self.build_hypothesis(a.left),
                    a.right.value,
                )
            if a.value in INV_ACTION_TOKENS:
                index = self.apply(a).index()
                ptra.actions()[i]._index = index
                ptra.arguments()[i]._index = index

                # ground = Thm(
                #     index,
                #     self.build_hypothesis(ptra.arguments()[i].left),
                #     ptra.arguments()[i].right.value,
                # )
                # assert self._fusion._theorems[index].thm_string() == \
                #     ground.thm_string()

        last = self._fusion._theorems[ptra.actions()[-2].index()]
        assert last.thm_string() == target.thm_string()

        return last

    def prepare(
            self,
            ptra: ProofTraceActions,
    ) -> Thm:
        for i, a in enumerate(ptra.actions()):
            if a.value == PROOFTRACE_TOKENS['TARGET']:
                target = Thm(
                    ptra.actions()[i].index(),
                    self.build_hypothesis(ptra.actions()[i].left),
                    ptra.actions()[i].right.value,
                )
            if a.value == PROOFTRACE_TOKENS['PREMISE']:
                self.apply(a)

        return target

    def copy(
            self,
    ):
        repl = REPL(self._fusion._t)
        repl._fusion = self._fusion.copy()

        return repl


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

    print("============================")
    print("ProofTrace REPL testing \\o/")
    print("----------------------------")

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    dataset_dir = "./data/prooftrace/{}/train_traces".format(
        config.get("prooftrace_dataset_size"),
    )
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    for p in files:
        if re.search("\\.actions$", p) is None:
            continue
        with gzip.open(p, 'rb') as f:
            ptra = pickle.load(f)

        Log.out("Replaying ProofTraceActions", {
            "path": p,
            "actions_count": len(ptra.actions()),
        })

        repl = REPL(tokenizer)
        repl.prepare(ptra)
        repl.replay(ptra)
