import argparse
import os
import pexpect
import pickle
import re

from dataset.prooftrace import \
    ACTION_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.fusion import Fusion, Thm

from utils.config import Config
from utils.log import Log


class REPL():
    def __init__(
            self,
            config: Config,
    ) -> None:
        self._config = config

        with open(
                os.path.join(
                    os.path.expanduser(config.get('prooftrace_dataset_dir')),
                    config.get('prooftrace_dataset_size'),
                    'traces.tokenizer',
                ), 'rb') as f:
            self._t = pickle.load(f)

    def build_hypothesis(
            self,
            hyp: Action,
    ):
        if hyp is None:
            return []
        if INV_ACTION_TOKENS[hyp.value] == 'HYPOTHESIS':
            return [hyp.left.value] + self.build_hypothesis(hyp.right)
        assert False

    def apply(
            self,
            fusion: Fusion,
            action: Action,
    ) -> int:
        action_token = INV_ACTION_TOKENS[action.value]

        thm = None

        if action_token == 'PREMISE':
            thm = Thm(
                action.index(),
                self.build_hypothesis(action.left),
                action.right.value,
            )
            thm = fusion.PREMISE(thm)
        elif action_token == 'REFL':
            thm = fusion.REFL(
                action.left.left.value
            )
        elif action_token == 'TRANS':
            thm = fusion.TRANS(
                action.left.index(),
                action.right.index(),
            )
        elif action_token == 'MK_COMB':
            thm = fusion.MK_COMB(
                action.left.index(),
                action.right.index(),
            )
        elif action_token == 'ABS':
            thm = fusion.ABS(
                action.left.index(),
                action.right.left.value
            )
        elif action_token == 'BETA':
            thm = fusion.BETA(
                action.left.left.value
            )
        elif action_token == 'ASSUME':
            thm = fusion.ASSUME(
                action.left.left.value
            )
        elif action_token == 'EQ_MP':
            thm = fusion.EQ_MP(
                action.left.index(),
                action.right.index(),
            )
        elif action_token == 'DEDUCT_ANTISYM_RULE':
            thm = fusion.DEDUCT_ANTISYM_RULE(
                action.left.index(),
                action.right.index(),
            )
        elif action_token == 'INST':
            def build_subst(subst):
                if subst is None:
                    return []
                if INV_ACTION_TOKENS[subst.value] == 'SUBST_PAIR':
                    return [[
                        subst.left.value,
                        subst.right.value,
                    ]]
                if INV_ACTION_TOKENS[subst.value] == 'SUBST':
                    return (
                        build_subst(subst.left) +
                        build_subst(subst.right)
                    )
                assert False
            thm = fusion.INST(
                action.left.index(),
                build_subst(action.right),
            )
        elif action_token == 'INST_TYPE':
            def build_subst_type(subst_type):
                if subst_type is None:
                    return []
                if INV_ACTION_TOKENS[subst_type.value] == 'SUBST_PAIR':
                    return [[
                        subst_type.left.value,
                        subst_type.right.value,
                    ]]
                if INV_ACTION_TOKENS[subst_type.value] == 'SUBST_TYPE':
                    return (
                        build_subst_type(subst_type.left) +
                        build_subst_type(subst_type.right)
                    )
                assert False
            thm = fusion.INST_TYPE(
                action.left.index(),
                build_subst_type(action.right),
            )
        else:
            assert False

        try:
            return thm.index()
        except AssertionError:
            Log.out("Action replay failure", {
                'action_token': action_token,
            })
            raise

    def replay(
            self,
            ptra: ProofTraceActions,
    ) -> None:
        fusion = Fusion(self._t)

        for a in ptra.actions():
            if a.value == ACTION_TOKENS['TARGET']:
                target = Thm(
                    a.index(),
                    self.build_hypothesis(a.left),
                    a.right.value,
                )
            if a.value not in \
                    [
                        ACTION_TOKENS['TARGET'],
                        ACTION_TOKENS['EMPTY'],
                        ACTION_TOKENS['SUBST'],
                        ACTION_TOKENS['SUBST_TYPE'],
                        ACTION_TOKENS['TERM'],
                    ]:

                a._index = self.apply(fusion, a)

        last = fusion._theorems[ptra.actions()[-1].index()]
        assert last.thm_string() == target.thm_string()


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

    repl = REPL(config)

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

        repl.replay(ptra)
