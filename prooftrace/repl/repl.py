import argparse
import os
import pexpect
import pexpect.replwrap
import pickle
import re

from dataset.prooftrace import \
    ACTION_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.actions import \
    ProofIndex, Term, Subst, SubstType, \
    REFL, TRANS, MK_COMB, ABS, BETA, ASSUME, EQ_MP, DEDUCT_ANTISYM_RULE, \
    INST, INST_TYPE

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

        self._ocaml = pexpect.spawn('/bin/bash', echo=False)
        self._ocaml.sendline('stty -icanon')
        self._ocaml.sendline(
            ocaml_path + " -I " + camlp5_path + " camlp5o.cma"
        )
        self._ocaml.expect_exact(["# "])

        self._var_index = 0

    def prepare(
            self,
            dataset_size
    ):
        hol_ml_path = os.path.expanduser(
            os.path.join(
                self._config.get("prooftrace_repl_hol_light_path"),
                "hol_{}.ml".format(dataset_size),
            )
        )

        self.run(
            "#use \"{}\";;".format(hol_ml_path),
            timeout=None,
            no_print=True,
        )

    def next_var(
            self,
    ) -> str:
        self._var_index += 1
        return "___" + str(self._var_index)

    def run(
            self,
            cmd: str,
            timeout=4,
            no_print=False,
    ) -> str:
        # Log.out("Running", {
        #     "command": cmd,
        # })
        self._ocaml.sendline(cmd)
        res = self._ocaml.expect_exact(["# "], timeout=timeout)
        assert res == 0
        out = self._ocaml.before.decode('utf-8')
        # if not no_print:
        #     Log.out("Run", {
        #         "command": cmd,
        #         "output": out,
        #     })
        return out

    def apply(
            self,
            action: Action,
    ) -> int:
        action_token = INV_ACTION_TOKENS[action.value]
        repl_action = None

        if action_token == 'REFL':
            repl_action = REFL([
                Term(action.left.left.value.term_string())
            ])
        elif action_token == 'TRANS':
            repl_action = TRANS([
                ProofIndex(action.left.index()),
                ProofIndex(action.right.index()),
            ])
        elif action_token == 'MK_COMB':
            repl_action = MK_COMB([
                ProofIndex(action.left.index()),
                ProofIndex(action.right.index()),
            ])
        elif action_token == 'ABS':
            repl_action = ABS([
                ProofIndex(action.left.index()),
                Term(action.right.left.value.term_string())
            ])
        elif action_token == 'BETA':
            repl_action = BETA([
                Term(action.left.left.value.term_string())
            ])
        elif action_token == 'ASSUME':
            repl_action = ASSUME([
                Term(action.left.left.value.term_string())
            ])
        elif action_token == 'EQ_MP':
            repl_action = EQ_MP([
                ProofIndex(action.left.index()),
                ProofIndex(action.right.index()),
            ])
        elif action_token == 'DEDUCT_ANTISYM_RULE':
            repl_action = DEDUCT_ANTISYM_RULE([
                ProofIndex(action.left.index()),
                ProofIndex(action.right.index()),
            ])
        elif action_token == 'INST':
            def build_subst(subst):
                if subst is None:
                    return []
                if INV_ACTION_TOKENS[subst.value] == 'SUBST_PAIR':
                    return [[
                        subst.left.value.term_string(),
                        subst.right.value.term_string(),
                    ]]
                if INV_ACTION_TOKENS[subst.value] == 'SUBST':
                    return (
                        build_subst(subst.left) +
                        build_subst(subst.right)
                    )
                assert False

            repl_action = INST([
                ProofIndex(action.left.index()),
                Subst(build_subst(action.right)),
            ])
        elif action_token == 'INST_TYPE':
            def build_subst_type(subst_type):
                if subst_type is None:
                    return []
                if INV_ACTION_TOKENS[subst_type.value] == 'SUBST_PAIR':
                    return [[
                        subst_type.left.value.type_string(),
                        subst_type.right.value.type_string(),
                    ]]
                if INV_ACTION_TOKENS[subst_type.value] == 'SUBST_TYPE':
                    return (
                        build_subst_type(subst_type.left) +
                        build_subst_type(subst_type.right)
                    )
                assert False

            repl_action = INST_TYPE([
                ProofIndex(action.left.index()),
                SubstType(build_subst_type(action.right)),
            ])
        else:
            assert False

        try:
            return repl_action.run(self)
        except AssertionError:
            Log.out("Action replay failure", {
                'action_token': action_token,
            })
            raise

    def replay(
            self,
            ptra: ProofTraceActions,
    ) -> None:
        for a in ptra.actions():
            if a.value not in \
                    [
                        ACTION_TOKENS['TARGET'],
                        ACTION_TOKENS['EMPTY'],
                        ACTION_TOKENS['SUBST'],
                        ACTION_TOKENS['SUBST_TYPE'],
                        ACTION_TOKENS['TERM'],
                        ACTION_TOKENS['PREMISE'],
                    ]:
                # from_proof_index = a.index()

                # Log.out("Replaying action", {
                #     "action":  INV_ACTION_TOKENS[a.value],
                #     "from_proof_index": from_proof_index,
                # })

                a._index = self.apply(a)

                # Log.out("Replayed action", {
                #     "action":  INV_ACTION_TOKENS[a.value],
                #     "from_proof_index": from_proof_index,
                #     "to_proof_index": a.index(),
                # })


class Pool():
    pass


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

    Log.out("Preparing HOL Light")
    repl.prepare(config.get('prooftrace_dataset_size'))
    Log.out("Prepared")

    # p = "./data/prooftrace/{}/".format(
    #     config.get("prooftrace_dataset_size"),
    # ) + \
    #     "test_traces/14850733_FLOOR_UNIQUE.actions"
    # # "test_traces/113466_EXISTS_TRIPLED_THM.actions"
    # # "test_traces/111905_EXISTS_PAIRED_THM.actions"
    # # "test_traces/136772_SHARED_136772.actions"

    # # "test_traces/139594_SHARED_139594.actions"
    # # "test_traces/137154_ZERO_DEF.actions"
    # # "test_traces/131615_ONTO.actions"
    # # "test_traces/134420_IND_SUC_0_EXISTS.actions"
    # # "train_traces/40114_bool_INDUCT.actions"
    # # "test_traces/13250313_GCD_LMUL.actions"

    # with open(p, 'rb') as f:
    #     ptra = pickle.load(f)

    #     Log.out("Replaying ProofTraceActions", {
    #         "path": p,
    #         "actions_count": len(ptra.actions()),
    #     })

    #     repl.replay(ptra)

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

        try:
            repl.replay(ptra)
        except AssertionError:
            Log.out("Replay FAILURE")

# [20190311_1344_23.424462] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/111132_FORALL_PAIRED_THM.actions actions_count=468
# [20190311_1344_40.580587] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/136778_NUM_REP_RULES.actions actions_count=8
# [20190311_1344_40.750817] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/110713_EXISTS_UNPAIR_FUN_THM.actions actions_count=283
# [20190311_1344_51.162581] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/140114_SUC_INJ.actions actions_count=517
# [20190311_1345_12.410976] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/136781_NUM_REP_CASES.actions actions_count=8
# [20190311_1345_12.593455] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/111905_EXISTS_PAIRED_THM.actions actions_count=770
# [20190311_1345_28.733598] Replay FAILURE
# [20190311_1345_28.738119] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/131226_ONE_ONE.actions actions_count=53
# [20190311_1345_30.466347] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/146205_num_INDUCTION.actions actions_count=29
# [20190311_1345_31.509633] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/134420_IND_SUC_0_EXISTS.actions actions_count=2028
# [20190311_1347_00.412423] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/110427_FORALL_UNPAIR_FUN_THM.actions actions_count=283
# [20190311_1347_10.900485] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/146187_NOT_SUC.actions actions_count=21
# [20190311_1347_11.584373] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/136772_SHARED_136772.actions actions_count=1601
# [20190311_1347_21.842229] Replay FAILURE
# [20190311_1347_21.847842] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/136775_SHARED_136775.actions actions_count=8
# [20190311_1347_22.025337] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/147312_num_CASES.actions actions_count=1142
# [20190311_1348_11.990096] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/137154_ZERO_DEF.actions actions_count=14
# [20190311_1348_12.448175] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/139594_SHARED_139594.actions actions_count=1654
# [20190311_1349_23.527939] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/146175_SHARED_146175.actions actions_count=92
# [20190311_1349_26.757410] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/113466_EXISTS_TRIPLED_THM.actions actions_count=963
# [20190311_1349_48.830881] Replay FAILURE
# [20190311_1349_48.875308] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/115216_CHOICE_PAIRED_THM.actions actions_count=1656
# [20190311_1350_59.811835] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/135405_SHARED_135405.actions actions_count=608
# [20190311_1351_24.340707] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/112480_FORALL_TRIPLED_THM.actions actions_count=626
# [20190311_1351_44.291550] Replay FAILURE
# [20190311_1351_44.504435] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/146229_num_Axiom.actions actions_count=4912
# [20190311_1353_11.671291] Replay FAILURE
# [20190311_1353_11.696073] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/113521_CHOICE_UNPAIR_THM.actions actions_count=72
# [20190311_1353_11.962242] Replay FAILURE
# [20190311_1353_11.963585] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/137567_SUC_DEF.actions actions_count=53
# [20190311_1353_13.713598] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/141032_SHARED_141032.actions actions_count=875
# [20190311_1353_51.059667] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/136784_NUM_REP_INDUCT.actions actions_count=8
# [20190311_1353_51.229026] Replaying ProofTraceActions: path=./data/prooftrace/small/test_traces/131615_ONTO.actions actions_count=53
