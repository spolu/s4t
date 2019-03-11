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
        Log.out("Running", {
            "command": cmd,
        })
        out = self._ocaml.run_command(cmd, timeout)
        if not no_print:
            Log.out("Run", {
                "command": cmd,
                "output": out,
            })
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

        return repl_action.run(self)

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

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    print("============================")
    print("ProofTrace REPL testing \\o/")
    print("----------------------------")

    repl = REPL(config)

    Log.out("Preparing HOL Light")
    repl.prepare()
    Log.out("Prepared")

    p = "./data/prooftrace/small/" + \
        "test_traces/111905_EXISTS_PAIRED_THM.actions"

    # "test_traces/113466_EXISTS_TRIPLED_THM.actions"
    # "test_traces/111905_EXISTS_PAIRED_THM.actions"
    # "test_traces/136772_SHARED_136772.actions"
    # "test_traces/139594_SHARED_139594.actions"

    # "test_traces/139594_SHARED_139594.actions"
    # "test_traces/137154_ZERO_DEF.actions"
    # "test_traces/131615_ONTO.actions"
    # "test_traces/134420_IND_SUC_0_EXISTS.actions"
    # "test_traces/113466_EXISTS_TRIPLED_THM.actions"
    # "train_traces/40114_bool_INDUCT.actions"

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

    # files = [
    #     os.path.join("./data/prooftrace/small/test_traces", f)
    #     for f in os.listdir("./data/prooftrace/small/test_traces")
    # ]
    # for p in files:
    #     if re.search("\\.actions$", p) is None:
    #         continue
    #     with open(p, 'rb') as f:
    #         ptra = pickle.load(f)

    #     Log.out("Replaying ProofTraceActions", {
    #         "path": p,
    #         "actions_count": len(ptra.actions()),
    #     })

    #     try:
    #         repl.replay(ptra)
    #     except AssertionError:
    #         Log.out("Replay FAILURE")
