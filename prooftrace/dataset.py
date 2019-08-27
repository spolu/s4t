import gzip
import os
import pickle
import re
import typing

from prooftrace.prooftrace import PREPARE_TOKENS, Action, ProofTraceTokenizer

from torch.utils.data import Dataset

from utils.log import Log


def lm_collate(
        batch
) -> typing.Tuple[
    typing.List[typing.List[Action]],
    typing.List[typing.List[Action]],
    typing.List[Action],
]:
    actions = []
    arguments = []
    truths = []

    for (act, arg, trh) in batch:
        actions.append(act)
        arguments.append(arg)
        truths.append(trh)

    return (actions, arguments, truths)


def trh_extract(
        trh,
        arg,
) -> typing.Tuple[
    typing.List[typing.List[int]],
    typing.List[typing.List[int]],
    typing.List[typing.List[int]],
]:
    trh_actions = []
    trh_lefts = []
    trh_rights = []
    for b in range(len(trh)):
        trh_actions += [[]]
        trh_lefts += [[]]
        trh_rights += [[]]
        for i in range(len(trh[b])):
            trh_actions[b] += [trh[b][i].value - len(PREPARE_TOKENS)]
            if trh[b][i].value == 0 or trh[b][i].value == 21:
                trh_lefts[b] += [1]
                trh_rights[b] += [1]
            else:
                trh_lefts[b] += [arg[b].index(trh[b][i].left)]
                trh_rights[b] += [arg[b].index(trh[b][i].right)]

    return trh_actions, trh_lefts, trh_rights


class ProofTraceLMDataset(Dataset):
    def __init__(
            self,
            rollout_dir: str,
            sequence_length: int,
            tokenizer: ProofTraceTokenizer,
    ) -> None:
        self._sequence_length = sequence_length
        self._tokenizer = tokenizer

        self._rdirs = []

        assert os.path.isdir(rollout_dir)
        self._rdirs = [
            os.path.join(rollout_dir, f)
            for f in os.listdir(rollout_dir)
            if os.path.isdir(os.path.join(rollout_dir, f))
        ]

        Log.out(
            "Loaded extracted ProofTraces Rollout Dataset", {
                'cases': len(self._rdirs),
            })

    def __len__(
            self,
    ) -> int:
        return len(self._rdirs)

    def __getitem__(
            self,
            idx: int,
    ) -> typing.Tuple[
        int,
        typing.List[Action],
        typing.List[Action],
        Action,
    ]:
        rdir = self._rdirs[idx]

        rfiles = sorted([
            os.path.join(rdir, f)
            for f in os.listdir(rdir) if re.search(".rollout$", f)
        ], reverse=True)

        with gzip.open(rfiles[0], 'rb') as f:
            rollout = pickle.load(f)

        # `actions/arguemnts` are going from 0 to `ptra.len()-1` padded with
        # EXTRACT (removing final QED). `truth` is going from 1 to `ptra.len()`
        # (with PREPARE_TOKENS replaced by EXTRACT) and padded with EXTRACT.

        ptra = rollout.positive()
        assert ptra.action_len() > 0

        actions = ptra.actions()[:-1]
        arguments = ptra.arguments()[:-1]

        empty = ptra.actions()[1]
        assert empty.value == PREPARE_TOKENS['EMPTY']

        extract = Action.from_action('EXTRACT', empty, empty)

        truth = [extract] * (ptra.prepare_len()-1) + \
            ptra.actions()[ptra.prepare_len():]

        while len(actions) < self._sequence_length:
            actions.append(extract)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)
        while len(truth) < self._sequence_length:
            truth.append(extract)

        return (actions, arguments, truth)
