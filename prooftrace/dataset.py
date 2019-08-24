import gzip
import os
import pickle
import random
import re
import typing

from prooftrace.prooftrace import PREPARE_TOKENS, Action, ProofTraceTokenizer

from torch.utils.data import Dataset

from utils.log import Log


def lm_collate(
        batch
) -> typing.Tuple[
    typing.List[int],
    typing.List[typing.List[Action]],
    typing.List[typing.List[Action]],
    typing.List[Action],
]:
    indices = []
    actions = []
    arguments = []
    truths = []
    values = []

    for (idx, act, arg, trh, val) in batch:
        indices.append(idx)
        actions.append(act)
        arguments.append(arg)
        truths.append(trh)
        values.append(val)

    return (indices, actions, arguments, truths, values)


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

        # `actions/arguemnts` are going from 0 to index padded with EXTRACT.
        # `truth` is going from 1 to index+1 (with PREPARE_TOKENS replaced by
        # EMPTY) and padded with EXTRACT, index is therefore taken between
        # prepare_len() and ptra.len()-1 (removing the final QED from
        # `actions/arguments`)

        ptra = rollout.positive()
        index = random.randrange(
            ptra.prepare_len(),
            min(ptra.len()-1, self._sequence_length),
        )

        assert index <= self._sequence_length

        actions = ptra.actions()[:index]
        arguments = ptra.arguments()[:index]

        assert ptra.action_len() > 0
        assert index >= ptra.prepare_len()

        empty = ptra.actions()[1]
        assert empty.value == PREPARE_TOKENS['EMPTY']

        extract = Action.from_action('EXTRACT', empty, empty)

        truth = [extract] * (ptra.prepare_len()-1) + ptra.actions()[
            ptra.prepare_len():index+1
        ]

        while len(actions) < self._sequence_length:
            actions.append(extract)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)
        while len(truth) < self._sequence_length:
            truth.append(extract)

        value = float(index - ptra.prepare_len()) / ptra.action_len()

        return (index, actions, arguments, truth, value)
