import datetime
import gzip
import os
import pickle
import random
import re
import typing

from prooftrace.repl.repl import REPL
from prooftrace.prooftrace import INV_PREPARE_TOKENS, Action, \
    ProofTraceActions, ProofTraceTokenizer
from prooftrace.search.random import RandomSampler

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

    for (idx, act, arg, trh) in batch:
        indices.append(idx)
        actions.append(act)
        arguments.append(arg)
        truths.append(trh)

    return (indices, actions, arguments, truths)


class ProofTraceLMDataset(Dataset):
    def __init__(
            self,
            rollout_dir: str,
            sequence_length: int,
            tokenizer: ProofTraceTokenizer,
            augment: str = 'none',
            period: int = 5,
    ) -> None:
        self._sequence_length = sequence_length
        self._tokenizer = tokenizer
        self._augment = augment
        self._period = period

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

    def random_augment(
            self,
            ground: ProofTraceActions,
            index: int,
    ) -> ProofTraceActions:
        """ Augments through random sampling the passed ProofTraceActions

        Warning: the operation is destructive for the passed ProofTraceAction.
        """
        ptra = ProofTraceActions(
            'AUGMENT-{}-{}'.format(
                datetime.datetime.now().strftime("%Y%m%d_%H%M_%S.%f"),
                random.randint(0, 9999),
            ),
            [
                ground.actions()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
            [
                ground.arguments()[i] for i in range(ground.len())
                if ground.actions()[i].value in INV_PREPARE_TOKENS
            ],
        )
        repl = REPL(self._tokenizer)
        repl.prepare(ptra)

        # repl.replay(ground)
        # Log.out('DONE')
        # return ground

        sampler = RandomSampler(ptra)

        next_idx = ptra.len()

        while next_idx <= index:
            action = None
            sampled = False

            skip = False
            if (index + 1 - next_idx + ptra.len()) >= self._sequence_length:
                skip = True
            if (index == next_idx):
                skip = True

            if not skip and random.random() < (1.0 / self._period):
                action = sampler.sample(ptra, repl, 8)
                # Log.out('AUGMENT SAMPLE')

            if action is None:
                action = ground.actions()[next_idx]
                # Log.out('AUGMENT READ', {
                #     'next_idx': next_idx,
                #     'action_index': action._index,
                #     'action_hash': action.hash(),
                #     'action_id': id(action),
                # })
            else:
                sampled = True

            thm = repl.apply(action)

            if sampled:
                argument = ptra.build_argument(
                    thm.concl(), thm.hyp(), thm.index(),
                )
            else:
                argument = ground.arguments()[next_idx]
                argument._index = thm.index()
                next_idx += 1

            # Log.out('AUGMENT APPLY', {
            #     'action_index': action._index,
            #     'action_hash': action.hash(),
            #     'action_id': id(action),
            # })

            ptra.append(action, argument)

        return ptra, ptra.len()-1

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

        ptra = rollout.positive()
        index = random.randrange(
            ptra.prepare_len(),
            min(ptra.len(), self._sequence_length),
        )

        if self._augment == 'random':
            ptra, index = self.random_augment(ptra, index)

        assert index <= self._sequence_length

        truth = ptra.actions()[index]
        actions = ptra.actions()[:index]
        arguments = ptra.arguments()[:index]

        actions.append(Action.from_action('EXTRACT', None, None))

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._sequence_length:
            actions.append(empty)
        while len(arguments) < self._sequence_length:
            arguments.append(empty)

        return (index, actions, arguments, truth)
