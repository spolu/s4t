import argparse
import os
import time
import torch
import torch.distributed as distributed
import torch.optim as optim
import typing

from dataset.prooftrace import Action

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.lstm import H
from prooftrace.repl.env import Pool

from tensorboardX import SummaryWriter

from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Rollouts:
    def __init__(
            self,
            config,
    ):
        self._config = config
        self._device = torch.device(config.get('device'))

        self._rollout_size = config.get('prooftrace_ppo_rollout_size')
        self._pool_size = config.get('prooftrace_env_pool_size')

        self._sequence_length = config.get('prooftrace_sequence_length')
        self._hidden_size = config.get('prooftrace_hidden_size')

        self._gamma = config.get('prooftrace_ppo_gamma')
        self._tau = config.get('prooftrace_ppo_tau')
        self._batch_size = config.get('prooftrace_ppo_batch_size')

        self._epoch_count = config.get('prooftrace_ppo_epoch_count')
        self._clip = config.get('prooftrace_ppo_clip')

        self.observations = [
            [None] * self._pool_size for _ in range(self._rollout_size+1)
        ]
        self.embeds = torch.zeros(
            self._rollout_size+1, self._pool_size,
            self._sequence_length, self._hidden_size,
        ).to(self._device)

        self.actions = torch.zeros(
            self._rollout_size, self._pool_size, 3,
            dtype=torch.int64,
        ).to(self._device)

        self.log_probs = torch.zeros(
            self._rollout_size, self._pool_size, 3,
        ).to(self._device)

        self.rewards = torch.zeros(
            self._rollout_size, self._pool_size, 1
        ).to(self._device)
        self.values = torch.zeros(
            self._rollout_size+1, self._pool_size, 1
        ).to(self._device)

        self.masks = torch.ones(
            self._rollout_size+1, self._pool_size, 1,
        ).to(self._device)

        self.returns = torch.zeros(
            self._rollout_size+1, self._pool_size, 1,
        ).to(self._device)

    def insert(
            self,
            step: int,
            observations: typing.List[
                typing.Tuple[int, typing.List[Action]]
            ],
            embeds,
            actions,
            log_probs,
            values,
            rewards,
            masks,
    ):
        self.observations[step+1] = observations
        self.embeds[step+1] = embeds
        self.actions[step].copy_(actions)
        self.log_probs[step].copy_(log_probs)
        self.values[step].copy_(values)
        self.rewards[step].copy_(rewards)
        self.masks[step+1].copy_(masks)

    def compute_returns(
            self,
            next_values,
    ):
        self.values[-1].copy_(next_values)
        self.returns[-1].copy_(next_values)

        for step in reversed(range(self._rollout_size)):
            self.returns[step] = self.rewards[step] + \
                (self._gamma * self.returns[step+1] * self.masks[step+1])

    def after_update(
            self,
    ):
        self.observations[0] = self.observations[-1]
        self.masks[0].copy_(self.masks[-1])

    def generator(
            self,
            advantages,
    ):
        sampler = BatchSampler(
            SubsetRandomSampler(range(self._pool_size * self._rollout_size)),
            self._batch_size,
            drop_last=False,
        )
        for sample in sampler:
            indices = torch.LongTensor(sample).to(self._device)

            sz = self._pool_size
            yield \
                ([self.observations[:-1][i//sz][0][i % sz] for i in sample],
                 [self.observations[:-1][i//sz][1][i % sz] for i in sample]), \
                self.embeds[:-1].view(
                    -1, self.embeds.size(-2), self.embeds.size(-1),
                )[indices], \
                self.actions.view(-1, self.actions.size(-1))[indices], \
                self.returns[:-1].view(-1, 1)[indices], \
                self.masks[:-1].view(-1, 1)[indices], \
                self.log_probs.view(-1, 1)[indices], \
                advantages.view(-1, 1)[indices]


class PPO:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._rollout_size = config.get('prooftrace_ppo_rollout_size')
        self._pool_size = config.get('prooftrace_env_pool_size')
        self._epoch_count = config.get('prooftrace_ppo_epoch_count')
        self._clip = config.get('prooftrace_ppo_clip')

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model_E = E(self._config).to(self._device)
        self._inner_model_H = H(self._config).to(self._device)
        self._inner_model_PH = PH(self._config).to(self._device)
        self._inner_model_VH = VH(self._config).to(self._device)

        Log.out(
            "Initializing prooftrace PPO", {
                'parameter_count_E': self._inner_model_E.parameters_count(),
                'parameter_count_H': self._inner_model_H.parameters_count(),
                'parameter_count_PH': self._inner_model_PH.parameters_count(),
                'parameter_count_VH': self._inner_model_VH.parameters_count(),
            },
        )

        self._model_E = self._inner_model_E
        self._model_H = self._inner_model_H
        self._model_PH = self._inner_model_PH
        self._model_VH = self._inner_model_VH

        self._episode_stp_reward = [0.0] * self._pool_size
        self._episode_fnl_reward = [0.0] * self._pool_size

    def init_training(
            self,
    ):
        if self._config.get('distributed_training'):
            self._model_H = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_H,
                device_ids=[self._device],
            )
            self._model_PH = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_PH,
                device_ids=[self._device],
            )
            self._model_VH = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_VH,
                device_ids=[self._device],
            )

        self._optimizer = optim.Adam(
            [
                {'params': self._model_H.parameters()},
                {'params': self._model_PH.parameters()},
                {'params': self._model_VH.parameters()},
            ],
            lr=self._config.get('prooftrace_ppo_learning_rate'),
        )

        self._rollouts = Rollouts(self._config)

        self._pool = Pool(self._config, False)
        self._rollouts.observations[0] = self._pool.reset()
        with torch.no_grad():
            (idx, trc) = self._pool.reset()
            embeds = self._model_E(trc).detach()

            self._rollouts.observations[0] = (idx, trc)
            self._rollouts.embeds[0] = embeds

        Log.out('Training initialization', {
            "world_size": self._config.get('distributed_world_size'),
            "pool_size": self._config.get('prooftrace_env_pool_size'),
            "rollout_size": self._config.get('prooftrace_ppo_rollout_size'),
            "batch_size": self._config.get('prooftrace_ppo_batch_size'),
        })

    def init_testing(
            self,
            test_dataset,
    ):
        pass

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            Log.out(
                "Loading prooftrace", {
                    'load_dir': self._load_dir,
                })
            if os.path.isfile(self._load_dir + "/model_E_{}.pt".format(rank)):
                self._inner_model_E.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_E_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_H_{}.pt".format(rank)):
                self._inner_model_H.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_H_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_PH_{}.pt".format(rank)):
                self._inner_model_PH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_PH_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_VH_{}.pt".format(rank)):
                self._inner_model_VH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_VH_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )

            # At initial transfer of the pre-trained model we don't transfer
            # the optimizer.
            if training and os.path.isfile(
                    self._load_dir + "/optimizer_{}.pt".format(rank)
            ):
                self._optimizer.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/optimizer_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )

        return self

    def save(
            self,
    ):
        rank = self._config.get('distributed_rank')

        if self._save_dir:
            Log.out(
                "Saving prooftrace models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_model_E.state_dict(),
                self._save_dir + "/model_E_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_H.state_dict(),
                self._save_dir + "/model_H_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_PH.state_dict(),
                self._save_dir + "/model_PH_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_VH.state_dict(),
                self._save_dir + "/model_VH_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer_{}.pt".format(rank),
            )

    def batch_train(
            self,
            epoch,
    ):
        self._model_E.eval()
        self._model_H.train()
        self._model_PH.train()
        self._model_VH.train()

        stp_reward_meter = Meter()
        fnl_reward_meter = Meter()
        act_loss_meter = Meter()
        val_loss_meter = Meter()
        # ent_loss_meter = Meter()

        batch_start = time.time()

        for step in range(self._rollout_size):
            with torch.no_grad():
                (idx, trc) = self._rollouts.observations[step]
                embeds = self._model_E(trc).detach()

                hiddens = self._model_H(embeds)

                head = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    hiddens[i][0].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)

                prd_actions, prd_lefts, prd_rights = \
                    self._model_PH(head, targets)

                values = \
                    self._model_VH(head, targets)

                actions = torch.cat((
                    Categorical(
                        torch.exp(prd_actions)
                    ).sample().unsqueeze(1),
                    Categorical(
                        torch.exp(prd_lefts)
                    ).sample().unsqueeze(1),
                    Categorical(
                        torch.exp(prd_rights)
                    ).sample().unsqueeze(1),
                ), dim=1)

                log_probs = torch.cat((
                    prd_actions.gather(1, actions[:, 0].unsqueeze(1)),
                    prd_lefts.gather(1, actions[:, 1].unsqueeze(1)),
                    prd_rights.gather(1, actions[:, 2].unsqueeze(1)),
                ), dim=1)

            # self._rollouts.observations[0] = self._pool.reset()
            observations, rewards, dones = self._pool.step(
                [tuple(a) for a in actions.detach().cpu().numpy()]
            )
            for i, r in enumerate(rewards):
                self._episode_stp_reward[i] += r[0]
                self._episode_fnl_reward[i] += r[1]
                if dones[i]:
                    stp_reward_meter.update(self._episode_stp_reward[i])
                    fnl_reward_meter.update(self._episode_fnl_reward[i])
                    self._episode_stp_reward[i] = 0.0
                    self._episode_fnl_reward[i] = 0.0

            self._rollouts.insert(
                step,
                observations,
                embeds,
                actions.detach(),
                log_probs.detach(),
                values.detach(),
                torch.tensor(
                    [(r[0] + r[1]) for r in rewards], dtype=torch.int64,
                ).unsqueeze(1).to(self._device),
                torch.tensor(
                    [[0.0] if d else [1.0] for d in dones],
                ).to(self._device),
            )

        with torch.no_grad():
            (idx, trc) = self._rollouts.observations[-1]
            embeds = self._rollouts.embeds[-1]

            hiddens = self._model_H(embeds)

            head = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0)
                for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                hiddens[i][0].unsqueeze(0)
                for i in range(len(idx))
            ], dim=0)

            values = \
                self._model_VH(head, targets)

            self._rollouts.compute_returns(values.detach())

            advantages = \
                self._rollouts.returns[:-1] - self._rollouts.values[:-1]
            advantages = \
                (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(self._epoch_count):
            generator = self._rollouts.generator(advantages)

            for batch in generator:
                rollout_observations, \
                    rollout_embeds, \
                    rollout_actions, \
                    rollout_returns, \
                    rollout_masks, \
                    rollout_log_probs, \
                    rollout_advantages = batch

                (idx, trc) = rollout_observations

                hiddens = self._model_H(rollout_embeds)

                head = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    hiddens[i][0].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)

                prd_actions, prd_lefts, prd_rights = \
                    self._model_PH(head, targets)

                values = \
                    self._model_VH(head, targets)

                log_probs = torch.cat((
                    prd_actions.gather(1, rollout_actions[:, 0].unsqueeze(1)),
                    prd_lefts.gather(1, rollout_actions[:, 1].unsqueeze(1)),
                    prd_rights.gather(1, rollout_actions[:, 2].unsqueeze(1)),
                ), dim=1)

                ratio = torch.exp(log_probs - rollout_log_probs)

                action_loss = -torch.min(
                    ratio * rollout_advantages,
                    torch.clamp(ratio, 1.0 - self._clip, 1.0 + self._clip) *
                    rollout_advantages,
                ).mean()

                value_loss = (rollout_returns.detach() - values).pow(2).mean()

                # entropy_loss = -entropy.mean()
                # entropy_loss * self.entropy_loss_coeff).backward()

                self._optimizer.zero_grad()

                (value_loss + action_loss).backward()

                # if self.grad_norm_max > 0.0:
                #     nn.utils.clip_grad_norm(
                #         self.policy.parameters(), self.grad_norm_max,
                #     )

                self._optimizer.step()

                act_loss_meter.update(action_loss.item())
                val_loss_meter.update(value_loss.item())

        self._rollouts.after_update()

        Log.out("PROOFTRACE PPO TRAIN", {
            'epoch': epoch,
            'fps': "{:.2f}".format(
                self._pool_size * self._rollout_size /
                (time.time() - batch_start)
            ),
            'stp_reward_avg': "{:.4f}".format(stp_reward_meter.avg or 0.0),
            'fnl_reward_avg': "{:.4f}".format(fnl_reward_meter.avg or 0.0),
            'act_loss_avg': "{:.4f}".format(act_loss_meter.avg),
            'val_loss_avg': "{:.4f}".format(val_loss_meter.avg),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "train/prooftrace/ppo/act_loss",
                act_loss_meter.avg, epoch,
            )
            self._tb_writer.add_scalar(
                "train/prooftrace/ppo/val_loss",
                val_loss_meter.avg, epoch,
            )
            self._tb_writer.add_scalar(
                "train/prooftrace/ppo/stp_reward",
                stp_reward_meter.avg or 0.0, epoch,
            )
            self._tb_writer.add_scalar(
                "train/prooftrace/ppo/fnl_reward",
                fnl_reward_meter.avg or 0.0, epoch,
            )

    def batch_test(
            self,
    ):
        self._model_E.eval()
        self._model_H.eval()
        self._model_PH.eval()
        self._model_VH.eval()

        with torch.no_grad():
            pass


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_size',
        type=str, help="config override",
    )
    parser.add_argument(
        '--save_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )
    parser.add_argument(
        '--tensorboard_log_dir',
        type=str, help="config override",
    )

    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    parser.add_argument(
        '--distributed_training',
        type=str2bool, help="confg override",
    )
    parser.add_argument(
        '--distributed_world_size',
        type=int, help="config override",
    )
    parser.add_argument(
        '--distributed_rank',
        type=int, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

    if args.distributed_training is not None:
        config.override('distributed_training', args.distributed_training)
    if args.distributed_rank is not None:
        config.override('distributed_rank', args.distributed_rank)
    if args.distributed_world_size is not None:
        config.override('distributed_world_size', args.distributed_world_size)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )
    if args.save_dir is not None:
        config.override(
            'prooftrace_save_dir',
            os.path.expanduser(args.save_dir),
        )

    if config.get('distributed_training'):
        distributed.init_process_group(
            backend=config.get('distributed_backend'),
            init_method=config.get('distributed_init_method'),
            rank=config.get('distributed_rank'),
            world_size=config.get('distributed_world_size'),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    ppo = PPO(config)

    ppo.init_training()
    ppo.load(True)

    epoch = 0
    while True:
        ppo.batch_train(epoch)
        epoch += 1


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
    parser.add_argument(
        '--load_dir',
        type=str, help="config override",
    )

    parser.add_argument(
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    ppo = PPO(config)

    ppo.init_testing()
    ppo.load(False)

    epoch = 0
    while True:
        ppo.batch_test(epoch)
        epoch += 1