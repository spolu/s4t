import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import typing

from dataset.prooftrace import Action

from generic.iota import IOTAAck, IOTASyn

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.transformer import H
from prooftrace.repl.env import Pool

from tensorboardX import SummaryWriter

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from utils.config import Config
from utils.meter import Meter
from utils.log import Log


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
            actions,
            log_probs,
            values,
            rewards,
            masks,
    ):
        self.observations[step+1] = observations
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

        # for step in reversed(range(self._rollout_size)):
        #     self.returns[step] = self.rewards[step] + \
        #         (self._gamma * self.returns[step+1] * self.masks[step+1])

        gae = 0
        for step in reversed(range(self._rollout_size)):
            delta = (
                self.rewards[step] +
                self._gamma * self.values[step+1] * self.masks[step+1] -
                self.values[step]
            )
            gae = delta + self._gamma * self._tau * self.masks[step+1] * gae
            self.returns[step] = gae + self.values[step]

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
                self.actions.view(-1, self.actions.size(-1))[indices], \
                self.values[:-1].view(-1, 1)[indices], \
                self.returns[:-1].view(-1, 1)[indices], \
                self.masks[:-1].view(-1, 1)[indices], \
                self.log_probs.view(-1, self.log_probs.size(-1))[indices], \
                advantages.view(-1, 1)[indices]


class ACK:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._rollout_size = config.get('prooftrace_ppo_rollout_size')
        self._pool_size = config.get('prooftrace_env_pool_size')
        self._epoch_count = config.get('prooftrace_ppo_epoch_count')
        self._clip = config.get('prooftrace_ppo_clip')
        self._grad_norm_max = config.get('prooftrace_ppo_grad_norm_max')
        self._entropy_coeff = config.get('prooftrace_ppo_entropy_coeff')
        self._value_coeff = config.get('prooftrace_ppo_value_coeff')
        self._learning_rate = config.get('prooftrace_ppo_learning_rate')

        self._reset_gamma = \
            config.get('prooftrace_ppo_reset_gamma')
        self._explore_alpha = \
            config.get('prooftrace_ppo_explore_alpha')
        self._explore_beta = \
            config.get('prooftrace_ppo_explore_beta')
        self._explore_beta_width = \
            config.get('prooftrace_ppo_explore_beta_width')
        self._step_reward_prob = \
            config.get('prooftrace_ppo_step_reward_prob')
        self._match_reward_prob = \
            config.get('prooftrace_ppo_match_reward_prob')

        self._device = torch.device(config.get('device'))

        self._modules = {
            'E': E(self._config).to(self._device),
            'H': H(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        self._ack = IOTAAck(
            config.get('prooftrace_ppo_iota_sync_dir'),
            self._modules,
        )

        self._rollouts = Rollouts(self._config)

        self._pool = Pool(self._config, False)
        self._rollouts.observations[0] = self._pool.reset(self._reset_gamma)

        self._episode_stp_reward = [0.0] * self._pool_size
        self._episode_mtc_reward = [0.0] * self._pool_size
        self._episode_fnl_reward = [0.0] * self._pool_size

        Log.out('ACK initialization', {
            "pool_size": self._config.get('prooftrace_env_pool_size'),
            "rollout_size": self._config.get('prooftrace_ppo_rollout_size'),
            "batch_size": self._config.get('prooftrace_ppo_batch_size'),
        })

    def update(
            self,
            config: Config,
    ) -> None:
        self._config = config

        coeff = self._config.get('prooftrace_ppo_entropy_coeff')
        if coeff != self._entropy_coeff:
            self._entropy_coeff = coeff
            Log.out("Updated", {
                "prooftrace_ppo_entropy_coeff": coeff,
            })

        coeff = self._config.get('prooftrace_ppo_value_coeff')
        if coeff != self._value_coeff:
            self._value_coeff = coeff
            Log.out("Updated", {
                "prooftrace_ppo_value_coeff": coeff,
            })

        gamma = self._config.get('prooftrace_ppo_reset_gamma')
        if gamma != self._reset_gamma:
            self._reset_gamma = gamma
            Log.out("Updated", {
                "prooftrace_ppo_reset_gamma": gamma,
            })

        alpha = self._config.get('prooftrace_ppo_explore_alpha')
        if alpha != self._explore_alpha:
            self._explore_alpha = alpha
            Log.out("Updated", {
                "prooftrace_ppo_explore_alpha": alpha,
            })

        beta = self._config.get('prooftrace_ppo_explore_beta')
        if beta != self._explore_beta:
            self._explore_beta = beta
            Log.out("Updated", {
                "prooftrace_ppo_explore_beta": beta,
            })

        width = self._config.get('prooftrace_ppo_explore_beta_width')
        if width != self._explore_beta_width:
            self._explore_beta_width = width
            Log.out("Updated", {
                "prooftrace_ppo_explore_beta_width": width,
            })

        prob = self._config.get('prooftrace_ppo_step_reward_prob')
        if prob != self._step_reward_prob:
            self._step_reward_prob = prob
            Log.out("Updated", {
                "prooftrace_ppo_step_reward_prob": prob,
            })

        prob = self._config.get('prooftrace_ppo_match_reward_prob')
        if prob != self._match_reward_prob:
            self._match_reward_prob = prob
            Log.out("Updated", {
                "prooftrace_ppo_match_reward_prob": prob,
            })

    def run_once(
            self,
            epoch,
    ):
        for m in self._modules:
            self._modules[m].train()

        info = self._ack.fetch(self._device)
        if info is not None:
            self.update(info['config'])

        stp_reward_meter = Meter()
        mtc_reward_meter = Meter()
        fnl_reward_meter = Meter()
        act_loss_meter = Meter()
        val_loss_meter = Meter()
        entropy_meter = Meter()
        match_count_meter = Meter()
        demo_length_meter = Meter()

        frame_count = 0

        for step in range(self._rollout_size):
            with torch.no_grad():
                (idx, act) = self._rollouts.observations[step]

                embeds = self._modules['E'](act).detach()
                hiddens = self._modules['H'](embeds)

                heads = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    embeds[i][0].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)

                prd_actions, prd_lefts, prd_rights = \
                    self._modules['PH'](heads, targets)

                values = \
                    self._modules['VH'](heads, targets)

                actions, count = self._pool.explore(
                    prd_actions,
                    prd_lefts,
                    prd_rights,
                    self._explore_alpha,
                    self._explore_beta,
                    self._explore_beta_width,
                )
                frame_count += count

                observations, rewards, dones, infos = self._pool.step(
                    [tuple(a) for a in actions.detach().cpu().numpy()],
                    self._step_reward_prob,
                    self._match_reward_prob,
                    self._reset_gamma,
                )
                frame_count += actions.size(0)
                for i, info in enumerate(infos):
                    if 'match_count' in info:
                        assert dones[i]
                        match_count_meter.update(info['match_count'])
                    if 'demo_length' in info:
                        assert dones[i]
                        demo_length_meter.update(info['demo_length'])

                log_probs = torch.cat((
                    prd_actions.gather(1, actions[:, 0].unsqueeze(1)),
                    prd_lefts.gather(1, actions[:, 1].unsqueeze(1)),
                    prd_rights.gather(1, actions[:, 2].unsqueeze(1)),
                ), dim=1)

            for i, r in enumerate(rewards):
                self._episode_stp_reward[i] += r[0]
                self._episode_mtc_reward[i] += r[1]
                self._episode_fnl_reward[i] += r[2]
                if dones[i]:
                    stp_reward_meter.update(self._episode_stp_reward[i])
                    mtc_reward_meter.update(self._episode_mtc_reward[i])
                    fnl_reward_meter.update(self._episode_fnl_reward[i])
                    self._episode_stp_reward[i] = 0.0
                    self._episode_mtc_reward[i] = 0.0
                    self._episode_fnl_reward[i] = 0.0

            self._rollouts.insert(
                step,
                observations,
                actions.detach(),
                log_probs.detach(),
                values.detach(),
                torch.tensor(
                    [(r[0] + r[1] + r[2]) for r in rewards], dtype=torch.int64,
                ).unsqueeze(1).to(self._device),
                torch.tensor(
                    [[0.0] if d else [1.0] for d in dones],
                ).to(self._device),
            )

        with torch.no_grad():
            (idx, act) = self._rollouts.observations[-1]

            embeds = self._modules['E'](act)
            hiddens = self._modules['H'](embeds)

            heads = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0)
                for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                embeds[i][0].unsqueeze(0)
                for i in range(len(idx))
            ], dim=0)

            values = \
                self._modules['VH'](heads, targets)

            self._rollouts.compute_returns(values.detach())

            advantages = \
                self._rollouts.returns[:-1] - self._rollouts.values[:-1]
            advantages = \
                (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        ignored = False
        for e in range(self._epoch_count):
            if ignored:
                continue

            generator = self._rollouts.generator(advantages)

            for batch in generator:
                if ignored:
                    continue

                rollout_observations, \
                    rollout_actions, \
                    rollout_values, \
                    rollout_returns, \
                    rollout_masks, \
                    rollout_log_probs, \
                    rollout_advantages = batch

                (idx, act) = rollout_observations

                embeds = self._modules['E'](act)
                hiddens = self._modules['H'](embeds)

                heads = torch.cat([
                    hiddens[i][idx[i]].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)
                targets = torch.cat([
                    embeds[i][0].unsqueeze(0)
                    for i in range(len(idx))
                ], dim=0)

                prd_actions, prd_lefts, prd_rights = \
                    self._modules['PH'](heads, targets)

                values = \
                    self._modules['VH'](heads, targets)

                log_probs = torch.cat((
                    prd_actions.gather(1, rollout_actions[:, 0].unsqueeze(1)),
                    prd_lefts.gather(1, rollout_actions[:, 1].unsqueeze(1)),
                    prd_rights.gather(1, rollout_actions[:, 2].unsqueeze(1)),
                ), dim=1)
                entropy = -(log_probs * torch.exp(log_probs)).mean()

                # Clipped action loss.
                ratio = torch.exp(log_probs - rollout_log_probs)
                action_loss = -torch.min(
                    ratio * rollout_advantages,
                    torch.clamp(ratio, 1.0 - self._clip, 1.0 + self._clip) *
                    rollout_advantages,
                ).mean()

                # Clipped value loss.
                clipped_values = rollout_values + \
                    (values - rollout_values).clamp(-self._clip, self._clip)
                value_loss = torch.max(
                    F.mse_loss(values, rollout_returns),
                    F.mse_loss(clipped_values, rollout_returns),
                )
                # value_loss = F.mse_loss(values, rollout_returns)

                # Log.out("RATIO/ADV/LOSS", {
                #     'clipped_ratio': torch.clamp(
                #         ratio, 1.0 - self._clip, 1.0 + self._clip
                #     ).mean().item(),
                #     'ratio': ratio.mean().item(),
                #     'advantages': rollout_advantages.mean().item(),
                #     'action_loss': action_loss.item(),
                # })

                if abs(action_loss.item()) > 10e2 or \
                        abs(value_loss.item()) > 10e2 or \
                        math.isnan(value_loss.item()) or \
                        math.isnan(entropy.item()):
                    Log.out("IGNORING", {
                        'epoch': epoch,
                        'act_loss': "{:.4f}".format(action_loss.item()),
                        'val_loss': "{:.4f}".format(value_loss.item()),
                        'entropy': "{:.4f}".format(entropy.item()),
                    })
                    ignored = True
                else:
                    # Backward pass.
                    for m in self._modules:
                        self._modules[m].zero_grad()

                    (action_loss +
                     self._value_coeff * value_loss -
                     self._entropy_coeff * entropy).backward()

                    if self._grad_norm_max > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self._modules['VH'].parameters(),
                            self._grad_norm_max,
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self._modules['PH'].parameters(),
                            self._grad_norm_max,
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self._modules['H'].parameters(),
                            self._grad_norm_max,
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self._modules['E'].parameters(),
                            self._grad_norm_max,
                        )

                    act_loss_meter.update(action_loss.item())
                    val_loss_meter.update(value_loss.item())
                    entropy_meter.update(entropy.item())

                    self._ack.push({
                        'frame_count': frame_count,
                        'match_count': (match_count_meter.avg or 0.0),
                        'demo_length': (demo_length_meter.max or 0.0),
                        'stp_reward': (stp_reward_meter.avg or 0.0),
                        'mtc_reward': (mtc_reward_meter.avg or 0.0),
                        'fnl_reward': (fnl_reward_meter.avg or 0.0),
                        'act_loss': act_loss_meter.avg,
                        'val_loss': val_loss_meter.avg,
                        'entropy': entropy_meter.avg,
                    })
                    if frame_count > 0:
                        frame_count = 0

                info = self._ack.fetch(self._device)
                if info is not None:
                    self.update(info['config'])

        self._rollouts.after_update()

        Log.out("PROOFTRACE PPO ACK RUN", {
            'epoch': epoch,
            'ignored': ignored,
            'match_count': "{:.2f}".format(match_count_meter.avg or 0.0),
            'demo_length': "{:.0f}".format(demo_length_meter.max or 0.0),
            'stp_reward': "{:.4f}".format(stp_reward_meter.avg or 0.0),
            'mtc_reward': "{:.4f}".format(mtc_reward_meter.avg or 0.0),
            'fnl_reward': "{:.4f}".format(fnl_reward_meter.avg or 0.0),
            'act_loss': "{:.4f}".format(act_loss_meter.avg or 0.0),
            'val_loss': "{:.4f}".format(val_loss_meter.avg or 0.0),
            'entropy': "{:.4f}".format(entropy_meter.avg or 0.0),
        })


class SYN:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._learning_rate = config.get('prooftrace_ppo_learning_rate')
        self._min_update_count = \
            config.get('prooftrace_ppo_iota_min_update_count')
        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._epoch = 0
        self._last_update = None

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            self._tb_writer = SummaryWriter(
                self._config.get('tensorboard_log_dir'),
            )

        self._modules = {
            'E': E(self._config).to(self._device),
            'H': H(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

        Log.out(
            "SYN Initializing", {
                'parameter_count_E': self._modules['E'].parameters_count(),
                'parameter_count_H': self._modules['H'].parameters_count(),
                'parameter_count_PH': self._modules['PH'].parameters_count(),
                'parameter_count_VH': self._modules['VH'].parameters_count(),
            },
        )

        self._syn = IOTASyn(
            config.get('prooftrace_ppo_iota_sync_dir'),
            self._modules,
        )

        self._optimizer = optim.Adam(
            [
                {'params': self._modules['E'].parameters()},
                {'params': self._modules['H'].parameters()},
                {'params': self._modules['PH'].parameters()},
                {'params': self._modules['VH'].parameters()},
            ],
            lr=self._learning_rate,
        )

    def load(
            self,
            training=True,
    ):
        if self._load_dir:
            Log.out(
                "Loading prooftrace", {
                    'load_dir': self._load_dir,
                })
            if os.path.isfile(self._load_dir + "/model_E.pt"):
                Log.out('Loading E')
                self._modules['E'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_H.pt"):
                Log.out('Loading H')
                self._modules['H'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_H.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_PH.pt"):
                Log.out('Loading PH')
                self._modules['PH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_PH.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_VH.pt"):
                Log.out('Loading VH')
                self._modules['VH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_VH.pt",
                        map_location=self._device,
                    ),
                )

            if training and os.path.isfile(self._load_dir + "/optimizer.pt"):
                Log.out('Loading Optimizer')
                self._optimizer.load_state_dict(
                    torch.load(
                        self._load_dir + "/optimizer.pt",
                        map_location=self._device,
                    ),
                )

        return self

    def save(
            self,
    ):
        if self._save_dir:
            Log.out(
                "Saving prooftrace models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._modules['E'].state_dict(),
                self._save_dir + "/model_E.pt",
            )
            torch.save(
                self._modules['H'].state_dict(),
                self._save_dir + "/model_H.pt",
            )
            torch.save(
                self._modules['PH'].state_dict(),
                self._save_dir + "/model_PH.pt",
            )
            torch.save(
                self._modules['VH'].state_dict(),
                self._save_dir + "/model_VH.pt",
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer.pt",
            )

    def update(
            self,
    ) -> None:
        update = self._config.update()
        if update:
            if 'prooftrace_ppo_learning_rate' in update:
                lr = self._config.get('prooftrace_ppo_learning_rate')
                if lr != self._learning_rate:
                    self._learning_rate = lr
                    for group in self._optimizer.param_groups:
                        group['lr'] = lr
                    Log.out("Updated", {
                        "prooftrace_ppo_learning_rate": lr,
                    })
            if 'prooftrace_ppo_iota_min_update_count' in update:
                cnt = self._config.get('prooftrace_ppo_iota_min_update_count')
                if cnt != self._min_update_count:
                    self._min_update_count = cnt
                    Log.out("Updated", {
                        "prooftrace_ppo_iota_min_update_count": cnt,
                    })

            if self._tb_writer is not None:
                for k in update:
                    if k in [
                            'prooftrace_ppo_learning_rate',
                            'prooftrace_ppo_iota_min_update_count',
                            'prooftrace_ppo_entropy_coeff',
                            'prooftrace_ppo_value_coeff',
                            'prooftrace_ppo_explore_alpha',
                            'prooftrace_ppo_explore_beta',
                            'prooftrace_ppo_explore_beta_width',
                            'prooftrace_ppo_step_reward_prob',
                            'prooftrace_ppo_match_reward_prob',
                    ]:
                        self._tb_writer.add_scalar(
                            "prooftrace_ppo_train_run/{}".format(k),
                            update[k], self._epoch,
                        )

    def run_once(
            self,
    ):
        for m in self._modules:
            self._modules[m].train()

        run_start = time.time()

        if self._epoch == 0:
            self._syn.broadcast({'config': self._config})

        self._optimizer.zero_grad()
        infos = self._syn.aggregate(self._device, self._min_update_count)

        if len(infos) == 0:
            if self._epoch == 0:
                self._epoch += 1
            time.sleep(1)
            return

        self._optimizer.step()
        self._syn.broadcast({'config': self._config})

        if self._last_update is not None:
            update_delta = time.time() - self._last_update
        else:
            update_delta = 0.0
        self._last_update = time.time()

        frame_count_meter = Meter()
        match_count_meter = Meter()
        demo_length_meter = Meter()
        stp_reward_meter = Meter()
        mtc_reward_meter = Meter()
        fnl_reward_meter = Meter()
        tot_reward_meter = Meter()
        act_loss_meter = Meter()
        val_loss_meter = Meter()
        entropy_meter = Meter()

        for info in infos:
            frame_count_meter.update(info['frame_count'])
            match_count_meter.update(info['match_count'])
            demo_length_meter.update(info['demo_length'])
            stp_reward_meter.update(info['stp_reward'])
            mtc_reward_meter.update(info['mtc_reward'])
            fnl_reward_meter.update(info['fnl_reward'])
            tot_reward_meter.update(
                info['stp_reward'] +
                info['mtc_reward'] +
                info['fnl_reward']
            )
            act_loss_meter.update(info['act_loss'])
            val_loss_meter.update(info['val_loss'])
            entropy_meter.update(info['entropy'])

        Log.out("PROOFTRACE PPO SYN RUN", {
            'epoch': self._epoch,
            'run_time': "{:.2f}".format(time.time() - run_start),
            'update_count': len(infos),
            'frame_count': frame_count_meter.sum,
            'update_delta': "{:.2f}".format(update_delta),
            'match_count': "{:.2f}".format(match_count_meter.avg or 0.0),
            'demo_length': "{:.0f}".format(demo_length_meter.max or 0.0),
            'stp_reward': "{:.4f}".format(stp_reward_meter.avg or 0.0),
            'mtc_reward': "{:.4f}".format(mtc_reward_meter.avg or 0.0),
            'fnl_reward': "{:.4f}".format(fnl_reward_meter.avg or 0.0),
            'tot_reward': "{:.4f}".format(tot_reward_meter.avg or 0.0),
            'act_loss': "{:.4f}".format(act_loss_meter.avg or 0.0),
            'val_loss': "{:.4f}".format(val_loss_meter.avg or 0.0),
            'entropy': "{:.4f}".format(entropy_meter.avg or 0.0),
        })

        if self._tb_writer is not None:
            if len(infos) > 0:
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/update_delta",
                    update_delta, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/match_count",
                    match_count_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/demo_length",
                    demo_length_meter.max, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/act_loss",
                    act_loss_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/val_loss",
                    val_loss_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/entropy",
                    entropy_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/stp_reward",
                    stp_reward_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/mtc_reward",
                    mtc_reward_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/fnl_reward",
                    fnl_reward_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/tot_reward",
                    tot_reward_meter.avg, self._epoch,
                )
                self._tb_writer.add_scalar(
                    "prooftrace_ppo_train/frame_count",
                    frame_count_meter.sum, self._epoch,
                )

        self._epoch += 1

        if self._epoch % 10 == 0:
            self.save()


def ack_run():
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
        '--device',
        type=str, help="config override",
    )
    parser.add_argument(
        '--sync_dir',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
    if args.sync_dir is not None:
        config.override(
            'prooftrace_ppo_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
        )

    if args.dataset_size is not None:
        config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    ack = ACK(config)

    epoch = 0
    while True:
        ack.run_once(epoch)
        epoch += 1


def syn_run():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
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
        '--sync_dir',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
    if args.sync_dir is not None:
        config.override(
            'prooftrace_ppo_iota_sync_dir',
            os.path.expanduser(args.sync_dir),
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

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    syn = SYN(config).load(True)

    while True:
        syn.update()
        syn.run_once()
