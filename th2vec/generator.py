import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim

from dataset.holstep import HolStepKernel, HolStepSet
from dataset.holstep import HolStepTermDataset

from tensorboardX import SummaryWriter

from th2vec.models.cnn import G, D

from torch.distributions.categorical import Categorical

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Th2VecGenerator:
    def __init__(
            self,
            config: Config,
            kernel: HolStepKernel,
    ):
        self._config = config
        self._kernel = kernel

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('th2vec_save_dir')
        self._load_dir = config.get('th2vec_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model_G = G(self._config).to(self._device)
        self._inner_model_D = D(self._config).to(self._device)

        Log.out(
            "Initializing th2vec", {
                'G_parameter_count': self._inner_model_G.parameters_count(),
                'D_parameter_count': self._inner_model_D.parameters_count(),
            },
        )

        self._model_G = self._inner_model_G
        self._model_D = self._inner_model_D
        self._loss = nn.NLLLoss()

    def init_training(
            self,
            train_dataset,
    ):
        if self._config.get('distributed_training'):
            self._model_G = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_G,
                device_ids=[self._device],
            )
            self._model_D = torch.nn.parallel.DistributedDataParallel(
                self._inner_model_D,
                device_ids=[self._device],
            )

        self._optimizer_G = optim.Adam(
            self._model_G.parameters(),
            lr=self._config.get('th2vec_learning_rate'),
        )
        self._optimizer_D = optim.Adam(
            self._model_D.parameters(),
            lr=self._config.get('th2vec_learning_rate'),
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        pin_memory = False
        if self._config.get('device') != 'cpu':
            pin_memory = True

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config.get('th2vec_batch_size'),
            shuffle=(self._train_sampler is None),
            pin_memory=pin_memory,
            num_workers=8,
            sampler=self._train_sampler,
        )

        self._train_batch = 0

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_{}.pt".format(rank)
            ):
                Log.out(
                    "Loading th2vec models", {
                        'save_dir': self._load_dir,
                    })
                self._inner_model_G.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_G_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                self._inner_model_D.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_D_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                if training:
                    self._optimizer_G.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/optimizer_G_{}.pt".format(rank),
                            map_location=self._device,
                        ),
                    )
                    self._optimizer_D.load_state_dict(
                        torch.load(
                            self._load_dir +
                            "/optimizer_D_{}.pt".format(rank),
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
                "Saving th2vec models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_model_G.state_dict(),
                self._save_dir + "/model_G_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer_G.state_dict(),
                self._save_dir + "/optimizer_G_{}.pt".format(rank),
            )
            torch.save(
                self._inner_model_D.state_dict(),
                self._save_dir + "/model_D_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer_D.state_dict(),
                self._save_dir + "/optimizer_D_{}.pt".format(rank),
            )

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model_G.train()
        self._model_D.train()

        dis_loss_meter = Meter()
        gen_loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)

        for it, trm in enumerate(self._train_loader):
            nse = torch.randn(
                trm.size(0),
                self._config.get('th2vec_transformer_hidden_size'),
            ).to(self._device)

            trm_rel = trm.to(self._device)
            trm_gen = self._model_G(nse)

            m = Categorical(torch.exp(trm_gen))
            trm_smp = m.sample()

            onh_rel = self._inner_model_D.one_hot(trm_rel)
            onh_gen = self._inner_model_D.one_hot(trm_smp)
            onh_gen.requires_grad = True

            dis_rel = self._model_D(onh_rel)
            dis_gen = self._model_D(onh_gen)

            # Label smoothing
            ones = torch.ones(*dis_rel.size()) - \
                torch.rand(*dis_rel.size()) / 10
            zeros = torch.zeros(*dis_rel.size()) + \
                torch.rand(*dis_rel.size()) / 10

            dis_loss_rel = \
                F.binary_cross_entropy(dis_rel, ones.to(self._device))
            dis_loss_gen = \
                F.binary_cross_entropy(dis_gen, zeros.to(self._device))
            dis_loss = dis_loss_rel + dis_loss_gen

            self._optimizer_D.zero_grad()
            dis_loss.backward()
            self._optimizer_D.step()

            # REINFORCE
            reward = onh_gen.grad.detach()
            reward -= \
                torch.mean(reward, 2).unsqueeze(-1).expand(*reward.size())
            reward /= \
                torch.std(reward, 2).unsqueeze(-1).expand(*reward.size())

            gen_loss = trm_gen * reward  # (-logp * -grad_dis)
            gen_loss = gen_loss.mean()

            self._optimizer_G.zero_grad()
            gen_loss.backward()
            self._optimizer_G.step()

            dis_loss_meter.update(dis_loss.item())
            gen_loss_meter.update(gen_loss.item())

            self._train_batch += 1

            if self._train_batch % 10 == 0:
                Log.out("TH2VEC GENERATOR TRAIN", {
                    'train_batch': self._train_batch,
                    'dis_loss_avg': dis_loss_meter.avg,
                    'gen_loss_avg': gen_loss_meter.avg,
                })

                Log.out("<<<", {
                    'term': self._kernel.detokenize(
                        trm_smp[0].cpu().data.numpy(),
                    ),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/th2vec/generator/dis_loss",
                        dis_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/generator/gen_loss",
                        gen_loss_meter.avg, self._train_batch,
                    )

                dis_loss_meter = Meter()
                gen_loss_meter = Meter()

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--train_dataset_dir',
        type=str, help="train dataset directory",
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

    if args.train_dataset_dir is not None:
        config.override(
            'th2vec_train_dataset_dir',
            os.path.expanduser(args.train_dataset_dir),
        )
    if args.tensorboard_log_dir is not None:
        config.override(
            'tensorboard_log_dir',
            os.path.expanduser(args.tensorboard_log_dir),
        )
    if args.load_dir is not None:
        config.override(
            'th2vec_load_dir',
            os.path.expanduser(args.load_dir),
        )
    if args.save_dir is not None:
        config.override(
            'th2vec_save_dir',
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

    kernel = HolStepKernel(config.get('th2vec_theorem_length'))

    train_set = HolStepSet(
        kernel,
        os.path.expanduser(config.get('th2vec_train_dataset_dir')),
        premise_only=True,
    )

    # kernel.postprocess_compression(4096)
    # train_set.postprocess()

    train_dataset = HolStepTermDataset(train_set)

    th2vec = Th2VecGenerator(config, kernel)

    th2vec.init_training(train_dataset)

    th2vec.load(True)

    epoch = 0
    while True:
        th2vec.batch_train(epoch)
        th2vec.save()
        epoch += 1
