import argparse
import os
import torch
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F

from dataset.holstep import HolStepRelationsDataset

from tensorboardX import SummaryWriter

from th2vec.models.transformer import E

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Th2Vec:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('th2vec_save_dir')
        self._load_dir = config.get('th2vec_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model = E(self._config).to(self._device)

        Log.out(
            "Initializing th2vec", {
                'parameter_count': self._inner_model.parameters_count()
            },
        )

        self._model = self._inner_model

    def init_training(
            self,
            train_dataset: HolStepRelationsDataset,
    ):
        if self._config.get('distributed_training'):
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._inner_model,
                device_ids=[self._device],
            )

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._config.get('th2vec_learning_rate'),
            betas=(
                self._config.get('th2vec_adam_beta_1'),
                self._config.get('th2vec_adam_beta_2'),
            ),
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

    def init_testing(
            self,
            test_dataset: HolStepRelationsDataset,
    ):
        pin_memory = False
        if self._config.get('device') != 'cpu':
            pin_memory = True

        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._config.get('th2vec_batch_size'),
            shuffle=False,
            num_workers=8,
            pin_memory=pin_memory,
        )

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
                self._inner_model.load_state_dict(
                    torch.load(
                        self._load_dir + "/model_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                if training:
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
                "Saving th2vec models", {
                    'save_dir': self._save_dir,
                })

            torch.save(
                self._inner_model.state_dict(),
                self._save_dir + "/model_{}.pt".format(rank),
            )
            torch.save(
                self._optimizer.state_dict(),
                self._save_dir + "/optimizer_{}.pt".format(rank),
            )

    def batch_train(self):
        assert self._train_loader is not None

        self._model.train()

        all_loss_meter = Meter()
        rel_loss_meter = Meter()
        rnd_loss_meter = Meter()
        nrm_loss_meter = Meter()
        nrm_mean_meter = Meter()
        rel_simi_meter = Meter()
        rnd_simi_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(self._train_batch)

        for it, (inp, out, rnd) in enumerate(self._train_loader):
            inp_embed = self._model(inp.to(self._device))
            out_embed = self._model(out.to(self._device))
            rnd_embed = self._model(rnd.to(self._device))

            rel_loss = F.mse_loss(inp_embed, out_embed.detach())
            rnd_loss = F.mse_loss(inp_embed, rnd_embed.detach())

            nrm_mean = torch.norm(inp_embed, dim=1).mean()
            nrm_loss = F.mse_loss(
                torch.norm(inp_embed, dim=1),
                torch.ones(inp_embed.size(0)).to(self._device),
            )

            rel_simi = F.cosine_similarity(
                inp_embed, out_embed.detach()
            )
            rnd_simi = F.cosine_similarity(
                inp_embed, rnd_embed.detach()
            )

            # all_loss = torch.norm(1 - rel_simi) + torch.norm(-1 - rnd_simi)
            all_loss = 100 * (rel_loss - rnd_loss) + nrm_loss / 10

            self._optimizer.zero_grad()
            all_loss.backward()
            self._optimizer.step()

            all_loss_meter.update(all_loss.item())
            rel_loss_meter.update(rel_loss.item())
            rel_simi_meter.update(rel_simi.mean().item())
            rnd_loss_meter.update(rnd_loss.item())
            rnd_simi_meter.update(rnd_simi.mean().item())
            nrm_loss_meter.update(nrm_loss.item())
            nrm_mean_meter.update(nrm_mean.item())

            self._train_batch += 1

            if self._train_batch % 10 == 0:
                Log.out("TH2VEC TRAIN", {
                    'train_batch': self._train_batch,
                    'all_loss_avg': all_loss_meter.avg,
                    'rel_loss_avg': rel_loss_meter.avg,
                    'rnd_loss_avg': rnd_loss_meter.avg,
                    'nrm_loss_avg': nrm_loss_meter.avg,
                    'rel_simi_avg': rel_simi_meter.avg,
                    'rnd_simi_avg': rnd_simi_meter.avg,
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/th2vec/all_loss",
                        all_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/rel_loss",
                        rel_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/rel_simi",
                        rel_simi_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/rnd_loss",
                        rnd_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/rnd_simi",
                        rnd_simi_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/nrm_loss",
                        nrm_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/th2vec/nrm_mean",
                        nrm_mean_meter.avg, self._train_batch,
                    )

                all_loss_meter = Meter()
                rnd_loss_meter = Meter()
                rel_loss_meter = Meter()
                nrm_loss_meter = Meter()
                nrm_mean_meter = Meter()
                rel_simi_meter = Meter()
                rnd_simi_meter = Meter()

            if self._train_batch % 100 == 0:
                self._model.eval()
                self.batch_test()
                self.save()
                self._model.train()

    def batch_test(
            self,
    ):
        assert self._test_loader is not None

        self._model.eval()

        all_loss_meter = Meter()
        rel_loss_meter = Meter()
        rnd_loss_meter = Meter()
        nrm_loss_meter = Meter()
        nrm_mean_meter = Meter()
        rel_simi_meter = Meter()
        rnd_simi_meter = Meter()

        with torch.no_grad():
            for it, (inp, out, rnd) in enumerate(self._test_loader):
                if (it+1) % 10 == 0:
                    break

                inp_embed = self._model(inp.to(self._device))
                out_embed = self._model(out.to(self._device))
                rnd_embed = self._model(rnd.to(self._device))

                rel_loss = F.mse_loss(inp_embed, out_embed.detach())
                rnd_loss = F.mse_loss(inp_embed, rnd_embed.detach())

                nrm_mean = torch.norm(inp_embed, dim=1).mean()
                nrm_loss = F.mse_loss(
                    torch.norm(inp_embed, dim=1),
                    torch.ones(inp_embed.size(0)).to(self._device),
                )

                rel_simi = F.cosine_similarity(
                    inp_embed, out_embed.detach()
                )
                rnd_simi = F.cosine_similarity(
                    inp_embed, rnd_embed.detach()
                )

                # all_loss = torch.norm(1 - rel_simi) + torch.norm(-1 - rnd_simi)
                all_loss = 100 * (rel_loss - rnd_loss) + nrm_loss / 10

                all_loss_meter.update(all_loss.item())
                rel_loss_meter.update(rel_loss.item())
                rel_simi_meter.update(rel_simi.mean().item())
                rnd_loss_meter.update(rnd_loss.item())
                rnd_simi_meter.update(rnd_simi.mean().item())
                nrm_loss_meter.update(nrm_loss.item())
                nrm_mean_meter.update(nrm_mean.item())

        Log.out("SAT TEST", {
            'batch_count': self._train_batch,
            'all_loss_avg': all_loss_meter.avg,
            'rel_loss_avg': rel_loss_meter.avg,
            'rnd_loss_avg': rnd_loss_meter.avg,
            'nrm_loss_avg': nrm_loss_meter.avg,
            'rel_simi_avg': rel_simi_meter.avg,
            'rnd_simi_avg': rnd_simi_meter.avg,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/th2vec/all_loss",
                all_loss_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/rel_loss",
                rel_loss_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/rel_simi",
                rel_simi_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/rnd_loss",
                rnd_loss_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/rnd_simi",
                rnd_simi_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/nrm_loss",
                nrm_loss_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/nrm_mean",
                nrm_mean_meter.avg, self._train_batch,
            )


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
        '--test_dataset_dir',
        type=str, help="test dataset directory",
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
    if args.test_dataset_dir is not None:
        config.override(
            'th2vec_test_dataset_dir',
            os.path.expanduser(args.test_dataset_dir),
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

    train_dataset = HolStepRelationsDataset(
        config,
        os.path.expanduser(config.get('th2vec_train_dataset_dir')),
    )
    test_dataset = HolStepRelationsDataset(
        config,
        os.path.expanduser(config.get('th2vec_test_dataset_dir')),
    )

    th2vec = Th2Vec(config).load(True)

    th2vec.init_training(train_dataset)
    th2vec.init_testing(test_dataset)

    while True:
        th2vec.batch_train()
