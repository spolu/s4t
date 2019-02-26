import argparse
import os
import torch
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F

from dataset.prooftrace import ProofTraceLMDataset, lm_collate, Action

from tensorboardX import SummaryWriter

from prooftrace.models.transformer import LM

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class LanguageModeler:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._accumulation_step_count = \
            config.get('prooftrace_accumulation_step_count')

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('prooftrace_save_dir')
        self._load_dir = config.get('prooftrace_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model = LM(self._config).to(self._device)

        Log.out(
            "Initializing prooftrace LanguageModeler", {
                'parameter_count': self._inner_model.parameters_count()
            },
        )

        self._model = self._inner_model
        self._train_batch = 0

    def init_training(
            self,
            train_dataset,
    ):
        if self._config.get('distributed_training'):
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._inner_model,
                device_ids=[self._device],
            )

        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._config.get('prooftrace_learning_rate'),
        )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        batch_size = self._config.get('prooftrace_batch_size') // \
            self._accumulation_step_count

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self._train_sampler is None),
            sampler=self._train_sampler,
            collate_fn=lm_collate,
        )

        Log.out('Training initialization', {
            "accumulation_step_count": self._accumulation_step_count,
            "world_size": self._config.get('distributed_world_size'),
            "batch_size": self._config.get('prooftrace_batch_size'),
            "dataloader_batch_size": batch_size,
            "effective_batch_size": (
                self._config.get('prooftrace_batch_size') *
                self._config.get('distributed_world_size')
            ),
        })

    def init_testing(
            self,
            test_dataset,
    ):
        self._test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._config.get('prooftrace_batch_size'),
            shuffle=False,
            collate_fn=lm_collate,
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
                    "Loading prooftrace models", {
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
                "Saving prooftrace models", {
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

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model.train()

        trg_loss_meter = Meter()
        trg_simi_meter = Meter()
        ext_loss_meter = Meter()
        ext_simi_meter = Meter()
        emd_norm_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)
        # self._scheduler.step()

        for it, (idx, trc) in enumerate(self._train_loader):
            embeds = self._inner_model.embed(trc)
            targets = embeds.clone().detach()

            extract = self._inner_model.embed(
                [[Action.from_action('EXTRACT')]]
            )

            for i, ext in enumerate(idx):
                embeds[i][ext] = extract[0][0]

            predictions = self._model(embeds)

            trg_loss = 0
            ext_loss = 0
            for i in range(len(idx)):
                trg_loss += F.mse_loss(
                    predictions[i][idx[i]],
                    targets[i][idx[i]]
                )
                ext_loss += F.mse_loss(
                    predictions[i][idx[i]],
                    embeds[i][idx[i]]
                )
            trg_loss /= len(idx)
            ext_loss /= len(idx)

            trg_loss.backward()

            if it % self._accumulation_step_count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()

            trg_loss_meter.update(trg_loss.item())
            ext_loss_meter.update(ext_loss.item())
            for i in range(len(idx)):
                trg_simi_meter.update(
                    F.cosine_similarity(
                        predictions[i][idx[i]],
                        targets[i][idx[i]],
                        dim=0).mean(),
                )
                ext_simi_meter.update(
                    F.cosine_similarity(
                        predictions[i][idx[i]],
                        embeds[i][idx[i]],
                        dim=0).mean(),
                )
            emd_norm_meter.update(torch.norm(embeds, dim=2).mean())

            if self._train_batch % 10 == 0:
                Log.out("PROOFTRACE TRAIN", {
                    'train_batch': self._train_batch,
                    'ext_loss_avg': "{:.4f}".format(ext_loss_meter.avg),
                    'trg_loss_avg': "{:.4f}".format(trg_loss_meter.avg),
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/prooftrace/language_modeler/trg_loss",
                        trg_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/prooftrace/language_modeler/trg_simi",
                        trg_simi_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/prooftrace/language_modeler/ext_loss",
                        ext_loss_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/prooftrace/language_modeler/ext_simi",
                        ext_simi_meter.avg, self._train_batch,
                    )
                    self._tb_writer.add_scalar(
                        "train/prooftrace/language_modeler/emd_norm",
                        emd_norm_meter.avg, self._train_batch,
                    )

                trg_loss_meter = Meter()
                trg_simi_meter = Meter()
                ext_loss_meter = Meter()
                ext_simi_meter = Meter()
                emd_norm_meter = Meter()

            if self._train_batch % 100 == 0:
                self.save()

            self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })

    def batch_test(
            self,
    ):
        assert self._test_loader is not None

        self._model.eval()
        loss_meter = Meter()

        with torch.no_grad():
            for it, (trc, idx) in enumerate(self._test_loader):
                embeds = self._inner_model.embed(trc)
                targets = embeds.clone().detach()

                extract = self._inner_model.embed(
                    [[Action.from_action('EXTRACT')]]
                )

                for i, idx in enumerate(idx):
                    embeds[i][idx] = extract[0][0]

                predictions = self._model(embeds)

                loss = F.mse_loss(predictions, targets)

                loss_meter.update(loss.item())

        Log.out("PROOFTRACE TEST", {
            'batch_count': self._train_batch,
            'loss_avg': loss_meter.avg,
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/prooftrace/language_modeler/loss",
                loss_meter.avg, self._train_batch,
            )


def train():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        '--dataset_dir',
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

    if args.dataset_dir is not None:
        config.override(
            'prooftrace_dataset_dir',
            os.path.expanduser(args.dataset_dir),
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

    train_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            'train_traces',
        ),
        config.get('prooftrace_sequence_length'),
    )
    test_dataset = ProofTraceLMDataset(
        os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            'test_traces',
        ),
        config.get('prooftrace_sequence_length'),
    )

    lmodeler = LanguageModeler(config)

    lmodeler.init_training(train_dataset)
    lmodeler.init_testing(test_dataset)

    lmodeler.load(True)

    epoch = 0
    while True:
        lmodeler.batch_train(epoch)
        lmodeler.batch_test()
        lmodeler.save()
        epoch += 1
