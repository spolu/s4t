import argparse
import os
import torch
import torch.distributed as distributed
import torch.utils.data.distributed
import torch.optim as optim
import torch.nn.functional as F

from dataset.holstep import HolStepKernel, HolStepSet
from dataset.holstep import HolStepPremiseDataset, HolStepClassificationDataset

# from generic.lr_scheduler import RampUpCosineLR

from tensorboardX import SummaryWriter

from th2vec.models.transformer import DP

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class Th2VecDirectPremiser:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config
        self._accumulation_step_count = \
            config.get('th2vec_accumulation_step_count')

        self._device = torch.device(config.get('device'))

        self._save_dir = config.get('th2vec_save_dir')
        self._load_dir = config.get('th2vec_load_dir')

        self._tb_writer = None
        if self._config.get('tensorboard_log_dir'):
            if self._config.get('distributed_rank') == 0:
                self._tb_writer = SummaryWriter(
                    self._config.get('tensorboard_log_dir'),
                )

        self._inner_model = DP(self._config).to(self._device)

        Log.out(
            "Initializing th2vec", {
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
            lr=self._config.get('th2vec_learning_rate'),
        )
        # self._scheduler = RampUpCosineLR(
        #     self._optimizer,
        #     self._config.get('th2vec_learning_rate_ramp_up'),
        #     self._config.get('th2vec_learning_rate_period'),
        #     self._config.get('th2vec_learning_rate_annealing'),
        # )

        self._train_sampler = None
        if self._config.get('distributed_training'):
            self._train_sampler = \
                torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                )

        pin_memory = False
        if self._config.get('device') != 'cpu':
            pin_memory = True

        batch_size = self._config.get('th2vec_batch_size') // \
            self._accumulation_step_count

        self._train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(self._train_sampler is None),
            pin_memory=pin_memory,
            num_workers=8,
            sampler=self._train_sampler,
        )

        Log.out('Training initialization', {
            "accumulation_step_count": self._accumulation_step_count,
            "world_size": self._config.get('distributed_world_size'),
            "batch_size": self._config.get('th2vec_batch_size'),
            "dataloader_batch_size": batch_size,
            "effective_batch_size": (
                self._config.get('th2vec_batch_size') *
                self._config.get('distributed_world_size')
            ),
        })

    def init_testing(
            self,
            test_dataset,
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
                    # self._scheduler.load_state_dict(
                    #     torch.load(
                    #         self._load_dir +
                    #         "/scheduler_{}.pt".format(rank),
                    #         map_location=self._device,
                    #     ),
                    # )

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
            # torch.save(
            #     self._scheduler.state_dict(),
            #     self._save_dir + "/scheduler_{}.pt".format(rank),
            # )

    def batch_train(
            self,
            epoch,
    ):
        assert self._train_loader is not None

        self._model.train()
        loss_meter = Meter()

        if self._config.get('distributed_training'):
            self._train_sampler.set_epoch(epoch)
        # self._scheduler.step()

        for it, (cnj, thr, pre) in enumerate(self._train_loader):
            res = self._model(
                cnj.to(self._device),
                thr.to(self._device),
            )

            loss = F.binary_cross_entropy(res, pre.to(self._device))

            loss.backward()

            if it % self._accumulation_step_count == 0:
                self._optimizer.step()
                self._optimizer.zero_grad()

            loss_meter.update(loss.item())

            self._train_batch += 1

            if self._train_batch % 10 == 0:
                Log.out("TH2VEC TRAIN", {
                    'train_batch': self._train_batch,
                    'loss_avg': loss_meter.avg,
                })

                if self._tb_writer is not None:
                    self._tb_writer.add_scalar(
                        "train/th2vec/direct_premiser/loss",
                        loss_meter.avg, self._train_batch,
                    )

                loss_meter = Meter()

        Log.out("EPOCH DONE", {
            'epoch': epoch,
            # 'learning_rate': self._scheduler.get_lr(),
        })

    def batch_test(
            self,
    ):
        assert self._test_loader is not None

        self._model.eval()
        loss_meter = Meter()

        hit = 0
        total = 0
        pos = 0
        neg = 0

        with torch.no_grad():
            for it, (cnj, thr, pre) in enumerate(self._test_loader):
                res = self._model(
                    cnj.to(self._device),
                    thr.to(self._device),
                )

                loss = F.binary_cross_entropy(res, pre.to(self._device))

                loss_meter.update(loss.item())

                limit = 0.50
                for i in range(res.size(0)):
                    if res[i].item() >= limit and pre[i].item() >= limit:
                        hit += 1
                    if res[i].item() < limit and pre[i].item() < limit:
                        hit += 1
                    if res[i].item() >= limit:
                        pos += 1
                    if res[i].item() < limit:
                        neg += 1
                    total += 1

        Log.out("TH2VEC TEST", {
            'batch_count': self._train_batch,
            'loss_avg': loss_meter.avg,
            'hit_rate': "{:.3f}".format(hit / total),
            'pos_rate': "{:.3f}".format(pos / total),
        })

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(
                "test/th2vec/direct_premiser/loss",
                loss_meter.avg, self._train_batch,
            )
            self._tb_writer.add_scalar(
                "test/th2vec/direct_premiser/hit_rate",
                hit / total, self._train_batch,
            )

    def embed(
            self,
    ):
        self._model.eval()


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

    kernel = HolStepKernel(config.get('th2vec_theorem_length'))

    train_set = HolStepSet(
        kernel,
        os.path.expanduser(config.get('th2vec_train_dataset_dir')),
        raw_formula=config.get('th2vec_raw_formula'),
        premise_only=config.get('th2vec_premise_only'),
    )
    test_set = HolStepSet(
        kernel,
        os.path.expanduser(config.get('th2vec_test_dataset_dir')),
        raw_formula=config.get('th2vec_raw_formula'),
        premise_only=config.get('th2vec_premise_only'),
    )

    # kernel.postprocess_compression(4096)
    # train_set.postprocess()
    # test_set.postprocess()

    train_dataset = None
    test_dataset = None

    if config.get('th2vec_premiser_dataset_type') == 'premise':
        train_dataset = HolStepPremiseDataset(train_set)
        test_dataset = HolStepPremiseDataset(test_set)
    if config.get('th2vec_premiser_dataset_type') == 'classification':
        train_dataset = HolStepClassificationDataset(train_set)
        test_dataset = HolStepClassificationDataset(test_set)

    assert train_dataset is not None
    assert test_dataset is not None

    th2vec = Th2VecDirectPremiser(config)

    th2vec.init_training(train_dataset)
    th2vec.init_testing(test_dataset)

    th2vec.load(True)

    epoch = 0
    while True:
        th2vec.batch_train(epoch)
        if epoch % 2 == 0:
            th2vec.batch_test()
            th2vec.save()
        epoch += 1


def test():
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
    if args.load_dir is not None:
        config.override(
            'th2vec_load_dir',
            os.path.expanduser(args.load_dir),
        )

    if config.get('device') != 'cpu':
        torch.cuda.set_device(torch.device(config.get('device')))

    kernel = HolStepKernel(config.get('th2vec_theorem_length'))

    train_set = HolStepSet(
        kernel,
        os.path.expanduser(config.get('th2vec_train_dataset_dir')),
        premise_only=config.get('th2vec_premise_only'),
    )
    test_set = HolStepSet(
        kernel,
        os.path.expanduser(config.get('th2vec_test_dataset_dir')),
        premise_only=config.get('th2vec_premise_only'),
    )

    train_dataset = None
    test_dataset = None

    if config.get('th2vec_premiser_dataset_type') == 'premise':
        train_dataset = HolStepPremiseDataset(train_set)
        test_dataset = HolStepPremiseDataset(test_set)
    if config.get('th2vec_premiser_dataset_type') == 'classification':
        train_dataset = HolStepClassificationDataset(train_set)
        test_dataset = HolStepClassificationDataset(test_set)

    assert train_dataset is not None
    assert test_dataset is not None

    th2vec = Th2VecDirectPremiser(config)

    th2vec.init_testing(test_dataset)
    th2vec.load(False)

    th2vec.batch_test()
