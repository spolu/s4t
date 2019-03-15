import argparse
import os
import torch
import torch.distributed as distributed
import torch.optim as optim

from prooftrace.models.heads import PH, VH
from prooftrace.models.transformer import H

from tensorboardX import SummaryWriter

from utils.config import Config
from utils.meter import Meter
from utils.log import Log
from utils.str2bool import str2bool


class PPO:
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

        self._inner_model_H = H(self._config).to(self._device)
        self._inner_model_PH = PH(self._config).to(self._device)
        self._inner_model_VH = VH(self._config).to(self._device)

        Log.out(
            "Initializing prooftrace PPO", {
                'parameter_count_H': self._inner_model_H.parameters_count(),
                'parameter_count_PH': self._inner_model_PH.parameters_count(),
                'parameter_count_VH': self._inner_model_VH.parameters_count(),
            },
        )

        self._model_H = self._inner_model_H
        self._model_PH = self._inner_model_PH
        self._model_VH = self._inner_model_VH

        self._train_batch = 0

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
            lr=self._config.get('prooftrace_learning_rate'),
        )

        self._batch_size = self._config.get('prooftrace_batch_size') // \
            self._accumulation_step_count

        Log.out('Training initialization', {
            "accumulation_step_count": self._accumulation_step_count,
            "world_size": self._config.get('distributed_world_size'),
            "batch_size": self._config.get('prooftrace_batch_size'),
            "dataloader_batch_size": self._batch_size,
            "effective_batch_size": (
                self._config.get('prooftrace_batch_size') *
                self._config.get('distributed_world_size')
            ),
        })

    def init_testing(
            self,
            test_dataset,
    ):
        self._batch_size = self._config.get('prooftrace_batch_size') // \
            self._accumulation_step_count

    def load(
            self,
            training=True,
    ):
        rank = self._config.get('distributed_rank')

        if self._load_dir:
            if os.path.isfile(
                    self._load_dir + "/model_H_{}.pt".format(rank)
            ):
                Log.out(
                    "Loading prooftrace", {
                        'load_dir': self._load_dir,
                    })
                self._inner_model_H.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_H_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                self._inner_model_PH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_PH_{}.pt".format(rank),
                        map_location=self._device,
                    ),
                )
                self._inner_model_VH.load_state_dict(
                    torch.load(
                        self._load_dir +
                        "/model_VH_{}.pt".format(rank),
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
        self._model_H.train()
        self._model_PH.train()
        self._model_VH.train()

        self._train_batch += 1

        Log.out("EPOCH DONE", {
            'epoch': epoch,
        })

    def batch_test(
            self,
    ):
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
