import argparse
import copy
import torch
import os
import torch.optim as optim

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.lstm import H

from utils.config import Config
from utils.log import Log


def run():
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
        '--device',
        type=str, help="config override",
    )

    args = parser.parse_args()

    config = Config.from_file(args.config_path)

    if args.device is not None:
        config.override('device', args.device)
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

    device = torch.device(config.get('device'))
    load_dir = config.get('prooftrace_load_dir')
    save_dir = config.get('prooftrace_save_dir')

    modules = {
        'E': E(config).to(device),
        'HP': H(config).to(device),
        'HV': H(config).to(device),
        'PH': PH(config).to(device),
        'VH': VH(config).to(device),
    }
    optimizer = optim.Adam(
        [
            {'params': modules['E'].parameters()},
            {'params': modules['HP'].parameters()},
            {'params': modules['PH'].parameters()},
            {'params': modules['VH'].parameters()},
        ],
        lr=config.get('prooftrace_ppo_learning_rate')
    )

    modules['E'].load_state_dict(torch.load(
        load_dir + "/model_E.pt",
        map_location=device,
    ))
    modules['HP'].load_state_dict(torch.load(
        load_dir + "/model_H.pt",
        map_location=device,
    ))
    modules['HV'].load_state_dict(torch.load(
        load_dir + "/model_H.pt",
        map_location=device,
    ))
    modules['PH'].load_state_dict(torch.load(
        load_dir + "/model_PH.pt",
        map_location=device,
    ))
    modules['VH'].load_state_dict(torch.load(
        load_dir + "/model_VH.pt",
        map_location=device,
    ))
    optimizer.load_state_dict(torch.load(
        load_dir + "/optimizer.pt",
        map_location=device,
    ))

    new_params = copy.deepcopy(optimizer.param_groups[1])

    optimizer.param_groups.insert(1, new_params)

    if save_dir:
        torch.save(
            modules['E'].state_dict(),
            save_dir + "/model_E.pt",
        )
        torch.save(
            modules['HP'].state_dict(),
            save_dir + "/model_PHI.pt",
        )
        torch.save(
            modules['HV'].state_dict(),
            save_dir + "/model_VHI.pt",
        )
        torch.save(
            modules['PH'].state_dict(),
            save_dir + "/model_PH.pt",
        )
        torch.save(
            modules['VH'].state_dict(),
            save_dir + "/model_VH.pt",
        )
        torch.save(
            optimizer.state_dict(),
            save_dir + "/optimizer.pt",
        )

    Log.out('DONE')


def verify():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
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
    if args.load_dir is not None:
        config.override(
            'prooftrace_load_dir',
            os.path.expanduser(args.load_dir),
        )

    device = torch.device(config.get('device'))
    load_dir = config.get('prooftrace_load_dir')

    modules = {
        'E': E(config).to(device),
        'PHI': H(config).to(device),
        'VHI': H(config).to(device),
        'PH': PH(config).to(device),
        'VH': VH(config).to(device),
    }

    optimizer = optim.Adam(
        [
            {'params': modules['E'].parameters()},
            {'params': modules['PHI'].parameters()},
            {'params': modules['VHI'].parameters()},
            {'params': modules['PH'].parameters()},
            {'params': modules['VH'].parameters()},
        ],
        lr=config.get('prooftrace_ppo_learning_rate')
    )

    assert load_dir is not None

    modules['E'].load_state_dict(torch.load(
        load_dir + "/model_E.pt",
        map_location=device,
    ))
    modules['PHI'].load_state_dict(torch.load(
        load_dir + "/model_PHI.pt",
        map_location=device,
    ))
    modules['VHI'].load_state_dict(torch.load(
        load_dir + "/model_VHI.pt",
        map_location=device,
    ))
    modules['PH'].load_state_dict(torch.load(
        load_dir + "/model_PH.pt",
        map_location=device,
    ))
    modules['VH'].load_state_dict(torch.load(
        load_dir + "/model_VH.pt",
        map_location=device,
    ))
    optimizer.load_state_dict(torch.load(
        load_dir + "/optimizer.pt",
        map_location=device,
    ))

    Log.out('OK')
