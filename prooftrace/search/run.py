import argparse
import datetime
import gzip
import pickle
import os
import random
import re
import time

from prooftrace.models.model import LModel
from prooftrace.prooftrace import INV_PREPARE_TOKENS, ProofTraceActions
from prooftrace.repl.repl import REPL
from prooftrace.search.beam import Beam
from prooftrace.search.particle_filter import ParticleFilter
from prooftrace.search.policy_sample import PolicySample
from prooftrace.search.random import Random

from utils.config import Config
from utils.log import Log
from utils.str2bool import str2bool


def search():
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
    parser.add_argument(
        '--train',
        type=str2bool, help="search training set",
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

    train = False
    if args.train is not None:
        train = args.train

    if train:
        dataset_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            'train_traces'
        )
    else:
        dataset_dir = os.path.join(
            os.path.expanduser(config.get('prooftrace_dataset_dir')),
            config.get('prooftrace_dataset_size'),
            'test_traces'
        )

    assert os.path.isdir(dataset_dir)
    files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]
    cases = []

    with gzip.open(
            os.path.join(
                os.path.expanduser(config.get('prooftrace_dataset_dir')),
                config.get('prooftrace_dataset_size'),
                'traces.tokenizer',
            ), 'rb') as f:
        tokenizer = pickle.load(f)

    for p in files:
        match = re.search("_(\\d+)_(\\d+)\\.actions$", p)
        if match is None:
            continue
        ptra_len = int(match.group(1))
        cases.append((p, ptra_len))

    Log.out(
        "Loaded ProofTraceActions", {
            'cases': len(cases),
        })

    model = LModel(config).load()

    cases = sorted(cases, key=lambda c: c[1])

    for i in range(len(cases)):
        c = cases[i][0]
        with gzip.open(c, 'rb') as f:
            ground = pickle.load(f)

        ptra = ProofTraceActions(
            'SEARCH-{}-{}'.format(
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
        repl = REPL(tokenizer)
        target = repl.prepare(ptra)

        offset = 0
        fixed_gamma = config.get('prooftrace_search_fixed_gamma')
        if fixed_gamma > 0:
            gamma_len = max(ground.action_len() - fixed_gamma, 0)
            offset = ground.prepare_len() + gamma_len

            for i in range(gamma_len):
                assert ground.prepare_len() + i < ground.len() - 1
                pos = ground.prepare_len() + i

                action = ground.actions()[pos]
                argument = ground.arguments()[pos]

                thm = repl.apply(action)

                action._index = thm.index()
                argument._index = thm.index()

                ptra.append(action, argument)

        Log.out("TARGET", {
            'name': ground.name(),
            'prepare_length': ground.prepare_len(),
            'action_length': ground.action_len(),
            'summary': ground.summary(offset),
            'theorem': target.thm_string(False, True),
        })

        search = None
        if config.get('prooftrace_search_type') == 'beam':
            search = Beam(config, model, ptra, repl, target)
        if config.get('prooftrace_search_type') == 'particle_filter':
            search = ParticleFilter(config, model, ptra, repl, target)
        if config.get('prooftrace_search_type') == 'policy_sample':
            search = PolicySample(config, model, ptra, repl, target)
        if config.get('prooftrace_search_type') == 'random':
            search = Random(config, model, ptra, repl, target)
        assert search is not None

        depth = config.get('prooftrace_sequence_length') - \
            ground.prepare_len()

        if fixed_gamma != 0:
            if 4 * fixed_gamma < depth:
                depth = fixed_gamma * 4
        else:
            if 2 * ground.action_len() < depth:
                depth = 2 * ground.action_len()

        for i in range(depth):
            if fixed_gamma != 0:
                conclusion = (i >= fixed_gamma * 2)
            else:
                conclusion = (i >= ground.action_len())

            step_start = time.time()
            done, ptra, proved = search.step(offset, conclusion)
            step_end = time.time()

            # Log.out('STEP', {
            #     'i': i,
            #     'done': done,
            #     'proved': proved,
            #     'time': "{:.2f}".format(step_end - step_start),
            # })
            if done:
                if proved:
                    Log.out("DEMONSTRATED", {
                        'theorem': target.thm_string(False, True),
                    })
                break

            if (step_end - step_start) > \
                    config.get('prooftrace_search_step_timeout'):
                break

        Log.out("FINISH", {
            'summary': ptra.summary(offset),
        })
        if config.get('prooftrace_search_type') == 'random' \
                and search.last_thm() is not None:
            Log.out("GENERATED", {
                'theorem': search.last_thm().thm_string(False, True)
            })
