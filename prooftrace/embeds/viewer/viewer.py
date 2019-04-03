import argparse
import eventlet
import eventlet.wsgi
import os
import pickle

from flask import Flask
from flask import render_template

from eventlet.green import threading

from utils.config import Config
from utils.log import Log
from utils.str2bool import str2bool

_app = Flask(__name__)

_config = None

_embeds = None
_dump = None


@_app.route('/prooftrace_embeds')
def view_emebeds():
    global _dump

    return render_template(
        'prooftrace_embeds.html',
        dump=_dump,
    )


def run_server():
    global _app

    Log.out(
        "Starting embeds viewer server", {
            'port': 5000,
        })
    address = ('0.0.0.0', 5000)
    try:
        eventlet.wsgi.server(eventlet.listen(address), _app)
    except KeyboardInterrupt:
        Log.out(
            "Stopping viewer server", {})


def run():
    global _config
    global _embeds
    global _dump

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
        '--test',
        type=str2bool, help="confg override",
    )

    args = parser.parse_args()

    _config = Config.from_file(args.config_path)

    if args.dataset_size is not None:
        _config.override(
            'prooftrace_dataset_size',
            args.dataset_size,
        )

    if args.test:
        dataset_dir = os.path.join(
            os.path.expanduser(_config.get('prooftrace_dataset_dir')),
            _config.get('prooftrace_dataset_size'),
            "test_traces",
        )
    else:
        dataset_dir = os.path.join(
            os.path.expanduser(_config.get('prooftrace_dataset_dir')),
            _config.get('prooftrace_dataset_size'),
            "train_traces",
        )

    ptre_path = os.path.join(dataset_dir, 'traces.embeds')
    Log.out("Loading ProofTraceEmbeds", {
        'path': ptre_path,
    })
    with open(ptre_path, 'rb') as f:
        _embeds = pickle.load(f)
        _dump = dict(_embeds)

    # files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    # for p in files:
    #     if re.search("\\.actions$", p) is None:
    #         continue
    #     Log.out("Loading ProofTraceActions", {
    #         'path': p,
    #     })
    #     with open(p, 'rb') as f:
    #         ptra = pickle.load(f)
    #         _traces.append(ptra)

    t = threading.Thread(target=run_server)
    t.start()
    t.join()
