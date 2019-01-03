import argparse
import os
import random

from satispy import Variable, Cnf
from satispy.solver import Minisat

from utils.config import Config
from utils.log import Log
from utils.minisat import Minisat

class UnifKGenerator:
    def __init__(
            self,
            config: Config
    ) -> None:
        self._variable_count = config.get("generator_unif_k_variable_count")
        self._clause_length = config.get("generator_unif_k_clause_length")
        self._clause_count = config.get("generator_unif_k_clause_count")

    def generate(
            self,
    ) -> str:
        cnf = "c generator: UnifKGenerator"
        cnf += "\nc variable_count: {}".format(self._variable_count)
        cnf += "\nc clause_length: {}".format(self._clause_length)
        cnf += "\nc clause_count: {}".format(self._clause_count)
        cnf += "\nc clause_to_variable_ratio: {:.2f}".format(
            self._clause_count / self._variable_count,
        )
        cnf += "\np cnf {} {}".format(
            self._variable_count,
            self._clause_count,
        )

        variables = range(1, self._variable_count+1)
        for c in range(self._clause_count):
            clause = random.sample(variables, self._clause_length)
            cnf += "\n"
            for a in clause:
                if random.random() < 0.5:
                    cnf += "{} ".format(-a)
                else:
                    cnf += "{} ".format(a)
            cnf += "0"

        return cnf

class SelsamGenerator:
    def __init__(
            self,
    ) -> None:
        pass

def generate():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'config_path',
        type=str, help="path to the config file",
    )
    parser.add_argument(
        'sample_count',
        type=int, help="number of sample to generate",
    )
    parser.add_argument(
        'output_dir',
        type=str, help="directory to dump samples to",
    )

    parser.add_argument(
        '--generator_unif_k_variable_count',
        type=int, help="config override",
    )
    parser.add_argument(
        '--generator_unif_k_clause_length',
        type=int, help="config override",
    )
    parser.add_argument(
        '--generator_unif_k_clause_count',
        type=float, help="config override",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config_path)
    sample_count = args.sample_count
    output_dir = os.path.expanduser(args.output_dir)

    if args.generator_unif_k_variable_count is not None:
        config.override(
            'generator_unif_k_variable_count',
            args.generator_unif_k_variable_count,
        )
    if args.generator_unif_k_clause_length is not None:
        config.override(
            'generator_unif_k_clause_length',
            args.generator_unif_k_clause_length,
        )
    if args.generator_unif_k_clause_count is not None:
        config.override(
            'generator_unif_k_clause_count',
            args.generator_unif_k_clause_count,
        )

    generator = None

    if config.get("generator") == "unif_k":
        generator = UnifKGenerator(config)

    assert generator is not None

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    minisat = Minisat()

    sat = 0
    unsat = 0
    generated = 0

    total = 0
    total_sat = 0
    total_unsat = 0

    while generated < sample_count:
        cnf = generator.generate()
        success, assignment = minisat.solve(cnf)

        store = False
        total += 1
        if success:
            total_sat += 1
            if sat <= unsat:
                generated += 1
                sat += 1
                store = True
        if not success:
            total_unsat += 1
            if unsat <= sat:
                generated += 1
                unsat += 1
                store = True

        if store:
            with open(os.path.join(
                    output_dir, "{}.cnf".format(generated)
            ), 'w') as f:
                f.write(cnf)
                f.flush()

        if total % 10 == 0:
            Log.out(
                "Generating samples", {
                    'generator': config.get("generator"),
                    'total': total,
                    'generated': generated,
                    'sat_ratio': "{:.3f}".format(
                        total_sat / (total_sat + total_unsat)
                    ),
                })
