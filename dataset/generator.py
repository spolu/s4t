import argparse
import os
import random
import threading
import concurrent.futures

from utils.config import Config
from utils.log import Log
from utils.minisat import Minisat


class BaseRandKGenerator:
    def __init__(
            self,
            clause_length,
            variable_count,
    ) -> None:
        self._clause_length = clause_length
        self._variable_count = variable_count

    def _produce(
            self,
            clause_count,
    ) -> str:
        cnf = "c generator: {}".format(self.name())
        cnf += "\nc clause_length: {}".format(self._clause_length)
        cnf += "\nc variable_count: {}".format(self._variable_count)
        cnf += "\nc clause_count: {}".format(clause_count)
        cnf += "\nc clause_to_variable_ratio: {:.2f}".format(
            clause_count / self._variable_count,
        )
        cnf += "\np cnf {} {}".format(
            self._variable_count,
            clause_count,
        )

        variables = range(1, self._variable_count+1)
        for c in range(clause_count):
            clause = random.sample(variables, self._clause_length)
            cnf += "\n"
            for a in clause:
                if random.random() < 0.5:
                    cnf += "{} ".format(-a)
                else:
                    cnf += "{} ".format(a)
            cnf += "0"

        return cnf

    def generate(
            self,
            dataset_dir,
            prefix,
            sample_count,
            num_workers=8,
    ) -> None:
        minisat = Minisat()
        chunk = int(sample_count/num_workers)

        def process(worker):
            total = 0
            total_sat = 0
            total_unsat = 0

            generated = 0
            sat = 0
            unsat = 0

            while generated < chunk:
                cnf = self.produce()
                success, assignment = minisat.solve(cnf)

                store = False
                header = ""
                total += 1
                if success:
                    total_sat += 1
                    if sat <= unsat:
                        header = "c SAT\n"
                        generated += 1
                        sat += 1
                        store = True
                if not success:
                    total_unsat += 1
                    if unsat <= sat:
                        header = "c UNSAT\n"
                        generated += 1
                        unsat += 1
                        store = True

                if store:
                    with open(os.path.join(
                            dataset_dir, "{}_{}.cnf".format(
                                prefix, worker*chunk + generated,
                            )
                    ), 'w') as f:
                        f.write(header)
                        f.write(cnf)
                        f.flush()

                if total % 100 == 0:
                    Log.out(
                        "Generating samples", {
                            'generator': self.name(),
                            'total': total,
                            'sat_ratio': "{:.3f}".format(
                                total_sat / (total_sat + total_unsat)
                            ),
                            'worker': worker,
                            'generated': generated,
                        })

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
        ) as executor:
            executor.map(process, range(num_workers))


class ExtRandKGenerator(BaseRandKGenerator):
    def __init__(
            self,
            clause_length,
            variable_count,
            min_clause_count,
            max_clause_count,
    ) -> None:
        super(ExtRandKGenerator, self).__init__(
            clause_length,
            variable_count,
        )

        self._min_clause_count = min_clause_count
        self._max_clause_count = max_clause_count

    def produce(
            self,
    ) -> str:
        clause_count = random.randint(
            self._min_clause_count,
            self._max_clause_count,
        )

        return super(ExtRandKGenerator, self)._produce(clause_count)

    def name(
            self,
    ) -> str:
        return 'ext_rand_k'


class FixRandKGenerator(BaseRandKGenerator):
    def __init__(
            self,
            clause_length,
            variable_count,
            clause_count,
    ) -> None:
        super(FixRandKGenerator, self).__init__(
            clause_length,
            variable_count,
        )

        self._clause_count = clause_count

    def produce(
            self,
    ) -> str:
        return super(FixRandKGenerator, self)._produce(self._clause_count)

    def name(
            self,
    ) -> str:
        return 'fix_rand_k'


class SelsamGenerator:
    def __init__(
            self,
    ) -> None:
        pass


class BaseMixedGenerator:
    def __init__(
            self,
            plan,
    ) -> None:
        self._plan = plan

    def generate(
            self,
            dataset_dir,
            num_workers=8,
    ) -> None:
        def generate(instr):
            instr['generator'].generate(
                dataset_dir,
                instr['prefix'],
                instr['sample_count'],
                instr['num_workers'],
            )

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
        ) as executor:
            executor.map(generate, self._plan)


class MixedRandK8Generator(BaseMixedGenerator):
    def __init__(
            self,
            test,
    ) -> None:
        samples = 100_000
        if test:
            samples = 100

        fix_args = [
            [3, 4, 20],
            [4, 4, 48],
            [3, 8, 40],
            [4, 8, 88],
            [6, 8, 370],
        ]
        ext_args = [
            [3, 4, 10, 30],
            [4, 4, 36, 72],
            [3, 8, 20, 60],
            [4, 8, 66, 132],
        ]

        plan = []

        for args in fix_args:
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args in ext_args:
            plan.append({
                'generator': ExtRandKGenerator(*args),
                'prefix': "ext_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedRandK8Generator, self).__init__(plan)


class MixedRandK16Generator(BaseMixedGenerator):
    def __init__(
            self,
            test,
    ) -> None:
        samples = 200_000
        if test:
            samples = 200

        fix_args = [
            [3, 4, 20],
            [4, 4, 48],
            [3, 8, 40],
            [4, 8, 88],
            [6, 8, 370],
            [3, 16, 74],
            [4, 16, 166],
        ]
        ext_args = [
            [3, 4, 10, 30],
            [4, 4, 36, 72],
            [3, 8, 20, 60],
            [4, 8, 66, 132],
            [3, 16, 55, 110],
            [4, 16, 124, 249],
        ]

        plan = []

        for args in fix_args:
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args in ext_args:
            plan.append({
                'generator': ExtRandKGenerator(*args),
                'prefix': "ext_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedRandK16Generator, self).__init__(plan)


class MixedRandK32Generator(BaseMixedGenerator):
    def __init__(
            self,
            test,
    ) -> None:
        samples = 300_000
        if test:
            samples = 300

        fix_args = [
            [3, 4, 20],
            [4, 4, 48],
            [3, 8, 40],
            [4, 8, 88],
            [6, 8, 370],
            [3, 16, 74],
            [4, 16, 166],
            [3, 32, 142],
            [4, 32, 320],
        ]
        ext_args = [
            [3, 4, 10, 30],
            [4, 4, 36, 72],
            [3, 8, 20, 60],
            [4, 8, 66, 132],
            [3, 16, 55, 110],
            [4, 16, 124, 249],
            [3, 32, 106, 213],
            [4, 32, 240, 480],
        ]

        plan = []

        for args in fix_args:
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args in ext_args:
            plan.append({
                'generator': ExtRandKGenerator(*args),
                'prefix': "ext_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedRandK32Generator, self).__init__(plan)


def generate():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        'generator',
        type=str, help="generator to use",
    )
    parser.add_argument(
        'dataset_dir',
        type=str, help="directory to dump samples to",
    )

    args = parser.parse_args()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    generator = None

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)

    if args.generator == "fix_rand_k":
        generator = FixRandKGenerator(
            # 3, 4, 20,
            # 4, 4, 48,
            # 3, 8, 40,
            # 4, 8, 88,
            # 6, 8, 370,
            # 3, 16, 74,
            # 4, 16, 166,
            # 3, 32, 142,
            # 4, 32, 320,

            4, 16, 124,
        )
        generator.generate(dataset_dir, '', 1000, 8)
    if args.generator == "ext_rand_k":
        generator = ExtRandKGenerator(
            # 3, 4, 10, 30,
            # 4, 4, 36, 72,
            # 3, 8, 20, 60,
            # 4, 8, 66, 132,
            # 3, 16, 55, 110,
            # 4, 16, 124, 249,
            # 3, 32, 106, 213,
            # 4, 32, 240, 480,

            4, 32, 240, 480,
        )
        generator.generate(dataset_dir, '', 1000, 8)
    if args.generator == "mixed_rand_k_8":
        generator = MixedRandK8Generator(False)
        generator.generate(dataset_dir, 2)
    if args.generator == "mixed_rand_k_16":
        generator = MixedRandK16Generator(False)
        generator.generate(dataset_dir, 2)
    if args.generator == "mixed_rand_k_32":
        generator = MixedRandK32Generator(False)
        generator.generate(dataset_dir, 2)

    assert generator is not None
