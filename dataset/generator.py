import argparse
import concurrent.futures
import numpy as np
import os
import random
import typing

from utils.log import Log
from utils.minisat import Minisat
from utils.str2bool import str2bool


def cnf_from_clauses(
        clauses: typing.List[typing.List[int]],
        generator_name: str,
) -> str:
    variable_count = 0

    for cl in clauses:
        for v in cl:
            t = max(v, -v)
            assert t > 0
            if t > variable_count:
                variable_count = t

    cnf = "c generator: {}".format(generator_name)
    cnf += "\np cnf {} {}".format(variable_count, len(clauses))
    for cl in clauses:
        cnf += "\n"
        for v in cl:
            cnf += "{} ".format(v)
        cnf += "0"

    return cnf, variable_count, len(clauses)


def convert_to_8cnf(
        clauses: typing.List[typing.List[int]],
) -> typing.List[typing.List[int]]:
    good_clauses = [c for c in clauses if len(c) <= 8]
    bad_clauses = [c for c in clauses if len(c) > 8]

    def variable_count(clauses):
        return len(
            set(abs(literal) for clause in clauses for literal in clause)
        )

    def reduce_clause(clause, next_var):
        cur_var = next_var
        if len(clause) <= 8:
            return [clause], next_var

        reduced, next_var = reduce_clause(
            [cur_var] + list(clause[2:]),
            next_var+1,
        )

        return [[clause[0], clause[1], -cur_var]] + reduced, next_var

    next_var = variable_count(clauses) + 1
    for c in bad_clauses:
        reduced, next_var = reduce_clause(c, next_var)
        good_clauses.extend(reduced)

    # print("{}/{} => {}/{}".format(
    #     len(clauses), variable_count(clauses),
    #     len(good_clauses), variable_count(good_clauses),
    # ))

    return good_clauses


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
        clauses = []
        variables = range(1, self._variable_count+1)

        for c in range(clause_count):
            sample = random.sample(variables, self._clause_length)
            clause = []
            for v in sample:
                if random.random() < 0.5:
                    clause.apennd(-v)
                else:
                    clause.apennd(v)
            clauses.append(clause)

        cnf, _, _ = cnf_from_clauses(clauses, self.name())

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
                success, _ = minisat.solve(cnf)

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
            variable_count,
            variable_count_max,
            clause_count_max,
    ) -> None:
        self._variable_count = variable_count
        self._variable_count_max = variable_count_max
        self._clause_count_max = clause_count_max
        self._minisat = Minisat()

    def produce_pair(
            self,
    ):
        satisfiable = True
        clauses = []
        variables = range(1, self._variable_count+1)

        pos = 0
        build = True
        step = 12
        while satisfiable:
            if build:
                k = min(
                    2 + np.random.binomial(1, 0.3) + np.random.geometric(0.4),
                    self._variable_count,
                )

                sample = random.sample(variables, k)
                clause = []
                for v in sample:
                    if random.random() < 0.5:
                        clause.append(-v)
                    else:
                        clause.append(v)
                clauses.append(clause)
                pos += 1

                if pos % step == 0:
                    cnf, _, _ = cnf_from_clauses(clauses, self.name())
                    success, _ = self._minisat.solve(cnf)

                    if not success:
                        assert pos >= step
                        pos -= step
                        build = False

            else:
                pos += 1
                cnf, _, _ = cnf_from_clauses(clauses[:pos], self.name())
                success, _ = self._minisat.solve(cnf)

                if not success:
                    clauses = clauses[:pos]
                    satisfiable = False

        unsat_cnf, unsat_variable_count, unsat_clause_count = cnf_from_clauses(
            convert_to_8cnf(clauses),
            self.name(),
        )
        flip = random.randint(0, len(clauses[-1])-1)
        clauses[-1][flip] = -clauses[-1][flip]
        sat_cnf, sat_variable_count, sat_clause_count = cnf_from_clauses(
            convert_to_8cnf(clauses),
            self.name(),
        )

        # success, _ = self._minisat.solve(sat_cnf)
        # assert success is True
        # success, _ = self._minisat.solve(unsat_cnf)
        # assert success is False

        if unsat_variable_count > self._variable_count_max or \
                unsat_clause_count > self._clause_count_max:
            print("SKIPPING {} {}".format(
                unsat_variable_count,
                unsat_clause_count,
            ))
            unsat_cnf = None
        if sat_variable_count > self._variable_count_max or \
                sat_clause_count > self._clause_count_max:
            print("SKIPPING {} {}".format(
                sat_variable_count,
                sat_clause_count,
            ))
            sat_cnf = None

        return (unsat_cnf, sat_cnf)

    def generate(
            self,
            dataset_dir,
            prefix,
            sample_count,
            num_workers=8,
    ) -> None:
        chunk = int(sample_count/num_workers)

        def process(worker):
            generated = 0

            while generated < chunk:
                unsat_cnf, sat_cnf = self.produce_pair()

                if unsat_cnf is None or sat_cnf is None:
                    continue
                # print(sat_cnf.split('\n')[1])

                unsat_cnf = "c UNSAT\n" + unsat_cnf
                sat_cnf = "c SAT\n" + sat_cnf

                generated += 1
                with open(os.path.join(
                        dataset_dir, "{}_{}.cnf".format(
                            prefix, worker*chunk + generated,
                        )
                ), 'w') as f:
                    f.write(unsat_cnf)
                    f.flush()
                generated += 1
                with open(os.path.join(
                        dataset_dir, "{}_{}.cnf".format(
                            prefix, worker*chunk + generated,
                        )
                ), 'w') as f:
                    f.write(sat_cnf)
                    f.flush()

                if generated % 20 == 0:
                    Log.out(
                        "Generating samples", {
                            'generator': self.name(),
                            'total': generated,
                            'worker': worker,
                            'generated': generated,
                        })

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
        ) as executor:
            executor.map(process, range(num_workers))

    def name(
            self,
    ) -> str:
        return "selsam_{}".format(self._variable_count)


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
        fix_args = [
            ([3, 4, 19], 100_000),
            ([3, 4, 20], 100_000),
            ([3, 4, 21], 100_000),

            ([3, 8, 38], 100_000),
            ([3, 8, 39], 100_000),
            ([3, 8, 40], 100_000),
            ([3, 8, 41], 100_000),
            ([3, 8, 42], 100_000),
        ]
        ext_args = [
            ([3, 4, 10, 30], 100_000),
            ([3, 8, 20, 60], 100_000),
        ]

        plan = []

        for args, samples in fix_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args, samples in ext_args:
            if test:
                samples = int(samples / 1000)
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
        samples = 100_000
        if test:
            samples = 100

        fix_args = [
            ([3, 4, 19], 100_000),
            ([3, 4, 20], 100_000),
            ([3, 4, 21], 100_000),

            ([3, 8, 38], 100_000),
            ([3, 8, 39], 100_000),
            ([3, 8, 40], 100_000),
            ([3, 8, 41], 100_000),
            ([3, 8, 42], 100_000),

            ([3, 16, 70], 100_000),
            ([3, 16, 71], 100_000),
            ([3, 16, 72], 100_000),
            ([3, 16, 73], 100_000),
            ([3, 16, 74], 100_000),
            ([3, 16, 75], 100_000),
            ([3, 16, 76], 100_000),
            ([3, 16, 77], 100_000),
            ([3, 16, 78], 100_000),
        ]
        ext_args = [
            ([3, 4, 10, 30], 200_000),
            ([3, 8, 20, 60], 200_000),
            ([3, 16, 55, 110], 200_000),
        ]

        plan = []

        for args, samples in fix_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args, samples in ext_args:
            if test:
                samples = int(samples / 1000)
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
        ext_args = [
            [3, 4, 10, 30],
            [3, 8, 20, 60],
            [3, 16, 55, 110],
            [3, 32, 106, 213],
        ]
        fix_args = [
            ([3, 4, 19], 100_000),
            ([3, 4, 20], 100_000),
            ([3, 4, 21], 100_000),

            ([3, 8, 38], 100_000),
            ([3, 8, 39], 100_000),
            ([3, 8, 40], 100_000),
            ([3, 8, 41], 100_000),
            ([3, 8, 42], 100_000),

            ([3, 16, 70], 100_000),
            ([3, 16, 71], 100_000),
            ([3, 16, 72], 100_000),
            ([3, 16, 73], 100_000),
            ([3, 16, 74], 100_000),
            ([3, 16, 75], 100_000),
            ([3, 16, 76], 100_000),
            ([3, 16, 77], 100_000),
            ([3, 16, 78], 100_000),

            ([3, 32, 134], 100_000),
            ([3, 32, 135], 100_000),
            ([3, 32, 136], 100_000),
            ([3, 32, 137], 100_000),
            ([3, 32, 138], 100_000),
            ([3, 32, 139], 100_000),
            ([3, 32, 140], 100_000),
            ([3, 32, 141], 100_000),
            ([3, 32, 142], 100_000),
            ([3, 32, 143], 100_000),
            ([3, 32, 144], 100_000),
            ([3, 32, 145], 100_000),
            ([3, 32, 146], 100_000),
            ([3, 32, 147], 100_000),
            ([3, 32, 148], 100_000),
            ([3, 32, 149], 100_000),
            ([3, 32, 150], 100_000),
        ]
        ext_args = [
            ([3, 4, 10, 30], 300_000),
            ([3, 8, 20, 60], 300_000),
            ([3, 16, 55, 110], 300_000),
            ([3, 32, 106, 213], 300_000),
        ]

        plan = []

        for args, samples in fix_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': FixRandKGenerator(*args),
                'prefix': "fix_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })
        for args, samples in ext_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': ExtRandKGenerator(*args),
                'prefix': "ext_rand_k_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedRandK32Generator, self).__init__(plan)


class MixedSelsam8Generator(BaseMixedGenerator):
    def __init__(
            self,
            test,
    ) -> None:
        selsam_args = [
            ([4, 4, 256], 50_000),
            ([5, 5, 256], 50_000),
            ([6, 6, 256], 50_000),
            ([7, 7, 256], 100_000),
            ([8, 8, 256], 200_000),
        ]

        plan = []

        for args, samples in selsam_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': SelsamGenerator(*args),
                'prefix': "selsam_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedSelsam8Generator, self).__init__(plan)


class MixedSelsam16Generator(BaseMixedGenerator):
    def __init__(
            self,
            test,
    ) -> None:
        selsam_args = [
            ([4, 4, 288], 10_000),
            ([5, 5, 288], 10_000),
            ([6, 6, 288], 10_000),
            ([7, 7, 288], 10_000),
            ([8, 8, 288], 10_000),
            ([8, 8, 288], 50_000),
            ([9, 80, 288], 50_000),
            ([10, 80, 288], 50_000),
            ([11, 80, 288], 50_000),
            ([12, 80, 288], 100_000),
            ([13, 80, 288], 100_000),
            ([14, 80, 288], 100_000),
            ([15, 80, 288], 100_000),
            ([16, 80, 288], 100_000),
        ]

        plan = []

        for args, samples in selsam_args:
            if test:
                samples = int(samples / 1000)
            plan.append({
                'generator': SelsamGenerator(*args),
                'prefix': "selsam_{}".format('_'.join(map(str, args))),
                'sample_count': samples,
                'num_workers': 4,
            })

        super(MixedSelsam16Generator, self).__init__(plan)


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
    parser.add_argument(
        '--test',
        type=str2bool, help="generate a test set",
    )

    args = parser.parse_args()

    dataset_dir = os.path.expanduser(args.dataset_dir)
    generator = None
    if args.test is None:
        args.test = False

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

            # 3, 500, 3550,

            3, 8, 40,
        )
        generator.generate(dataset_dir, 'fix_rand_k_plus_3_8_40', 200_000, 8)
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
        generator.generate(dataset_dir, '', 1000, 1)

    if args.generator == "selsam_4":
        generator = SelsamGenerator(4, 4, 64)
        generator.generate(dataset_dir, 'selsam_4', 1000, 4)
    if args.generator == "selsam_8":
        generator = SelsamGenerator(8, 8, 256)
        generator.generate(dataset_dir, 'selsam_8', 1000, 8)
    if args.generator == "selsam_16":
        generator = SelsamGenerator(16, 80, 288)
        generator.generate(dataset_dir, 'selsam_16', 1000, 1)
    if args.generator == "selsam_40":
        generator = SelsamGenerator(40, 150, 600)
        generator.generate(dataset_dir, 'selsam_40', 1000, 1)
    if args.generator == "selsam_64":
        generator = SelsamGenerator(64, 250, 900)
        generator.generate(dataset_dir, 'selsam_64', 1000, 1)

    if args.generator == "mixed_rand_k_8":
        generator = MixedRandK8Generator(args.test)
        generator.generate(dataset_dir, 2)
    if args.generator == "mixed_rand_k_16":
        generator = MixedRandK16Generator(args.test)
        generator.generate(dataset_dir, 2)
    if args.generator == "mixed_rand_k_32":
        generator = MixedRandK32Generator(args.test)
        generator.generate(dataset_dir, 2)

    if args.generator == "mixed_selsam_8":
        generator = MixedSelsam8Generator(args.test)
        generator.generate(dataset_dir, 2)
    if args.generator == "mixed_selsam_16":
        generator = MixedSelsam16Generator(args.test)
        generator.generate(dataset_dir, 2)

    assert generator is not None
