from setuptools import setup

setup(
    name='s4t',
    version='0.0.2',
    install_requires=[
    ],
    packages=[
        'utils',
    ],
    package_data={
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sat_generate_dataset=sat.generator:generate',
            'sat_train_solver=sat.solver.solver:train',
            'sat_test_solver=sat.solver.solver:test',
        ],
    },
)
