from setuptools import setup

setup(
    name='s4t',
    version='0.0.1',
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
            'generate_dataset=dataset.generator:generate',
            'train_solver=solver.solver:train',
            'test_solver=solver.solver:test',
        ],
    },
)
