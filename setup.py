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
            'holstep_preprocess=dataset.holstep:preprocess',
            'th2vec_train_direct_premiser=th2vec.direct_premiser:train',
            'th2vec_test_direct_premiser=th2vec.direct_premiser:test',
            'th2vec_train_premise_embedder=th2vec.premise_embedder:train',
            'th2vec_train_autoencoder_embedder=th2vec.autoencoder_embedder:train',
            'th2vec_train_premiser=th2vec.premiser:train',
            'th2vec_train_generator=th2vec.generator:train',
            'prooftrace_extract=dataset.prooftrace:extract',
            'generic_test_tree_lstm=generic.tree_lstm:test',
            'formal_test_embedder=formal.models.embedder:test',
        ],
    },
)
