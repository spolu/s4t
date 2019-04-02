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
            # 'sat_generate_dataset=sat.generator:generate',
            # 'sat_train_solver=sat.solver.solver:train',
            # 'sat_test_solver=sat.solver.solver:test',
            # 'holstep_preprocess=dataset.holstep:preprocess',
            # 'th2vec_train_direct_premiser=th2vec.direct_premiser:train',
            # 'th2vec_test_direct_premiser=th2vec.direct_premiser:test',
            # 'th2vec_train_premise_embedder=th2vec.premise_embedder:train',
            # 'th2vec_train_autoencoder_embedder=th2vec.autoencoder_embedder:train',
            # 'th2vec_train_premiser=th2vec.premiser:train',
            # 'th2vec_train_generator=th2vec.generator:train',

            'run_surgery_20190401_ppo_iota_separate_hiddens=surgeries.20190401_ppo_iota_separate_hiddens:run',
            'verify_surgery_20190401_ppo_iota_separate_hiddens=surgeries.20190401_ppo_iota_separate_hiddens:verify',

            'prooftrace_extract=dataset.prooftrace:extract',
            'prooftrace_dump_shared=dataset.prooftrace:dump_shared',
            'generic_test_tree_lstm=generic.tree_lstm:test',
            'prooftrace_test_embedder=prooftrace.models.embedder:test',
            'prooftrace_train_language_model=prooftrace.language_model:train',
            'prooftrace_test_language_model=prooftrace.language_model:test',
            'prooftrace_train_value=prooftrace.value:train',
            'prooftrace_test_value=prooftrace.value:test',
            'prooftrace_test_repl=prooftrace.repl.repl:test',
            'prooftrace_test_fusion=prooftrace.repl.fusion:test',
            'prooftrace_test_repl_env=prooftrace.repl.env:test',
            'prooftrace_train_ppo=prooftrace.ppo:train',
            'prooftrace_ppo_syn_run=prooftrace.ppo_iota:syn_run',
            'prooftrace_ppo_ack_run=prooftrace.ppo_iota:ack_run',
            'prooftrace_search_language_model=prooftrace.language_model_search:search',
            'prooftrace_language_model_syn_run=prooftrace.language_model_iota:syn_run',
            'prooftrace_language_model_ack_run=prooftrace.language_model_iota:ack_run',
            'generic_test_tsne=generic.tsne:test',
            'prooftrace_tsne_embed_targets=prooftrace.tsne:embed_targets',
        ],
    },
)
