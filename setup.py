from setuptools import setup

setup(
    name='z3ta',
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
            'prooftrace_extract=prooftrace.prooftrace:extract',
            'prooftrace_dump=prooftrace.tools:dump',
            'prooftrace_generate_testset=prooftrace.tools:generate_testset',

            'generic_test_tree_lstm=generic.tree_lstm:test',

            'prooftrace_test_embedder=prooftrace.models.embedder:test',
            'prooftrace_test_repl=prooftrace.repl.repl:test',
            'prooftrace_test_fusion=prooftrace.repl.fusion:test',
            'prooftrace_test_repl_env=prooftrace.repl.env:test',

            'prooftrace_train_lm=prooftrace.language_model:train',
            'prooftrace_lm_syn_run=prooftrace.language_model_iota:syn_run',
            'prooftrace_lm_ack_run=prooftrace.language_model_iota:ack_run',
            'prooftrace_val_syn_run=prooftrace.value_iota:syn_run',
            'prooftrace_val_ack_run=prooftrace.value_iota:ack_run',
            'prooftrace_tree_search=prooftrace.tree_search:mcts',

            'prooftrace_ppo_syn_run=prooftrace.ppo_iota:syn_run',
            'prooftrace_ppo_ack_run=prooftrace.ppo_iota:ack_run',

            'prooftrace_embeds_tsne_extract=prooftrace.embeds.tsne:extract',
            'prooftrace_embeds_viewer=prooftrace.embeds.viewer.viewer:run',
        ],
    },
)
