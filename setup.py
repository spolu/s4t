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
            'prooftrace_dump_shared=prooftrace.prooftrace:dump_shared',

            'generic_test_tree_lstm=generic.tree_lstm:test',

            'prooftrace_test_embedder=prooftrace.models.embedder:test',
            'prooftrace_test_repl=prooftrace.repl.repl:test',
            'prooftrace_test_fusion=prooftrace.repl.fusion:test',
            'prooftrace_test_repl_env=prooftrace.repl.env:test',

            'prooftrace_ppo_syn_run=prooftrace.ppo_iota:syn_run',
            'prooftrace_ppo_ack_run=prooftrace.ppo_iota:ack_run',
            'prooftrace_embeds_tsne_extract=prooftrace.embeds.tsne:extract',
            'prooftrace_embeds_viewer=prooftrace.embeds.viewer.viewer:run',
        ],
    },
)
