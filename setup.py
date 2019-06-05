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
            'prooftrace_load_all=prooftrace.prooftrace:load_all',
            'prooftrace_dump=prooftrace.tools:dump',
            'prooftrace_generate_testset=prooftrace.tools:generate_testset',

            'generic_test_tree_lstm=generic.tree_lstm:test',

            'prooftrace_test_embedder=prooftrace.models.embedder:test',
            'prooftrace_test_repl=prooftrace.repl.repl:test',
            'prooftrace_test_fusion=prooftrace.repl.fusion:test',
            'prooftrace_test_repl_env=prooftrace.repl.env:test',

            'prooftrace_ppo_syn_run=prooftrace.ppo_iota:syn_run',
            'prooftrace_ppo_ack_run=prooftrace.ppo_iota:ack_run',

            'prooftrace_search=prooftrace.search:search',
            'prooftrace_search_rollout_bootstrap=prooftrace.search_rollout:bootstrap',
            'prooftrace_search_rollout_inspect=prooftrace.search_rollout:inspect',
            'prooftrace_search_syn_run=prooftrace.search_iota:syn_run',
            'prooftrace_search_ack_run=prooftrace.search_iota:ack_run',
            'prooftrace_search_rollout_ctl_run=prooftrace.search_rollout:ctl_run',
            'prooftrace_search_rollout_wrk_run=prooftrace.search_rollout:wrk_run',
            'prooftrace_search_search_ctl_run=prooftrace.search_test:ctl_run',
            'prooftrace_search_search_wrk_run=prooftrace.search_test:wrk_run',

            'prooftrace_embeds_tsne_extract=prooftrace.embeds.tsne:extract',
            'prooftrace_embeds_viewer=prooftrace.embeds.viewer.viewer:run',
        ],
    },
)
