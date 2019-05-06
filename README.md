## PROOFTRACE PPO/IOTA Distributed Training

```
mkdir -p ~/tmp/prooftrace/`git rev-parse HEAD` && prooftrace_ppo_syn_run configs/prooftrace_ppo.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/prooftrace/`git rev-parse HEAD` --device=cuda:0 --sync_dir=/mnt/iota/ppo
```

```
prooftrace_ppo_ack_run configs/prooftrace_ppo.json --sync_dir=~/tmp/iota/ --device=cuda:0
```

```
[20190505_0326_56.550245] ================================================
[20190505_0326_56.550245]  ProofTraces Length
[20190505_0326_56.550245] ------------------------------------------------
[20190505_0326_56.550245]      <0064 ***                              7606
[20190505_0326_56.550245]  0064-0128 *****                            13243
[20190505_0326_56.550245]  0128-0256 ********                         20294
[20190505_0326_56.550245]  0256-0512 **********                       24583
[20190505_0326_56.550245]  0512-1024 ****                             11477
[20190505_0326_56.550245]  1024-2048                                  261
[20190505_0326_56.550245]  2048-4096                                  0
[20190505_0326_56.550245]      >4096                                  0
[20190505_0326_56.550245] ------------------------------------------------
```

