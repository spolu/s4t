## PROOFTRACE PPO/IOTA Distributed Training

```
mkdir -p ~/tmp/prooftrace/`git rev-parse HEAD` && prooftrace_ppo_syn_run configs/prooftrace_ppo.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/prooftrace/`git rev-parse HEAD` --device=cuda:0 --sync_dir=/mnt/iota/ppo
```

```
prooftrace_ppo_ack_run configs/prooftrace_ppo.json --sync_dir=~/tmp/iota/ --device=cuda:0
```

```
[20190308_1145_05.347773] ================================================
[20190308_1145_05.347773]  ProofTraces Length
[20190308_1145_05.347773] ------------------------------------------------
[20190308_1145_05.347773]      <0064 ****                             481
[20190308_1145_05.347773]  0064-0128 ***                              443
[20190308_1145_05.347773]  0128-0256 ***                              448
[20190308_1145_05.347773]  0256-0512 ****                             462
[20190308_1145_05.347773]  0512-1024 ***                              382
[20190308_1145_05.347773]  1024-2048 ****                             508
[20190308_1145_05.347773]  2048-4096 ***                              413
[20190308_1145_05.347773]      >4096 ****                             486
[20190308_1145_05.347773] ------------------------------------------------
```
