## Prerequesite

`minisat` must be present in the `$PATH`

## SAT Distributed Training

```
mkdir -p ~/tmp/s4t/`git rev-parse HEAD` && MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 sat_train_solver configs/dev.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/s4t/`git rev-parse HEAD` --load_dir=~/tmp/s4t/`git rev-parse HEAD` --distributed_training=true --distributed_world_size=4 --distributed_rank=0 --device=cuda:0
```

## TH2VEC Distributed Training

```
mkdir -p ~/tmp/th2vec/`git rev-parse HEAD` && MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 th2vec_train_embedder configs/dev.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/th2vec/`git rev-parse HEAD` --load_dir=~/tmp/th2vec/`git rev-parse` --distributed_training=true --distributed_world_size=4 --distributed_rank=0 --device=cuda:0
```

## PROOFTRACE Distributed Training

```
mkdir -p ~/tmp/prooftrace/`git rev-parse HEAD` && MASTER_ADDR=127.0.0.1 MASTER_PORT=9999 prooftrace_train_language_modeler configs/dev.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --save_dir=~/tmp/prooftrace/`git rev-parse HEAD` --load_dir=~/tmp/prooftrace/`git rev-parse` --distributed_training=true --distributed_world_size=4 --distributed_rank=0 --device=cuda:0
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

# TODO REPL

- [ ] optimize command execution with lets
- [ ] prevent large command bug by separating instantiations or maybe only long ones?
