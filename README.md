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
