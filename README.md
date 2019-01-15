## Prerequesite

`minisat` must be present in the `$PATH`

## Training commands

```
mkdir ~/tmp/`git rev-parse HEAD` && train_solver configs/dev.json --tensorboard_log_dir=~/tmp/tensorboard/`git rev-parse HEAD`_`now` --solver_save_dir=~/tmp/s4t/`git rev-parse HEAD` --solver_load_dir=~/tmp/s4t/`git rev-parse HEAD` --device=cuda:0
```

