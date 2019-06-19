#!/bin/bash

sshfs stan@syn:/home/stan/tmp/iota ~/tmp/iota

cd ~/src/z3ta
source /opt/anaconda3/bin/activate z3ta

git pull origin master
python setup.py develop

tmux -S ~/lm_rollout_wrk new-session -d -s lm_rollout_wrk

/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 0
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 1
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 2
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 3

/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 0
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 1
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 2
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 3

/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 0
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 1
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 2
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 3

/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 0
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 1
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 2
/home/stan/src/z3ta/scripts/lm_rollout_wrk++ 1 3
