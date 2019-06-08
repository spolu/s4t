#!/bin/bash

sshfs stan@syn:/home/stan/tmp/iota ~/tmp/iota

cd ~/src/z3ta
source /opt/anaconda3/bin/activate z3ta

git pull origin master
python setup.py develop

tmux -S ~/ppo_ack new-session -d -s ppo_ack

/home/stan/src/z3ta/scripts/ppo_ack++ 1 0
/home/stan/src/z3ta/scripts/ppo_ack++ 1 1
/home/stan/src/z3ta/scripts/ppo_ack++ 1 2
/home/stan/src/z3ta/scripts/ppo_ack++ 1 3

/home/stan/src/z3ta/scripts/ppo_ack++ 1 0
/home/stan/src/z3ta/scripts/ppo_ack++ 1 1
/home/stan/src/z3ta/scripts/ppo_ack++ 1 2
/home/stan/src/z3ta/scripts/ppo_ack++ 1 3

/home/stan/src/z3ta/scripts/ppo_ack++ 1 0
/home/stan/src/z3ta/scripts/ppo_ack++ 1 1
/home/stan/src/z3ta/scripts/ppo_ack++ 1 2
/home/stan/src/z3ta/scripts/ppo_ack++ 1 3

/home/stan/src/z3ta/scripts/ppo_ack++ 1 0
/home/stan/src/z3ta/scripts/ppo_ack++ 1 1
/home/stan/src/z3ta/scripts/ppo_ack++ 1 2
/home/stan/src/z3ta/scripts/ppo_ack++ 1 3
