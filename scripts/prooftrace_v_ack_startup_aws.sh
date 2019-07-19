#!/bin/bash

sshfs -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no stan@ec2-13-58-204-144.us-east-2.compute.amazonaws.com:/home/stan/tmp/iota ~/tmp/iota

cd ~/src/z3ta
source ~/opt/miniconda3/bin/activate z3ta

git pull origin master
python setup.py develop

tmux -S ~/v_ack new-session -d -s v_ack

/home/stan/src/z3ta/scripts/v_ack++ 1 0
/home/stan/src/z3ta/scripts/v_ack++ 1 1
/home/stan/src/z3ta/scripts/v_ack++ 1 2
/home/stan/src/z3ta/scripts/v_ack++ 1 3
/home/stan/src/z3ta/scripts/v_ack++ 1 4
/home/stan/src/z3ta/scripts/v_ack++ 1 5
/home/stan/src/z3ta/scripts/v_ack++ 1 6
/home/stan/src/z3ta/scripts/v_ack++ 1 7
