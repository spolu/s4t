#!/bin/sh

su stan

sshfs stan@syn:/home/stan/tmp/iota ~/tmp/iot

cd ~/src/z3ta
source activate z3ta

git pull origin master
python setup.py develop
