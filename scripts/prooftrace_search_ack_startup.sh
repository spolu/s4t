#!/bin/bash

cd /home/stan/src/z3ta
sudo -u stan git pull origin master
sudo -u stan /home/stan/src/z3ta/scripts/prooftrace_search_ack_startup_stan.sh

