#!/bin/bash
# 10-25 EXP-141 order 0, after its main run on GPU0: library only + each external ALONE on top of the library (k = 17)
# at weight 0.5 / 1, plain and soft-gated (is the clash library-vs-external or the 8 externals piling up?).
cd /home/lzhou/libero-eval; set -u
until grep -q "run DONE" results/e141_s0.log; do sleep 60; done
A=lib; for i in 0 1 2 3 4 5 6 7; do A=$A,x$i-plain_w1,x$i-soft_w1,x$i-plain_w0.5; done
CUDA_VISIBLE_DEVICES=GPU-8702ed42-9555-9082-d07a-3542fced5400 E112_DEV=cuda:0 E141_SEQ=0 E141_ARMS=$A .venv/bin/python -u sweep/exp141_actproxy.py
echo "E141X EXIT $(date '+%m-%d %H:%M')"
