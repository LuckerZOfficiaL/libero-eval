#!/bin/bash
# 10-22 GPU 2 tail: when its chain3 shares finish, join the restore units.
cd /home/lzhou/libero-eval; set -u
until grep -q "\[g2\] WP SHARD EXIT" results/chain3_gpu2.log 2>/dev/null; do sleep 300; done
export CUDA_VISIBLE_DEVICES=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba ORCH_CUDA_MAP="2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba" ORCH_EGL_MAP="2=0" E113_GPUS=2 E113_DEV=cuda:0 TAG=g2
E113_ROUND=2 bash sweep/restore_units.sh
