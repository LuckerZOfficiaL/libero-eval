#!/bin/bash
# 11-18 erl@0 eval: 4 of 24 tasks failed in chain67 (worker on GPU0 "server died during load" = OOM: trainer + 3 eval workers + a
# foreign IsaacLab job). Re-runs the missing jobs (exp147 run skips completed ones), 1 worker per GPU, port 10360; stops chain69
# (race-prone gap evals) 90 s in, once this run holds GPU0 (chain67 waits while >= 2 eval processes are on GPU0).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c67fix] start $(date +%m-%d\ %H:%M)"
( sleep 90; tmux kill-session -t c69 && echo "[c67fix] chain69 stopped $(date +%H:%M)" ) &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0
for try in 1 2 3; do E112_PORT0=10360 E145_SEQ=0 E147_VARS=erl E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
wait; echo "[c67fix] erl@0 EVAL EXIT $(date +%m-%d\ %H:%M)"
