#!/bin/bash
# 10-26 16:30 EXP-144 extension: collect + convert the 14 externals of orders 1-3 not in order 0; meanwhile order 0's new-skill
# weight sweep (0.125, 0.5; the pilot used 0.25); then base + merge for orders 1-3 (w 0.25) + reports. All 3 GPUs, EGL.
# 17:40 relaunched after the quota stop (dumps now on the NAS archive; failed partial collections set aside by collect())
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$U0 ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2" \
       E113_GPUS=0,1,2 E112_PORT0=8000 E144_SEQS=0,1,2,3
echo "[c15] collect start $(date '+%m-%d %H:%M')"
for try in 1 2; do E144_STAGE=collect .venv/bin/python -u sweep/exp144_selfconvert.py; done
touch results/analysis/new_loras_s16wp_self/.collect_done
echo "[c15] w sweep (order 0) start $(date '+%m-%d %H:%M')"
for try in 1 2; do E144_SEQS=0 E144_WS=0.125,0.5 E144_STAGE=run .venv/bin/python -u sweep/exp144_selfconvert.py; done
E144_SEQS=0 E144_WS=0.125,0.25,0.5 E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c15] E144 WSWEEP EXIT $(date '+%m-%d %H:%M')"
echo "[c15] train start $(date '+%m-%d %H:%M')"
for u in $U0 $U1 $U2; do CUDA_VISIBLE_DEVICES=$u E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py & sleep 30; done; wait
echo "[c15] E144 TRAIN EXIT $(date '+%m-%d %H:%M')"
for st in base run; do for try in 1 2; do E144_SEQS=1,2,3 E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c15] E144 ALL EXIT $(date '+%m-%d %H:%M')"
