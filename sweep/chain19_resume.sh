#!/bin/bash
# 11-04 shiva back (rebooted ~10-29, chain15 killed with xb0e1f6 untrained). Whole queue:
# 1) EXP-144: train the last converted adapter (GPU 1)  ||  EXP-145b FedAvg blends (chain17: GPUs 0+2, ports 8100+)
# 2) EXP-144 base + merges of orders 1-3 (all GPUs) + report
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
export E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2"
echo "[c19] start $(date '+%m-%d %H:%M')"
( CUDA_VISIBLE_DEVICES=$U1 E144_SEQS=0,1,2,3 E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py
  echo "[c19] E144 TRAIN EXIT $(date '+%m-%d %H:%M')" ) &
bash sweep/chain17_e145b.sh
wait
export CUDA_VISIBLE_DEVICES=$U0 E112_PER_GPU=2 E113_GPUS=0,1,2 E112_PORT0=8000
for st in base run; do for try in 1 2; do E144_SEQS=1,2,3 E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_SEQS=0,1,2,3 E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c19] E144 ALL EXIT $(date '+%m-%d %H:%M')"
# EXP-146 (CUSD) parked 11-04 by the user: preprint released today, not a mandatory baseline yet
echo "[c19] ALL EXIT $(date '+%m-%d %H:%M')"
