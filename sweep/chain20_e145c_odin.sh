#!/bin/bash
# 11-04 (shiva GPU1 off the bus again): EXP-145c on odin's P6000s — the no-replay FedAvg blend conv + 1/8 (ctx - conv)
cd /home/lzhou/libero-eval; set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID E113_DEV=cuda:2 E113_GPUS=1,2 E112_PER_GPU=2 E112_PORT0=8300 E145_BLEND=ctx E145_LAMS=0.125
echo "[c20] start $(date +%m-%d\ %H:%M)"
E145_VARS=ctx E145_STAGE=interp .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
for try in 1 2; do E145_VARS=ctx_l0.125 E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg; done
echo "[c20] E145C ALL EXIT $(date +%m-%d\ %H:%M)"
