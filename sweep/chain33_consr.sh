#!/bin/bash
# 11-06 EXP-147 consr (no demos at all): cons with the library replay drawn from the LIBRARY skills' own successful rollouts
# instead of 5 demos each (order 0). Collection now on the idle eval slots (1 worker per GPU on 0+2, port 9000; <= 3 workers +
# 1 trainer per A6000); training on GPU2 once chain31 has no consz order left for it (s2 + s3 claimed, GPU2 idle twice 3 min
# apart); eval port 9100. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c33] start $(date +%m-%d\ %H:%M)"
busy() { for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
export E145_SEQ=0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1
CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9000 E147_VARS=consr E147_STAGE=collect_lib .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c33] COLLECT EXIT $(date +%m-%d\ %H:%M)"
until [ -f $A/new_loras_cons_s2/.claim_consz ] && [ -f $A/new_loras_cons_s3/.claim_consz ] && ! busy $U2 && sleep 180 && ! busy $U2; do sleep 120; done
echo "[c33] trainer on $U2 $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=$U2 E147_VARS=consr E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E112_PORT0=9100
E147_VARS=consr E147_STAGE=interp .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
for try in 1 2 3; do E147_VARS=consr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=cons,consl,conso,consz,consf,consr E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c33] consr EVAL EXIT $(date +%m-%d\ %H:%M)"
