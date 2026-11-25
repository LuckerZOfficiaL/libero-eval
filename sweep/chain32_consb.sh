#!/bin/bash
# 11-06 EXP-147 consb (does the merge matter?): a plain LoRA from the BASE on the same data (own rollouts + 5 demos per library
# skill), 3,000 steps, rank 16, deployed alone. consf (plain LoRA inside the converted merge) = 91.0 / 91.3 ~ cons -> write
# protection is not what makes the correction work; if consb falls short, the merge is the warm start that does. GPU0 (consfz
# stopped at 800/3000, requeued last): consb -> consz orders (claims shared with chain31) -> consfz. Eval port 8900. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c32] start $(date +%m-%d\ %H:%M)"
( CUDA_VISIBLE_DEVICES=$U0 E145_SEQ=0 E147_VARS=consb E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  for s in 1 2 3; do CUDA_VISIBLE_DEVICES=$U0 E145_SEQ=$s E147_VARS=consz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg | sed "s/^/[s$s] /"; done
  CUDA_VISIBLE_DEVICES=$U0 E145_SEQ=0 E147_VARS=consfz E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg ) &
( export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8900 E145_SEQ=0
  until [ -f $A/new_loras_cons/train_consb/adapter_model.safetensors ]; do sleep 120; done; sleep 30
  for try in 1 2 3; do E147_VARS=consb E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
  E147_VARS=cons,consl,conso,consz,consf,consb E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  echo "[c32] consb EVAL EXIT $(date +%m-%d\ %H:%M)" ) &
wait
echo "[c32] ALL EXIT $(date +%m-%d\ %H:%M)"
