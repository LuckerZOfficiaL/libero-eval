#!/bin/bash
# 11-05 EXP-145 generality on shiva: in-context ctx trainings for orders 1-3 (shared NAS claim queue with odin's chain24),
# ctx first (needed by every candidate arm), ctxr after (may be replaced by one replay-only correction if EXP-147 consl works).
# GPU0 starts once EXP-147 cons is trained (its eval runs alongside), GPU2 once consl is trained. GPU1 left idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c26] start $(date +%m-%d\ %H:%M)"
trainer() { for v in ctx ctxr; do for s in 1 2 3; do
  [ -f results/analysis/stop_c26_$v ] && continue
  CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E145_VARS=$v E145_LAMS= E145_STAGE=train .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
done; done; }
( until [ -f results/analysis/new_loras_cons/train_cons/adapter_model.safetensors ]; do sleep 60; done; trainer $U0 ) &
( until [ -f results/analysis/new_loras_cons/train_consl/adapter_model.safetensors ]; do sleep 60; done; sleep 60; trainer $U2 ) &
wait
echo "[c26] ALL EXIT $(date +%m-%d\ %H:%M)"
