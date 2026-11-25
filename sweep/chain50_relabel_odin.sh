#!/bin/bash
# 11-07 EXP-155 relabeling on odin's P6000 (idle; trainings of plain LoRAs don't fit it), orders 2, 0, 1, 3 in turn.
# Output = NAS dumps (S.DUMPS/e155_*), read by chain51's trainings on shiva.
cd /home/lzhou/libero-eval; set -u
echo "[c50] start $(date +%m-%d\ %H:%M)"
for s in 2 0 1 3; do
  for try in 1 2; do CUDA_VISIBLE_DEVICES=GPU-653b4880-3c1a-c9b6-7bcf-6cb199869263 E145_SEQ=$s E155_STAGE=relabel .venv/bin/python -u sweep/exp155_dagger.py 2>&1 | grep -v hostcfg; done
  echo "[c50] RELABELED s$s $(date +%m-%d\ %H:%M)"
done
echo "[c50] ALL EXIT $(date +%m-%d\ %H:%M)"
