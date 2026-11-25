#!/bin/bash
# 11-18 v2 of chain69_parts.sh (stopped 21:55: its gap evals raced chain67's, GPU0 OOM). EXP-161 VLA part ablation, order 0,
# eval only, remaining arms ne / nl / nv (ne's completed tasks are kept). 1 worker per GPU, port 10500, lowest priority: only when
# chain67 / 68b / 70b have no pending eval, no other chain is serving and both A6000s have >= 16 GB free (150 s double check).
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
echo "[c69] start $(date +%m-%d\ %H:%M)"
for v in ne nl nv; do
  gapwait 150 "$C67JOBS $C68JOBS $C70JOBS"; evalrun $v@0 10500 c69
done
echo "[c69] ALL EXIT $(date +%m-%d\ %H:%M)"
