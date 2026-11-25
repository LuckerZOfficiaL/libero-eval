#!/bin/bash
# 11-05 rerun of the 36 scene records quarantined after CUDA-OOM corruption (server errors counted as failed episodes when
# evals shared a GPU with training; results/merge_quarantine/oom_1105). Patched orchestrator (crashed episodes -> status
# episode_error, per-port eval logs). Starts after EXP-147's cons eval (chain25); 1 worker per GPU on 0+2 next to the
# trainers (chain22's consl eval adds 2 per GPU later; <= 3 workers + 1 trainer per A6000). GPU1 left idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
echo "[c27] start $(date +%m-%d\ %H:%M)"
until grep -q "E147 CONS EXIT" results/chain25.log || ! tmux has-session -t c25 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_PER_GPU=1 E112_PORT0=8500
for st in base run; do for try in 1 2 3; do E144_SEQS=1,2,3 E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg; done; done
E144_SEQS=0,1,2,3 E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c27] E144 RERUN EXIT $(date +%m-%d\ %H:%M)"
export E144_SEQS=0; M="0.125:0.125,0.25:0.125"
for try in 1 2 3; do
  E145_VARS=ctx,ctxr E145_LAMS= E145_MIX=$M E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
  E145_VARS=ctx,ctxr E145_BLEND=ctx E145_LAMS=0.25 E145_STAGE=run .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
done
E145_VARS=ctxr,ctx,ctxr_l0.125,ctxr_l0.25,ctx_l0.125,ctx_l0.25 E145_LAMS= E145_MIX=$M E145_STAGE=report .venv/bin/python -u sweep/exp145_incontext.py 2>&1 | grep -v hostcfg
echo "[c27] ALL EXIT $(date +%m-%d\ %H:%M)"
