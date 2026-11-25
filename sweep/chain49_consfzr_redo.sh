#!/bin/bash
# 11-07 redo of consfzr orders 2-3 (EXP-147 ablation: no conversion, no demos). The 14:56 / 17:16 runs trained a WP correction
# instead of a plain LoRA (train_lora if/elif bug from the EXP-149 resume edit, fixed 20:15); artifacts in
# results/merge_stale/bug1107_consfzr. Lowest priority: each A6000 takes r2 / r3 once chain48 has claimed all 4 of its
# trainings. Eval: s2 here (port 9350); s3 by chain43's evaluator (still waiting for train_consfzr s3). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis
echo "[c49] start $(date +%m-%d\ %H:%M)"
big() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2>=12000' | wc -l; }
busy() { [ $(big $1) -gt 0 ] && return 0; for p in $(pgrep -f sweep/train_lora.py); do tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && return 0; done; return 1; }
nev() { nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v g=$1 '$1==g && $2<12000' | wc -l; }
lane() {   # $1 GPU
  until [ -f $A/.c48_f6 ] && [ -f $A/.c48_j6 ] && [ -f $A/.c48_f5 ] && [ -f $A/.c48_j5 ]; do sleep 120; done
  for s in 2 3; do
    [ -f $A/.c49_r$s ] && continue
    until ! busy $1 && sleep 180 && ! busy $1; do sleep 120; done
    ( set -o noclobber; : > $A/.c49_r$s ) 2>/dev/null || continue
    echo "[c49] trainer r$s on $1 $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$1 E145_SEQ=$s E147_VARS=consfzr E147_STAGE=train .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
  done
}
lane $U2 &
lane $U0 &
export ORCH_CUDA_MAP="0=$U0,2=$U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=1 CUDA_VISIBLE_DEVICES=$U0 E113_DEV=cuda:0 E145_SEQ=2
until [ -f $A/new_loras_cons_s2/train_consfzr/adapter_model.safetensors ] && [ $(nev $U0) -le 2 ] && [ $(nev $U2) -le 2 ]; do sleep 120; done
sleep 60   # adapter fully written
for try in 1 2 3; do E112_PORT0=9350 E147_VARS=consfzr E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg; done
E147_VARS=cons,consr,consz,consfz,consfzr E147_STAGE=report .venv/bin/python -u sweep/exp147_consolidate.py 2>&1 | grep -v hostcfg
echo "[c49] ORDER 2 EVAL EXIT $(date +%m-%d\ %H:%M)"
wait; echo "[c49] ALL EXIT $(date +%m-%d\ %H:%M)"
