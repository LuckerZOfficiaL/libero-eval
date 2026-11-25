#!/bin/bash
# 11-22 EXP-170 (user-approved) the 32-skill library: arrival 1 (E145_SEQ 5 = 32 library + order 0's 8 new) ER (erl6) and ER
# from the base (consb6) next to ours (consfz6, 11-09), 6,000 steps; then the stream 30 / 31 / 32 (14 / 18 / 22 new, SCALE's
# order) ours (consfzl) vs ER (erll), 3,000 resumed steps per arrival; joint training on all 54 skills' demos (joint24k_s32,
# rank 64, 24,000 steps). Starts once every chain76 trainer is claimed; lanes wait 180 s, the joint lane 60 s (takes the first
# free A6000 once both arrival-1 runs are claimed). Evals port 10900 (joint 10950) under the eval lock. GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
C76JOBS=""; for a in 0 1 2; do for o in 21 24 27; do for v in consfzc erlc; do C76JOBS="$C76JOBS $v@$((o + a))"; done; done; done
JOBS="erl6@5 consb6@5 consfzl@30 erll@30 consfzl@31 erll@31 consfzl@32 erll@32"
JOUT=$A/joint_loras/joint24k_s32/adapter_model.safetensors
echo "[c78] start $(date +%m-%d\ %H:%M): $JOBS + joint24k_s32"
c76left() { local j; for j in $C76JOBS; do [ -f $A/.c76_${j/@/_} ] || return 0; done; return 1; }
while c76left; do sleep 300; done
echo "[c78] chain76 trainers all claimed $(date +%m-%d\ %H:%M), lanes start"
jlane() {
  until [ -f $A/.c78_erl6_5 ] && [ -f $A/.c78_consb6_5 ]; do sleep 60; done
  while true; do
    for g in $G_U0 $G_U2; do
      busy2 $g && continue; sleep 60; busy2 $g && continue
      ( set -o noclobber; : > $A/.c78_joint_32 ) 2>/dev/null || return
      reserve $g
      echo "[c78] trainer joint24k_s32 on $g $(date +%H:%M)"
      E136_SFX=24k E136_STEPS=24000 E136_ORDERS=32 E136_GPU=$g E136_STAGE=train .venv/bin/python -u sweep/exp136_joint.py 2>&1 | grep -v hostcfg
      return
    done
    sleep 60
  done
}
jlane &
sleep 20; lane2 $G_U0 c78 180 "$JOBS" &
sleep 40; lane2 $G_U2 c78 180 "$JOBS" &
left="$JOBS joint@32"
while [ -n "$left" ]; do
  for j in $left; do
    if [ $j = joint@32 ]; then
      [ -f $JOUT ] || continue
      sleep 60
      ( flock 9; export ORCH_CUDA_MAP="0=$G_U0,2=$G_U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$G_U0 E113_DEV=cuda:0
        for try in 1 2; do E112_PORT0=10950 E136_SFX=24k E136_ORDERS=32 E136_STAGE=eval .venv/bin/python -u sweep/exp136_joint.py 9>&- 2>&1 | grep -v hostcfg 9>&-; done ) 9> $A/.evlock
      echo "[c78] joint24k_s32 EVAL EXIT $(date +%m-%d\ %H:%M)"
    else
      [ -f $(out $j) ] || continue
      sleep 60; evalrun $j 10900 c78 2
    fi
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c78] ALL EXIT $(date +%m-%d\ %H:%M)"
