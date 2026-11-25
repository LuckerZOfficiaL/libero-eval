#!/bin/bash
# 11-23 EXP-172 (user: "queue all three", after chain79): where does re-merging the arrivals help? (EXP-167: not over ER at 5 demos)
#  a limited replay memory: the stream storing NO rollouts of earlier arrivals (consfzcm0 / erlcm0; just-arrived batch + library
#    replay), orders 0 (19 -> 20 -> 4) and 1 (21 -> 22 -> 23) from consfz / erl, 1,500 resumed steps per arrival (trainers)
#  b OOD, eval only: objects shifted 10 / 5 cm (EXP-81 ood_shift); arrival 1 (consfz / erl @0-3) at both, arrival 4 (consfzc /
#    erlc @4 23 26 29) at 10 cm
#  c unseen tasks, eval only: the arrival-4 models zero-shot on the 8 externals no order draws + 16 seeded LIBERO-90 tasks
# Trainer lanes start once every chain79 trainer is claimed (wait 240 s); evals (port 11100, eval lock) start once chain79 is
# ALL EXIT: a's evals as soon as their adapters exist, b / c fill the gaps (10 cm arrival 1, unseen, 5 cm, 10 cm arrival 4). GPU1 idle.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS E147_OOD E147_UNS
. sweep/gpu_gate.sh
eval "$(grep '^P[123]=' sweep/chain79_consolidate.sh)"; C79JOBS="$P1 $P2 $P3"
JOBS="consfzcm0@19 erlcm0@19 consfzcm0@21 erlcm0@21 consfzcm0@20 erlcm0@20 consfzcm0@22 erlcm0@22 consfzcm0@4 erlcm0@4 consfzcm0@23 erlcm0@23"
EV=""; for o in 0 1 2 3; do for a in consfz erl; do EV="$EV $a@$o:ood0.1"; done; done
for o in 4 23 26 29; do for a in consfzc erlc; do EV="$EV $a@$o:uns"; done; done
for o in 0 1 2 3; do for a in consfz erl; do EV="$EV $a@$o:ood0.05"; done; done
for o in 4 23 26 29; do for a in consfzc erlc; do EV="$EV $a@$o:ood0.1"; done; done
echo "[c80] start $(date +%m-%d\ %H:%M): $(echo $JOBS | wc -w) trainers, $(echo $EV | wc -w) eval-only models; waits for $(echo $C79JOBS | wc -w) chain79 claims"
c79left() { local j; for j in $C79JOBS; do [ -f $A/.c79_${j/@/_} ] || return 0; done; return 1; }
while c79left; do sleep 300; done
echo "[c80] chain79 trainers all claimed $(date +%m-%d\ %H:%M), lanes start"
lane2 $G_U0 c80 240 "$JOBS" &
sleep 40; lane2 $G_U2 c80 240 "$JOBS" &
evalx() {   # $1 arm@order:mode, mode = ood<d> | uns -> exp147 run with E147_OOD / E147_UNS (own tag suffix)
  local j=${1%:*} m=${1#*:}
  ( flock 9; export ORCH_CUDA_MAP="0=$G_U0,2=$G_U2" ORCH_EGL_MAP="0=0,2=2" E113_GPUS=0,2 E112_GPUS=0,2 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$G_U0 E113_DEV=cuda:0
    case $m in ood*) export E147_OOD=${m#ood};; uns) export E147_UNS=1;; esac
    for try in 1 2 3; do E112_PORT0=11100 E145_SEQ=${j#*@} E147_VARS=${j%@*} E147_STAGE=run .venv/bin/python -u sweep/exp147_consolidate.py 9>&- 2>&1 | grep -v hostcfg 9>&-; done ) 9> $A/.evlock
  echo "[c80] $1 EVAL EXIT $(date +%m-%d\ %H:%M) ($(cat results/merge/e112_c147s${j#*@}t*-${j%@*}${m}_w0.25_*.jsonl 2>/dev/null | grep -c '"status": "ok"') tasks ok)"
}
until grep -q "c79\] ALL EXIT" results/chain79.log 2>/dev/null; do sleep 300; done
echo "[c80] chain79 ALL EXIT seen $(date +%m-%d\ %H:%M), evals start"
left="$JOBS"; evleft=$(echo $EV)
while [ -n "$left" ] || [ -n "$evleft" ]; do
  did=0
  for j in $left; do
    [ -f $(out $j) ] || continue
    sleep 60; evalrun $j 11100 c80 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); did=1; break
  done
  if [ $did = 0 ] && [ -n "$evleft" ]; then
    e=${evleft%% *}; evalx $e; evleft=$(echo $evleft | cut -s -d' ' -f2-); did=1
  fi
  [ $did = 0 ] && sleep 120
done
wait; echo "[c80] ALL EXIT $(date +%m-%d\ %H:%M)"
