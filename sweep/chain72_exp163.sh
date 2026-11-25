#!/bin/bash
# 11-19 EXP-163 (pre-registered, conditional on EXP-158 vs erl ending TIE or erl ahead): pooled orders 0-2 are library +3.3 /
# new -7.5 and ADVANTAGE would need ER's order-3 library <= ~75 (it has been 84-87), so the condition is met in practice;
# started before erl@3 to save a day. Arms (exp147): consfzq / erlq = ours / ER at 750 steps, consfz10 / erl10 = on the first 10
# episodes per new skill; orders 0 / 1. Then chain71's one-shot jobs (pre-claimed there): lrc@0 consb@2 consb@3 consfa@1-3 consb64@0.
# Trainer lanes yield to chain71 (90 s double check vs its 60 s) -> continual rounds keep priority; evals (EXP-163 arms only;
# chain67b evaluates the extras) wait for no pending eval of chain67b / 68b / 70b, then take the eval lock; 2 workers per GPU.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
J163="consfzq@0 erlq@0 consfzq@1 erlq@1 consfaq@0 consfaq@1 consfz10@0 erl10@0 consfz10@1 erl10@1"
# 14:20 restart: ER tied ours at arrival 4 (EXP-159 call failed) -> EXP-163 decides whether merging the arrivals adds anything;
# its 750-step arms move ahead of chain71's remaining SeqFT / EWC cells (pre-claimed there, run here before the 10-rollout arms)
# 14:37 restart: + EXP-164 (the library merge as ER's start: consbc stream, erlr / consbr demo-free, erld / consbd demo budget)
J164="consbc@19 consbc@20 consbc@4 erlr@0 consbr@0 erld1@0 consbd1@0 erld25@0 consbd25@0"
# 14:45: the demo-budget arms (erld / consbd) moved behind the 10-rollout arms (one-shot merge vs base start already tested on orders 0-1)
JOBS="consfzq@0 erlq@0 consfzq@1 erlq@1 consfaq@0 consfaq@1 consbc@19 erlr@0 consbr@0 consbc@20 consbc@4 consb@2 consb@3 ewcc@20 seqc@4 ewcc@4 consfz10@0 erl10@0 consfz10@1 erl10@1 erld1@0 consbd1@0 erld25@0 consbd25@0 lrc@0 consfa@1 consfa@2 consfa@3 consb64@0"
echo "[c72] start $(date +%m-%d\ %H:%M)"
lane2 $G_U0 c72 90 "$JOBS" &
sleep 40; lane2 $G_U2 c72 90 "$JOBS" &
left="$J163 $J164"
while [ -n "$left" ]; do
  for j in $left; do
    [ -f $(out $j) ] || continue
    until ! pending "$C67JOBS $C68JOBS $C70JOBS"; do sleep 120; done
    sleep 60; evalrun $j 10600 c72 2
    left=$(echo $left | tr ' ' '\n' | grep -vx $j | tr '\n' ' ' | sed 's/ *$//'); break
  done
  sleep 120
done
wait; echo "[c72] ALL EXIT $(date +%m-%d\ %H:%M)"
