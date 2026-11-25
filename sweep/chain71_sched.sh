#!/bin/bash
# 11-19 single trainer scheduler, continual first (user 11-19: frame the project as CONTINUAL merging). chain67's lanes always had a
# one-shot job ready, so the serial continual chains (erlc / consfac / seqc / ewcc: 4 rounds each) and chain68b / 70b's trainers
# would start last. All their not-yet-started jobs are pre-claimed under the old chains' names (their lanes exit; their eval
# loops still evaluate every adapter that appears) and trained here in one explicit order: continual rounds and chain roots
# first (critical path), the one-shot ER orders interleaved, then lrc, then the one-shot extras. Lanes via sweep/gpu_gate.sh
# (60 s double check + reservation), claims .c71_<arm>_<order>; a continual round waits for its previous round's adapter.
cd /home/lzhou/libero-eval; set -u; unset E145_BLEND E145_LAMS E145_VARS E145_MIX E147_VARS
. sweep/gpu_gate.sh
JOBS="erlc@19 consfa@0 ewc@0 erl@2 erlc@20 seqc@19 consfac@19 ewcc@19 erl@3 erlc@4 lrc@0 seqc@20 consfac@20 ewcc@20 seqc@4 consfac@4 ewcc@4 consb@2 consb@3 consfa@1 consfa@2 consfa@3 consb64@0"
echo "[c71] start $(date +%m-%d\ %H:%M)"
lane2 $G_U0 c71 60 "$JOBS" &
sleep 40; lane2 $G_U2 c71 60 "$JOBS" &
wait; echo "[c71] ALL EXIT $(date +%m-%d\ %H:%M)"
