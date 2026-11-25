#!/bin/bash
# After sigs_ext: copy signatures, run the EXP-42 landing (166-target study), then EXP-44 (score the 16
# held-out targets on odin, build+commit their merges on shiva).
cd /home/lzhou/libero-eval
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
while $SSH shiva 'tmux has-session -t sigs_ext 2>/dev/null'; do sleep 120; done
$SSH shiva 'cd ~/libero-eval/results/analysis/act_sigs && tar cf - $(ls task1[0-6][0-9].npz task17[01].npz 2>/dev/null)' | tar xf - -C results/analysis/act_sigs/
echo "[land] signatures on odin: $(ls results/analysis/act_sigs/task1[0-6][0-9].npz results/analysis/act_sigs/task17[01].npz 2>/dev/null | wc -l)/66"
bash sweep/land_exp42.sh || { echo "[land] EXP-42 landing FAILED"; exit 1; }
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python sweep/exp44_deploy.py --stage plan || { echo "[land] EXP-44 plan FAILED"; exit 1; }
rsync -a results/analysis/exp44_picks.json results/analysis/hypernet_featsexta48.npz results/analysis/pair_labels_ext.json results/analysis/baselines_ext.json shiva:/home/lzhou/libero-eval/results/analysis/
rsync -a sweep/exp44_deploy.py sweep/plan_ext.json shiva:/home/lzhou/libero-eval/sweep/
$SSH shiva 'cd ~/libero-eval && tmux new-session -d -s e44 "bash -c \"while pgrep -f merge_orchestrat[o]r.py >/dev/null; do sleep 60; done; .venv/bin/python sweep/exp44_deploy.py --stage build && .venv/bin/python sweep/merge_orchestrator.py sweep/_e44_plan.json; echo [e44] done\" 2>&1 | tee results/merge_logs/e44.log"'
echo "[land] EXP-44 commits launched on shiva (tmux e44)"
echo "LAND ALL DONE"
