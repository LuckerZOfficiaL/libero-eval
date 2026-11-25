#!/bin/bash
# EXP-36 single-adapter trial, after the back-weighting run frees the GPUs.
cd /home/lzhou/libero-eval
while pgrep -f "merge_orchestrat[o]r.py" >/dev/null || tmux ls 2>/dev/null | grep -qE "^bw:|^lw:"; do sleep 120; done
.venv/bin/python sweep/train_lora.py --suite libero_goal_no_noops --task put_the_bowl_on_the_stove --steps 1500 --batch 8
