#!/bin/bash
# 10-26 13:30 EXP-144 after the shiva reboot (all 3 GPUs, EGL back). The 3 rollout sets collected with OSMesa (08:05-08:49)
# differ from EGL renders (mean |diff| ~1.3/255, 1-5% of pixels > 8) -> moved aside and recollected with EGL on GPUs 1/2;
# then 3 trainers (GPUs 0/1/2) take them from the claim queue while tmux t0 finishes xed0e9e on GPU 0; then base + merge on
# all GPUs + report.
cd /home/lzhou/libero-eval; set -u
U0=GPU-8702ed42-9555-9082-d07a-3542fced5400; U1=GPU-ccf73706-2608-2fbb-9d35-11cefe9dcd5e; U2=GPU-9abbca2b-49ca-32cd-22e8-feffe648aaba
A=results/analysis/new_loras_s16wp_self; mkdir -p results/exp144_dumps_osmesa results/merge_osmesa
for t in x47b567 x290bfc xb61165; do
  mv results/exp144_dumps/e144_${t}_self results/exp144_dumps_osmesa/ && mv results/merge/e144_${t}_self.jsonl results/merge_osmesa/
done
export E113_DEV=cuda:0 E112_PER_GPU=2 CUDA_VISIBLE_DEVICES=$U1 ORCH_CUDA_MAP="1=$U1,2=$U2" ORCH_EGL_MAP="1=1,2=2" E113_GPUS=1,2 E112_PORT0=8600
echo "[c14] collect start $(date '+%m-%d %H:%M')"
for try in 1 2; do E144_STAGE=collect .venv/bin/python -u sweep/exp144_selfconvert.py; done
rm -f $A/.claim_x47b567 $A/.claim_x290bfc $A/.claim_xb61165
echo "[c14] train start $(date '+%m-%d %H:%M')"
for u in $U1 $U2 $U0; do CUDA_VISIBLE_DEVICES=$u E144_STAGE=train .venv/bin/python -u sweep/exp144_selfconvert.py & sleep 30; done; wait
while tmux has-session -t t0 2>/dev/null; do sleep 60; done
echo "[c14] E144 TRAIN EXIT $(date '+%m-%d %H:%M')"
export ORCH_CUDA_MAP="0=$U0,1=$U1,2=$U2" ORCH_EGL_MAP="0=0,1=1,2=2" E113_GPUS=0,1,2 E112_PORT0=8000
for st in base run; do for try in 1 2; do E144_STAGE=$st .venv/bin/python -u sweep/exp144_selfconvert.py; done; done
E144_STAGE=report .venv/bin/python -u sweep/exp144_selfconvert.py 2>&1 | grep -v hostcfg
echo "[c14] E144 ALL EXIT $(date '+%m-%d %H:%M')"
