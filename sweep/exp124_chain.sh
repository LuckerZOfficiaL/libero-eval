#!/bin/bash
# EXP-124 chain (shiva, GPUs 1,2; GPU 0 left free): wait for EXP-123 fix -> train 11 write-protected adapters (2 shards)
# -> single-task baselines -> privileged merges (sequences 0-7, w = 1 / 0.25) -> report.
cd /home/lzhou/libero-eval
set -u
until grep -q "REPORT2 EXIT" results/exp123.log 2>/dev/null; do sleep 120; done
mkdir -p results/analysis/new_loras_s16wp results/merge_logs
train_shard() {  # $1 = physical GPU, $2 = shard index
  local i=0
  while read -r suite task; do
    if [ $((i % 2)) -eq "$2" ]; then
      local dir="results/analysis/new_loras_s16wp/${suite}__${task}"
      if [ ! -f "$dir/adapter_model.safetensors" ]; then
        echo "[gpu$1] training $suite/$task $(date +%H:%M)"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 .venv/bin/python sweep/train_lora.py --suite "$suite" --task "$task" \
          --steps 1500 --batch 8 --device cuda:0 --init s16wp --out "$dir" > "results/merge_logs/e124_${task}.log" 2>&1 \
          || echo "[gpu$1] FAILED $task"
      fi
    fi
    i=$((i + 1))
  done < results/analysis/exp124_wp_tasks.txt
  echo "[gpu$1] training shard done $(date +%H:%M)"
}
train_shard 1 0 & train_shard 2 1 &
wait
echo "[e124] TRAIN EXIT $(date +%H:%M)"
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e12*
export E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=3 E113_PER_GPU=1
E124_STAGE=base .venv/bin/python -u sweep/exp124_wpcontinual.py
echo "[e124] BASE EXIT $(date +%H:%M)"
unset E113_PER_GPU; E124_STAGE=run .venv/bin/python -u sweep/exp124_wpcontinual.py
echo "[e124] RUN EXIT $(date +%H:%M)"
E124_STAGE=report .venv/bin/python -u sweep/exp124_wpcontinual.py 2>&1 | grep -v hostcfg | tee results/exp124_report_console.txt
echo "[e124] REPORT EXIT $(date +%H:%M)"
