#!/bin/bash
# EXP-127 chain (shiva, GPUs 1,2; GPU 0 left free): wait for nothing -> train 4 write-protected adapters (2 shards)
# -> single-task baselines -> privileged merges (sequences 0-7, w = 1 / 0.25) -> report.
cd /home/lzhou/libero-eval
set -u
true
mkdir -p results/analysis/new_loras_s16wp results/merge_logs
train_shard() {  # $1 = physical GPU, $2 = shard index
  local i=0
  while read -r suite task; do
    if [ $((i % 2)) -eq "$2" ]; then
      local dir="results/analysis/new_loras_s16wp/${suite}__${task}"
      if [ ! -f "$dir/adapter_model.safetensors" ]; then
        echo "[gpu$1] training $suite/$task $(date +%H:%M)"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 .venv/bin/python sweep/train_lora.py --suite "$suite" --task "$task" \
          --steps 1500 --batch 8 --device cuda:0 --init s16wp --out "$dir" > "results/merge_logs/e127_${task}.log" 2>&1 \
          || echo "[gpu$1] FAILED $task"
      fi
    fi
    i=$((i + 1))
  done < results/analysis/exp127_wp_tasks.txt
  echo "[gpu$1] training shard done $(date +%H:%M)"
}
train_shard 1 0 & train_shard 2 1 &
wait
echo "[e127] TRAIN EXIT $(date +%H:%M)"
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e12*
export E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=3
E127_STAGE=base .venv/bin/python -u sweep/exp127_replicate_wp.py
echo "[e127] BASE EXIT $(date +%H:%M)"
E127_STAGE=run .venv/bin/python -u sweep/exp127_replicate_wp.py
echo "[e127] RUN EXIT $(date +%H:%M)"
E127_STAGE=report .venv/bin/python -u sweep/exp127_replicate_wp.py 2>&1 | grep -v hostcfg | tee results/exp127_report_console.txt
echo "[e127] REPORT EXIT $(date +%H:%M)"
