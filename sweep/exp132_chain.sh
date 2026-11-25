#!/bin/bash
# EXP-132 chain (shiva, all 3 GPUs; GPU 0 free again since 10-19): train 22 seed-1 write-protected adapters (3 shards)
# -> single-task baselines -> privileged merges (sequences 0-7, w = 1 / 0.25) -> report.
cd /home/lzhou/libero-eval
set -u
true
mkdir -p results/analysis/new_loras_s16wp_seed1 results/merge_logs
train_shard() {  # $1 = physical GPU, $2 = shard index
  local i=0
  while read -r suite task; do
    if [ $((i % 3)) -eq "$2" ]; then
      local dir="results/analysis/new_loras_s16wp_seed1/${suite}__${task}"
      if [ ! -f "$dir/adapter_model.safetensors" ]; then
        echo "[gpu$1] training $suite/$task $(date +%H:%M)"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 .venv/bin/python sweep/train_lora.py --suite "$suite" --task "$task" \
          --steps 1500 --batch 8 --device cuda:0 --init s16wp --seed 1 --out "$dir" > "results/merge_logs/e132_${task}.log" 2>&1 \
          || echo "[gpu$1] FAILED $task"
      fi
    fi
    i=$((i + 1))
  done < results/analysis/exp132_tasks.txt
  echo "[gpu$1] training shard done $(date +%H:%M)"
}
train_shard 0 0 & train_shard 1 1 & train_shard 2 2 &
wait
echo "[e132] TRAIN EXIT $(date +%H:%M)"
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e12*
export E113_GPUS=0,1,2 E113_DEV=cuda:1 E113_ROUND=3
E132_STAGE=base .venv/bin/python -u sweep/exp132_seeds.py
echo "[e132] BASE EXIT $(date +%H:%M)"
E132_STAGE=run .venv/bin/python -u sweep/exp132_seeds.py
echo "[e132] RUN EXIT $(date +%H:%M)"
E132_STAGE=report .venv/bin/python -u sweep/exp132_seeds.py 2>&1 | grep -v hostcfg | tee results/exp132_report_console.txt
echo "[e132] REPORT EXIT $(date +%H:%M)"
