#!/bin/bash
# EXP-126 chain (shiva, GPUs 1,2; GPU 0 left free): wait for EXP-125 -> train 12 ablation adapters (soft / readout, 2 shards)
# -> single-task baselines (prebuilt) -> privileged merges (sequences 0-3, w = 1 / 0.25) -> report.
cd /home/lzhou/libero-eval
set -u
until grep -q "REPORT EXIT" results/exp125.log 2>/dev/null; do sleep 120; done
mkdir -p results/analysis/new_loras_s16soft results/analysis/new_loras_readout results/merge_logs
train_shard() {  # $1 = physical GPU, $2 = shard index
  local i=0
  while read -r var suite task; do
    if [ $((i % 2)) -eq "$2" ]; then
      if [ "$var" = soft ]; then init=s16soft; root=new_loras_s16soft; else init=readout; root=new_loras_readout; fi
      local dir="results/analysis/${root}/${suite}__${task}"
      if [ ! -f "$dir/adapter_model.safetensors" ]; then
        echo "[gpu$1] training $var $suite/$task $(date +%H:%M)"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 .venv/bin/python sweep/train_lora.py --suite "$suite" --task "$task" \
          --steps 1500 --batch 8 --device cuda:0 --init "$init" --out "$dir" > "results/merge_logs/e126_${var}_${task}.log" 2>&1 \
          || echo "[gpu$1] FAILED $var $task"
      fi
    fi
    i=$((i + 1))
  done < results/analysis/exp126_tasks.txt
  echo "[gpu$1] training shard done $(date +%H:%M)"
}
train_shard 1 0 & train_shard 2 1 &
wait
echo "[e126] TRAIN EXIT $(date +%H:%M)"
rm -rf /dev/shm/mergesweep/cache /dev/shm/mergesweep/prebuilt/e12*
export E113_GPUS=1,2 E113_DEV=cuda:1 E113_ROUND=3
E126_STAGE=base .venv/bin/python -u sweep/exp126_ablations.py
echo "[e126] BASE EXIT $(date +%H:%M)"
E126_STAGE=run .venv/bin/python -u sweep/exp126_ablations.py
echo "[e126] RUN EXIT $(date +%H:%M)"
E126_STAGE=report .venv/bin/python -u sweep/exp126_ablations.py 2>&1 | grep -v hostcfg | tee results/exp126_report_console.txt
echo "[e126] REPORT EXIT $(date +%H:%M)"
