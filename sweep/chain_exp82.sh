#!/bin/bash
# EXP-82: wait for inputs -> base-policy pass over the demo bank (GPU 2, small footprint) -> CPU training.
cd /home/lzhou/libero-eval
until grep -q '^\[bank\] done' results/merge_logs/demo_bank.log 2>/dev/null && grep -q '^\[tok\] done' results/merge_logs/lora_col_tokens.log 2>/dev/null; do sleep 60; done
sleep 5
echo "=== [e82] inputs ready $(date) ==="
if [ ! -f results/analysis/demo_bank_base.npz ]; then
  # torch device order != nvidia-smi order (K620 is torch:2). Pick the P6000 with the most free memory.
  DEV=$(.venv/bin/python -c "import torch;c=[(torch.cuda.mem_get_info(i)[0],i) for i in range(torch.cuda.device_count()) if torch.cuda.get_device_properties(i).total_memory>16e9];print('cuda:%d'%max(c)[1])")
  echo "[e82] base pass on $DEV"
  TORCHDYNAMO_DISABLE=1 .venv/bin/python sweep/activation_probe.py --plan sweep/plan_ext.json --bank results/analysis/demo_bank.npz \
      --base demo_bank_base --outdir results/analysis --device $DEV
fi
[ -f results/analysis/demo_bank_base.npz ] || { echo "=== [e82] base pass FAILED ==="; exit 1; }
echo "=== [e82] base pass done $(date); training ==="
.venv/bin/python sweep/xattn_selector.py --seeds 36 --epochs 150 2>&1 | tee results/analysis/exp82_report.txt
echo "=== [e82] DONE $(date) ==="
