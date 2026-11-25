#!/bin/bash
# EXP-43b: raw-weight encoder at 106 targets (seeded), holdouts B (fleet) and C (new). Compare to the
# hand-feature model under the identical protocol (EXP-40 / EXP-43a).
cd /home/lzhou/libero-eval
export CUDA_VISIBLE_DEVICES=""
test -f results/analysis/lora_tokens_raw106.pt || .venv/bin/python sweep/lora_tokenizer.py --plan sweep/plan106.json --out results/analysis/lora_tokens_raw106.pt
for HO in fleet new; do
  .venv/bin/python sweep/raw_supervised.py --tokens results/analysis/lora_tokens_raw106.pt \
    --labels results/analysis/pair_labels106.json --feats results/analysis/hypernet_feats106.npz \
    --baselines results/analysis/baselines106.json --holdout $HO --seeds 12 \
    --out results/analysis/exp43_raw_${HO}.txt
done
echo "EXP43b DONE"
