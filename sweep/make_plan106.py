#!/usr/bin/env python3
"""Build sweep/plan106.json — the 76 fleet adapters plus the 30 S16-init experts as indices 76-105.

Every downstream tool (activation_probe, similarity_analysis, residual_rank, hypernet_features) keys
adapters by integer task_index through a plan file. Giving the new experts indices in the same
namespace is what lets those tools widen from 76 to 106 without touching their internals.

The new experts are legitimate members of this namespace only because of EXP-37: trained from S16 they
carry capture@16 0.885 and cosine 0.92 to the fleet, versus 0.001/0.003 for the zero-init cohort that
this file deliberately does NOT include.
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
import sys
SRC = REPO / "results/analysis/new_loras_s16"
EXTRA = [REPO / d for d in sys.argv[2:]]          # e.g. new_loras_seeds new_loras_var50 -> variants get indices 106+
OUT = sys.argv[1] if len(sys.argv) > 1 else "sweep/plan106.json"

plan = json.load(open(REPO / "sweep/plan.json"))
adapters = list(plan["adapters"])
nxt = max(a["task_index"] for a in adapters) + 1
added = []
for d in sorted(SRC.glob("libero_*__*")):
    ad = d / "adapter_model.safetensors"
    if not ad.exists():
        continue
    suite, task = d.name.split("__", 1)
    adapters.append({"task_index": nxt, "adapter_file": str(ad), "scenes": [task],
                     "language": task.replace("_", " "),
                     "suite": {"libero_goal_no_noops": "libero_goal",
                               "libero_object_no_noops": "libero_object",
                               "libero_spatial_no_noops": "libero_spatial"}[suite],
                     "source": "s16_init"})
    added.append((nxt, task))
    nxt += 1
SUITE = {"libero_goal_no_noops": "libero_goal", "libero_object_no_noops": "libero_object",
         "libero_spatial_no_noops": "libero_spatial", "libero_90_no_noops": "libero_90"}
for src in EXTRA:
    for d in sorted(src.glob("libero_*__*")):
        ad = d / "adapter_model.safetensors"
        if not ad.exists():
            continue
        suite, rest = d.name.split("__", 1)
        task = rest.split("__seed")[0]
        meta = json.load(open(d / "train_meta.json")) if (d / "train_meta.json").exists() else {}
        adapters.append({"task_index": nxt, "adapter_file": str(ad), "scenes": [task],
                         "language": meta.get("prompt", task.replace("_", " ")), "suite": SUITE[suite],
                         "source": f"s16_seed{meta.get('seed', '?')}", "variant_of": task})
        added.append((nxt, d.name)); nxt += 1
out = dict(plan, adapters=adapters)
json.dump(out, open(REPO / OUT, "w"), indent=1)
print(f"{OUT}: {len(adapters)} adapters ({len(added)} new, indices {added[0][0]}-{added[-1][0]})")
for i, t in added[:3]:
    print(f"  {i}: {t}")
