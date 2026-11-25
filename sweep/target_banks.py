#!/usr/bin/env python3
"""EXP-58 step 1 — per-target observation banks from each target's own demonstrations.

The shared 180-observation bank spans 90 scenes (2 per scene) and was built to make adapters directly
comparable. That is the right choice for comparing ADAPTERS, but the wrong one for predicting whether a
merge helps a SPECIFIC target: only ~2 of its 180 observations come from that target's task, so route 1
effectively asked "how did this merge change the policy globally" instead of "how did it change the
policy where the target has to work". This builds a bank from the target's own demo states.
"""
import json, pathlib, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import train_lora as TL
from reinforce_train import scenes as SC, target_baseline
N_OBS = 60
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
WEAK = [int(t) for t in RANK if target_baseline(int(t)) < 0.9]
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
LANG = {a["task_index"]: a["language"] for a in plan}
out = REPO / "results/analysis/obs_banks"; out.mkdir(parents=True, exist_ok=True)
for t in WEAK:
    f = out / f"t{t}.npz"
    if f.exists():
        continue
    scene = SC[t][0]
    try:
        s = TL.load_demo_samples("libero_90_no_noops", scene, n_demos=6, stride=4)
    except FileNotFoundError:
        print(f"  t{t}: no demo file for {scene[:40]}"); continue
    rng = np.random.default_rng(t)
    idx = rng.choice(len(s), min(N_OBS, len(s)), replace=False)
    np.savez_compressed(f,
        img=np.stack([s[i][0]["observation/image"] for i in idx]),
        wrist=np.stack([s[i][0]["observation/wrist_image"] for i in idx]),
        state=np.stack([s[i][0]["observation/state"] for i in idx]),
        prompt=np.array([LANG[t]] * len(idx)), scene=np.array([scene] * len(idx)))
    print(f"  t{t}: {len(idx)} observations from {scene[:44]}", flush=True)
print(f"[banks] {len(list(out.glob('t*.npz')))} target banks ready")
