#!/usr/bin/env python3
"""EXP-82 input: a bank of each target's OWN demo frames (the target-domain data), 16 per target, for all
172 plan_ext adapters. Same keys as obs_bank.npz plus 'target'. DA-MergeLoRA's query is the frozen
encoder's mean embedding of a batch of target-domain images; this is our analogue."""
import json, pathlib, sys, numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
from train_lora import load_demo_samples
K = 16
plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
img, wr, st, pr, sc, tg = [], [], [], [], [], []
for a in plan:
    t = a["task_index"]; suite = a.get("suite", "libero_90"); suite = suite if suite.endswith("_no_noops") else suite + "_no_noops"; task = a["scenes"][0]
    try:
        S = load_demo_samples(suite, task, n_demos=4, stride=4)
    except Exception as e:
        print(f"[bank] t{t} {suite}/{task[:40]}: FAILED {type(e).__name__}: {e}", flush=True); continue
    idx = np.linspace(0, len(S) - 1, K).round().astype(int)
    for i in idx:
        o, _ = S[i]
        img.append(np.asarray(o["observation/image"], dtype=np.uint8)); wr.append(np.asarray(o["observation/wrist_image"], dtype=np.uint8))
        st.append(np.asarray(o["observation/state"], dtype=np.float32)); pr.append(str(o.get("prompt", a["language"]))); sc.append(task); tg.append(t)
    if t % 20 == 0: print(f"[bank] t{t} ok ({len(S)} samples -> {K})", flush=True)
np.savez_compressed(REPO / "results/analysis/demo_bank.npz", img=np.stack(img), wrist=np.stack(wr), state=np.stack(st),
                    prompt=np.array(pr), scene=np.array(sc), target=np.array(tg))
print(f"[bank] done: {len(tg)} obs, {len(set(tg))} targets", flush=True)
