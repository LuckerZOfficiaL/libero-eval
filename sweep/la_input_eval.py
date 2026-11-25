#!/usr/bin/env python3
"""EXP-94 evaluation: data-aware linear-algebra donor scores (la_input_score.npz) vs the probe labels,
with exactly the EXP-91 protocol (within-target Spearman; score + sign chosen on half the targets)."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import la_pair_eval as E
z = np.load(E.REPO / "results/analysis/la_input_score.npz")
done = z["done"]; mag, dr, rn2 = z["mag"], z["dir"], z["rn2"]
def agg(x): return np.concatenate([x.sum(0, keepdims=True), x], 0)
mag, dr, rn2 = agg(mag), agg(dr), agg(rn2)
feats = {}
for b in range(6):
    self_ = np.array([mag[b, t, t] for t in range(E.N)])[:, None]
    feats[f"gain|{E.BANDS[b]}"] = mag[b] / np.maximum(rn2[b][None, :], 1e-30)
    feats[f"align|{E.BANDS[b]}"] = dr[b] / np.sqrt(np.maximum(mag[b] * self_, 1e-60))
    feats[f"rel_gain|{E.BANDS[b]}"] = (mag[b] / np.maximum(rn2[b][None, :], 1e-30)) / np.maximum(self_ / np.maximum(rn2[b][:, None], 1e-30), 1e-30)
lab = E.labels(); by = {}
for (t, d), v in lab.items():
    if done[t]: by.setdefault(t, {})[d] = v
T = sorted(t for t, v in by.items() if len(v) >= 5 and t not in (18, 62))
print(f"{int(done.sum())}/76 targets scored; {len(T)} with >=5 labelled donors")
E.run_protocol(feats, by, T, "EXP-94: data-aware LA donor scores (donor's first-order effect on the target's own layer inputs) vs probe labels",
               "exp94_la_input_eval.txt")
