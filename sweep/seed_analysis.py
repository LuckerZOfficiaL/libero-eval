#!/usr/bin/env python3
"""EXP-38 analysis — do same-task seed variants carry NEW information about donor affinity?

(a) weight space: cosine between adapters of the same task (different seeds) vs different tasks.
(b) label space (decisive): for each task, the 8 fleet candidates were probed for seed 0 (EXP-39, e51)
    and seeds 1,2 (e53) with the SAME 8 candidates (the driver draws candidates from the base task
    name, so '__seedK' variants share seed 0's list; the intersection check below is a safety net).
    Within-task Spearman ~ 1  -> variants are duplicates; the learning curve cannot be extended by seeds.
    Within-task Spearman low  -> the interaction varies with seed; variants are real targets.
    Reference: across-task Spearman on shared candidates (should be ~0).
"""
import json, pathlib, re, sys, itertools
import numpy as np
from scipy import stats
sys.path.insert(0, "sweep"); sys.path.insert(0, "scripts")
import torch, safetensors.torch as st
REPO = pathlib.Path(__file__).resolve().parent.parent

def labels(report):
    txt = (REPO / report).read_text()
    return json.loads(txt.split("labels: ", 1)[1].strip())

L0 = labels("results/analysis/newpool_probe_report.txt")          # seed 0 (e51)
L1 = labels("results/analysis/newpool_seeds_report.txt")          # seeds 1,2 (e53)
base_of = lambda name: name.split("__seed")[0]
by_task = {}
for name, v in list(L0.items()) + list(L1.items()):
    by_task.setdefault(base_of(name), {})[name] = v
by_task = {k: v for k, v in by_task.items() if len(v) >= 2}
print(f"tasks with >=2 seed variants: {len(by_task)}")

# (b) label-space agreement
within, across = [], []
def aff(v): return {int(p): 100 * (s - v["base"]) for p, s in v["probes"].items()}
for task, vs in by_task.items():
    names = sorted(vs)
    for a, b in itertools.combinations(names, 2):
        A, B = aff(vs[a]), aff(vs[b]); common = sorted(set(A) & set(B))
        if len(common) >= 4:
            r = stats.spearmanr([A[c] for c in common], [B[c] for c in common]).correlation
            within.append((task[:40], a[-6:], b[-6:], len(common), r))
tasks = sorted(by_task)
for ta, tb in itertools.combinations(tasks, 2):
    for a in by_task[ta]:
        for b in by_task[tb]:
            A, B = aff(by_task[ta][a]), aff(by_task[tb][b]); common = sorted(set(A) & set(B))
            if len(common) >= 4:
                across.append(stats.spearmanr([A[c] for c in common], [B[c] for c in common]).correlation)
print("\n(b) donor-affinity agreement (Spearman over shared candidates)")
for t, a, b, n, r in within:
    print(f"   same task {t:<40} {a} vs {b}  n={n}  rho={r:+.2f}")
w = np.array([r for *_, r in within if np.isfinite(r)]); x = np.array([r for r in across if np.isfinite(r)])
print(f"   WITHIN-task mean rho {w.mean():+.2f} (n={len(w)})   ACROSS-task mean rho {x.mean():+.2f} (n={len(x)})")
print("   ~1 within => duplicates (no new info)    ~across => seed changes the interaction (real new targets)")

# (a) weight-space cosine (subsampled stems)
from reinforce_train import stems
SUB = stems[::12]
def vec(path):
    w = st.load_file(str(path))
    return torch.cat([(w[f"{s}.lora_B.weight"].float() @ w[f"{s}.lora_A.weight"].float()).flatten() for s in SUB])
dirs = {}
for d in list((REPO / "results/analysis/new_loras_s16").glob("libero_*__*")) + list((REPO / "results/analysis/new_loras_seeds").glob("libero_*__*")):
    if (d / "adapter_model.safetensors").exists() and base_of(d.name) in by_task:
        dirs[d.name] = vec(d / "adapter_model.safetensors")
cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
wi, ac = [], []
names = sorted(dirs)
for a, b in itertools.combinations(names, 2):
    (wi if base_of(a) == base_of(b) else ac).append(cos(dirs[a], dirs[b]))
print(f"\n(a) full-delta cosine: same task/different seed {np.mean(wi):.3f} (n={len(wi)})   different tasks {np.mean(ac):.3f} (n={len(ac)})   [fleet-fleet ~0.87: the shared S16 dominates both]")
# residual-space cosine (remove S16) is the informative one
from reinforce_train import USIG, VS
def rvec(path):
    w = st.load_file(str(path)); out = []
    for s in SUB:
        D = w[f"{s}.lora_B.weight"].float() @ w[f"{s}.lora_A.weight"].float()
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        out.append((D - Un @ (Un.T @ D @ VS[s]) @ VS[s].T).flatten())
    return torch.cat(out)
rd = {n: rvec(REPO / ("results/analysis/new_loras_s16" if "__seed" not in n else "results/analysis/new_loras_seeds") / n / "adapter_model.safetensors") for n in names}
wi, ac = [], []
for a, b in itertools.combinations(names, 2):
    (wi if base_of(a) == base_of(b) else ac).append(cos(rd[a], rd[b]))
print(f"    RESIDUAL cosine:   same task/different seed {np.mean(wi):.3f}   different tasks {np.mean(ac):.3f}")
