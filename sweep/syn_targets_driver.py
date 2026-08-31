#!/usr/bin/env python3
"""EXP-23.2 — SYNTHETIC TARGETS: manufacture the training axis the hypernetwork is starved of.

The binary-objective learning curve (hypernet_lc2.txt) is monotone with slope ~0.018pp/target and
convex — reaching ~25% of the oracle gap extrapolates to ~140 targets. The fleet has 76. Candidates
per target is flat, labels are consolidated, the objective is fixed: the NUMBER OF TARGETS is the one
axis left, and this run buys it.

A synthetic target is a perturbed adapter that still does task t:
    r_syn = r_t + 0.5 * r_perturber          (evaluated on task t's scenes, t's language)
It has its own (lower) solo baseline and its own donor-affinity profile -- a genuinely new row for the
learner, cheap because its weight-space features derive LINEARLY from existing ones:
    <r_syn, r_j> = G[t,j] + 0.5*G[p,j]       (residual Gram is bilinear)
so no GPU is needed for features, only for labels.

Per synthetic target (single-scene tasks only, to control cost):
    1 solo baseline @10 rollouts + 8 donor probes @10 rollouts  (same operator as e27)

GO/NO-GO CHECKPOINT built in: after the first 25 synthetics, extend the learning curve with mixed
real+synthetic training sets. If synthetic targets do NOT continue the real-target trend, the premise
is falsified and the remaining GPU-days should not be spent -- abort rather than run all 80.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al

OUTD = REPO / "results/analysis/decomp_adapters"
K = 16
ALPHA = 0.5          # donor coefficient (same as the method)
PERT = 0.5           # perturber coefficient defining the synthetic target
ROLLOUTS = 10
N_SYN = 80
N_CAND = 8
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
USIG = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META["n_workers"] = 3                    # tmpfs cap: 3 x 6.8G + 6.8G building < 32G
META["gpu_per_worker"] = [1, 2, 1]
SINGLE = [t for t in range(76) if len(scenes[t]) == 1]
rng = np.random.default_rng(77)
# synthetic spec list: (base task, perturber, 8 candidate donors) -- all sampled, all logged
SYN = []
for i in range(N_SYN):
    t = int(rng.choice(SINGLE))
    p = int(rng.choice([x for x in range(76) if x != t]))
    cands = sorted(int(x) for x in rng.choice([x for x in range(76) if x not in (t, p)],
                                              N_CAND, replace=False))
    SYN.append({"id": i, "task": t, "pert": p, "cands": cands})
json.dump(SYN, open(REPO / "sweep/syn_spec.json", "w"), indent=1)
_cache = {}


def load_fold(t):
    if t in _cache:
        return _cache[t]
    if len(_cache) > 16:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    _cache[t] = out
    return out


def build(name, terms):
    """W = S_16 + sum(c * r_task). terms = [(task, coeff), ...]."""
    ABs = {t: load_fold(t) for t, _ in terms}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in terms:
            b, a = ABs[t][stem]
            M = U.T @ b
            N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def rate(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return None
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if len(ss) == len(scenes[t]) else None


def gpus_free():
    return subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                          capture_output=True).returncode != 0


def run_plan(jobs):
    if not jobs:
        return
    planp = REPO / "sweep/_syn_current.json"
    json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    while not gpus_free():
        time.sleep(120)
    for spec in SYN:
        i, t, p = spec["id"], spec["task"], spec["pert"]
        base_terms = [(t, 1.0), (p, PERT)]
        # one plan per synthetic: solo + 8 donor probes, no barrier between them
        jobs, built = [], []
        if rate(f"e32_syn{i}_solo", t) is None:
            path = build(f"syn{i}_solo", base_terms)
            built.append(path)
            jobs.append({"name": f"e32_syn{i}_solo", "method": "average", "params": {},
                         "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[t]})
        for d in spec["cands"]:
            if rate(f"e32_syn{i}_p{d}", t) is not None:
                continue
            path = build(f"syn{i}_p{d}", base_terms + [(d, ALPHA)])
            built.append(path)
            jobs.append({"name": f"e32_syn{i}_p{d}", "method": "average", "params": {},
                         "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[t]})
        if jobs:
            print(f"[syn {i}] task t{t} + {PERT}*t{p}: {len(jobs)} evals", flush=True)
            run_plan(jobs)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        print(f"[done] syn{i}", flush=True)
        if i + 1 in (25, 50, 80):
            print(f"[checkpoint] {i+1} synthetics complete — run the mixed learning curve now",
                  flush=True)


if __name__ == "__main__":
    main()
