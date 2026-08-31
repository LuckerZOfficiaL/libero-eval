#!/usr/bin/env python3
"""E1 + E2 — DOSING of the asymmetric operator, with a single probe-selected donor.

    W = S_16 + beta * r_target + alpha * r_donor

Everything so far has used exactly ONE dose (beta=1.0, alpha=0.5). Two questions:

  E1 (alpha grid, "how much donor?")     beta=1.0, alpha in {0.25, 0.5, 1.0}
  E2 (interpolation, "must the target    (beta, alpha) in {(0.75,0.25), (0.5,0.5)}
      stay protected at 1.0?")

The arms are chosen so three separate effects can be read off by PAIRING, not confounded together:
  (1.0,0.25) vs (0.75,0.25)  -> same donor dose, less target  = the price of un-protecting the target
  (1.0,1.0)  vs (0.5,0.5)    -> same ratio, half the scale    = overall magnitude vs balance
  (1.0,0.25) vs (1.0,0.5) vs (1.0,1.0) -> pure donor dose-response

FREE REUSE: (beta=1.0, alpha=0.5) with the top-1 donor is *exactly* the e27 pairwise probe operator at
the same 10 rollouts, so that cell is read from disk instead of re-run (8 of 40 cells).

Donor = each target's top-1 probe pick from e27 (its behaviourally best single partner).
Targets: 5 weak (headroom, where dose is visible) + 3 strong, two of which the ungated method HARMED
(t21 100->88, t69 100->92) -- if a smaller dose removes that harm, that is a deployable fix.
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
ROLLOUTS = 10
TARGETS = [6, 19, 56, 15, 50, 21, 69, 25]          # 5 weak + 3 strong (t21,t69 were harmed)
ARMS = [("b100_a025", 1.00, 0.25),                 # E1
        ("b100_a100", 1.00, 1.00),                 # E1
        ("b075_a025", 0.75, 0.25),                 # E2
        ("b050_a050", 0.50, 0.50),                 # E2
        # DE-BIASING ARM. The e27 probe cell is also (beta=1, alpha=.5), but it CANNOT be used as the
        # alpha=0.5 measurement: the donor was chosen BECAUSE it scored highest at alpha=0.5 among 8
        # candidates, so that cell is a max-of-8 and carries the winner's curse. Every other cell is a
        # fresh measurement of that same donor. Comparing them would manufacture a spike at exactly the
        # dose used for selection. This arm re-measures alpha=0.5 independently.
        ("b100_a050_fresh", 1.00, 0.50)]
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
# CAPACITY FIX: the orchestrator's build cache is tmpfs (/dev/shm, 32G) and each merged PI0.5 policy is
# 6.8G. Four workers hold four DISTINCT policies = 27G resident; the next build writes a .build temp dir
# first, needing 34G -> guaranteed ENOSPC ("No space left on device"). Three workers = 20.4G resident
# + 6.8G building = 27.2G, which fits with margin. Costs ~25% throughput, buys a run that finishes.
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]
# top-1 probe donor per target, from last week's all-76 run
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
DONOR = {T: RANK[str(T)][0][0] for T in TARGETS}
_cache = {}


def load_fold(t):
    if t in _cache:
        return _cache[t]
    if len(_cache) > 20:
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


def build(name, target, donor, beta, alpha):
    """W = S_16 + beta*r_target + alpha*r_donor, emitted as LoRA factors (stays low-rank)."""
    ABs = {t: load_fold(t) for t in (target, donor)}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in ((target, beta), (donor, alpha)):
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


def main():
    while not gpus_free():
        time.sleep(120)
    for T in TARGETS:
        d = DONOR[T]
        jobs, built = [], []
        for tag, beta, alpha in ARMS:
            job = f"e28_t{T}_{tag}"
            if rate(job, T) is not None:
                continue
            path = build(f"ds_t{T}_{tag}", T, d, beta, alpha)
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        if jobs:
            planp = REPO / "sweep/_dosing_current.json"
            json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
            print(f"[run] t{T} donor=t{d} — {len(jobs)} arms", flush=True)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                            str(planp)], cwd=REPO)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
    report()


def report():
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]

    def brate(t):
        ss = [s for s in scenes[t] if s in base]
        return sum(base[s] for s in ss) / len(ss)

    def probe050(t):
        """(beta=1, alpha=.5) with the top-1 donor == the e27 probe cell, reused free."""
        return RANK[str(t)][0][1]

    cells = [("b100_a025", "1.00 / 0.25"), ("b100_a050_fresh", "1.00/0.50 FRESH"),
             ("__probe__", "1.00/0.50 biased"), ("b100_a100", "1.00 / 1.00"),
             ("b075_a025", "0.75 / 0.25"), ("b050_a050", "0.50 / 0.50")]
    L = ["=== E1+E2 DOSING: W = S_16 + beta*r_target + alpha*r_donor  (single top-1 probe donor) ===",
         f"{ROLLOUTS} rollouts. '1.00/0.50' is the e27 probe cell (reused, not re-run).", "",
         f"{'target':<8}{'donor':>7}{'base':>7}" + "".join(f"{lab:>13}" for _, lab in cells)]
    acc = {tag: [] for tag, _ in cells}
    for T in TARGETS:
        row = f"t{T:<7}{'t'+str(DONOR[T]):>7}{brate(T)*100:>6.0f}%"
        for tag, _ in cells:
            v = probe050(T) if tag == "__probe__" else rate(f"e28_t{T}_{tag}", T)
            if v is None:
                row += f"{'—':>13}"
            else:
                d = (v - brate(T)) * 100
                acc[tag].append((T, d))
                row += f"{v*100:>7.0f}% {d:>+5.0f}"
        L.append(row)
    L.append("")
    weak = [t for t in TARGETS if brate(t) < 0.90]
    for nm, grp in (("ALL", TARGETS), ("WEAK", weak), ("STRONG", [t for t in TARGETS if t not in weak])):
        row = f"  {nm:<7} mean Δ "
        for tag, lab in cells:
            ds = [d for t, d in acc[tag] if t in grp]
            row += f"{lab}: {np.mean(ds):+5.1f}   " if ds else f"{lab}: —   "
        L.append(row)
    L.append("")
    L.append("  PAIRED CONTRASTS (same targets both sides):")
    def contrast(a, b, why):
        ta = dict(acc[a]); tb = dict(acc[b])
        both = [t for t in ta if t in tb]
        if not both:
            return f"    {why}: —"
        d = np.mean([ta[t] - tb[t] for t in both])
        return f"    {why}: {d:+.1f}pp  (n={len(both)})"
    L.append(contrast("b075_a025", "b100_a025", "un-protecting the target (0.75 vs 1.00, donor fixed 0.25)"))
    L.append(contrast("b050_a050", "b100_a100", "halving the whole scale  (0.5/0.5 vs 1.0/1.0, same ratio)"))
    L.append(contrast("b100_a100", "b100_a025", "quadrupling donor dose   (1.00 vs 0.25, target fixed 1.0)"))
    L.append(contrast("b100_a050_fresh", "b100_a025", "donor 0.50 vs 0.25 (both fresh, unbiased)"))
    L.append(contrast("b100_a050_fresh", "b100_a100", "donor 0.50 vs 1.00 (both fresh, unbiased)"))
    L.append(contrast("__probe__", "b100_a050_fresh", "WINNER'S CURSE: probe cell minus fresh re-measure"))
    open(REPO / "results/analysis/dosing_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
