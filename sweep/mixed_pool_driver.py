#!/usr/bin/env python3
"""EXP-28 — does ADDING foreign adapters to the pool hurt? (and does a capture gate fix it?)

EXP-26 replaced the donor pool with external experts and they failed (-1.6pp vs baseline, versus
+14.4pp for internal donors). The realistic scenario is different: the pool simply GROWS with foreign
adapters, and the probe is supposed to ignore them.

Recomputing selection over the combined 16-candidate pool from probes we already own shows it mostly
does -- internal donors take 12 of 15 top-3 slots -- but not always. Externals take 3 slots, and on t19
`spatial_expert` probed at 100%, topping the whole pool, while its committed score was 80% against the
internal top-3's 84%.

WHY: max-of-N selection is biased toward VARIANCE, and low-capture donors are erratic (their probe
overstatement is +17.2pp vs +5.0pp for internal ones). A foreign adapter can win the probe by being
noisy rather than good. So a bigger pool is not automatically a better pool.

This run commits the MIXED-pool top-3 on the two targets where externals actually entered the
selection (t19, t15) and compares against the internal-only commit already measured in e27:

    t19  mixed [spatial_expert, t28, goal_expert]     vs internal top-3 (84%)
    t15  mixed [t70, t32, 10-goal_expert]             vs internal top-3 (80%)

It also records what a CAPTURE GATE would have done. capture@16 separates the populations perfectly
(internal 0.84-0.87, external 0.19-0.39), costs zero rollouts, and would have excluded every external
candidate before probing -- static geometry cannot RANK donors, but it may be able to GATE them.
"""
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import hostcfg

K = 16
ALPHA = 0.5
COMMIT_R = 25
TARGETS = [19, 15]
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
META = hostcfg.apply(META)          # derive workers/gpus for THIS machine
EXTPATH = {p.split("/")[-3]: p for p in
           glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/adapter_model.safetensors")}
COEF = json.load(open(REPO / "results/analysis/ext_coeffs.json"))
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
_cache = {}


def fold(p):
    key = str(p)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 20:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(p))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return _cache.setdefault(key, out)


def capture16(f):
    num = den = 0.0
    for s in stems:
        if s not in f:
            continue
        b, a = f[s]
        D = b @ a
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        P = Un @ (Un.T @ D @ VS[s]) @ VS[s].T
        num += float((P ** 2).sum()); den += float((D ** 2).sum())
    return num / max(den, 1e-12)


def build(name, terms):
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for f, c in terms:
            if stem not in f:
                continue
            b, a = f[stem]
            M = U.T @ b
            N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
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


def mixed_top3(T):
    """Top-3 over the COMBINED pool, from probe scores already on disk."""
    c = [(s, ("int", p)) for p, s in RANK[str(T)]]
    for nm in EXTPATH:
        v = rate(f"e36_ext_t{T}_{nm}", T)
        if v is not None:
            c.append((v, ("ext", nm)))
    return sorted(c, reverse=True)[:3]


def main():
    jobs, built = [], []
    picks = {}
    for T in TARGETS:
        top = mixed_top3(T)
        picks[T] = top
        terms = []
        for _, (kind, who) in top:
            if kind == "int":
                terms.append((fold(afile[who]), ALPHA / 3))
            else:
                terms.append((fold(EXTPATH[who]), COEF[who] / 3))
        print(f"[mixed] t{T}: {[(k, w) for _, (k, w) in top]}", flush=True)
        job = f"e38_mixed_t{T}"
        if rate(job, T) is None:
            path = build(f"mix_t{T}", [(fold(afile[T]), 1.0)] + terms)
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
    if jobs:
        planp = REPO / "sweep/_mixed_current.json"
        json.dump(dict(META, n_trials=COMMIT_R, jobs=jobs), open(planp, "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
    for p in built:
        pathlib.Path(p).unlink(missing_ok=True)
    report(picks)


def report(picks):
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]

    def brate(t):
        ss = [s for s in scenes[t] if s in base]
        return sum(base[s] for s in ss) / len(ss)

    L = ["=== EXP-28: does ADDING foreign adapters to the donor pool hurt? ===",
         "mixed = top-3 of the combined 16-candidate pool; internal = e27 top-3 (pool-consistent only)", "",
         f"{'target':<8}{'base':>6}{'internal':>10}{'MIXED':>8}{'mixed-int':>11}  picks"]
    d = []
    for T in TARGETS:
        b, i3 = brate(T), rate(f"e27_commit_t{T}_top3", T)
        m3 = rate(f"e38_mixed_t{T}", T)
        row = f"t{T:<7}{b*100:>5.0f}%{'' if i3 is None else f'{i3*100:.0f}%':>10}"
        row += f"{'' if m3 is None else f'{m3*100:.0f}%':>8}"
        if i3 is not None and m3 is not None:
            d.append((m3 - i3) * 100)
            row += f"{(m3-i3)*100:>+11.0f}"
        row += "  " + ", ".join(f"{k}:{w}" for _, (k, w) in picks.get(T, []))
        L.append(row)
    if d:
        L += ["", f"  mixed pool vs pool-consistent: {np.mean(d):+.1f}pp (n={len(d)})"]
    # what a capture gate would have done -- costs zero rollouts
    L += ["", "  CAPTURE GATE (static, no rollouts): capture@16 of each candidate"]
    ic = [capture16(fold(afile[t])) for t in (6, 19, 50)]
    ec = {nm: capture16(fold(p)) for nm, p in list(EXTPATH.items())[:4]}
    L.append(f"    internal sample: {[round(x,3) for x in ic]}")
    L.append(f"    external sample: { {k: round(v,3) for k,v in ec.items()} }")
    L.append(f"    a threshold of 0.6 separates them perfectly and would exclude every external")
    L.append(f"    candidate BEFORE probing -- static geometry cannot RANK donors, but it can GATE them.")
    open(REPO / "results/analysis/mixed_pool_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
