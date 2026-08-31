#!/usr/bin/env python3
"""Q5: ASYMMETRIC interference-resolution on PARTNER residuals only.

User steer: "protect the target more — we are improving the target, not doing multitask merging."
So the target's residual is never touched (coefficient 1.0, dense); only the partners' residuals are
sparsified before admission:

    W = S_16 + 1.0 * r_target + alpha * mean_p( OP(r_p) )

  OP = topq(q)      : keep only the top-q SINGULAR components of the partner residual.

DESIGN NOTE — why not entrywise DARE/TIES. We validated both first and they fail here for structural
reasons: (1) an entrywise-sparsified residual is NO LONGER LOW-RANK, and our adapters must ship as LoRA
factors, so it must be re-encoded — a rank-64 re-encode loses 44% (trim20) to 88% (dare90) of the
sparsified signal, i.e. the operator we would deploy is not the operator we designed; (2) DARE's 1/(1-p)
rescaling INFLATES the residual norm (0.81 -> 1.14 at p=.5, -> 2.55 at p=.9), so those arms would inject
far more energy than the dense reference — exactly the dosing confound flagship-2 exposed. Spectral
truncation keeps the same intent ("transmit the partner's dominant structure, drop its idiosyncratic
tail"), is low-rank by construction (zero re-encode loss), and NEVER increases norm. Measured energy
kept: q=2 -> 0.54, q=4 -> 0.69, q=8 -> 0.86, q=16 -> 0.99 of the partner residual.
Arms are ENERGY-MATCHED to the dense reference by rescaling each truncated partner residual back to its
original norm, so the comparison isolates STRUCTURE (which directions) from DOSE (how much).

Motivation from the all-76 screen: the recipe's failure is VARIANCE (partner draw swings the outcome
by 11pp). If most of a partner's residual is task-specific noise from the target's perspective, keeping
only its largest components may transmit the useful part with less damage.

Arms (alpha=0.5, k=5 partners = best flagship config, SAME fixed draw as the all-76 run so the dense
reference e23 is exactly comparable): top2 / top4 / top8 singular components per partner residual.
Targets: the 5 flagship targets.
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
ALPHA = 0.5
TARGETS = [6, 19, 50, 15, 56]
ARMS = [("top2", 2), ("top4", 4), ("top8", 8)]   # keep top-q singular components of each partner residual
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
# SAME partner draw as the all-76 screen, so the dense reference is exactly comparable
PARTNERS = {T: sorted(int(x) for x in np.random.default_rng(1000 + T).choice(
    [i for i in range(76) if i != T], 5, replace=False)) for T in TARGETS}
_cache = {}


def load_fold(t):
    if t in _cache:
        return _cache[t]
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


def truncate(Rb, Ra, q):
    """Top-q singular components of the residual given as factors Rb @ Ra, returned as factors.
    Energy-matched: rescaled back to the ORIGINAL residual norm so arms differ in STRUCTURE, not dose."""
    Qb, Rbb = torch.linalg.qr(Rb, mode="reduced")
    Qa, Raa = torch.linalg.qr(Ra.T, mode="reduced")
    M = Rbb @ Raa.T
    Uu, Ss, Vt = torch.linalg.svd(M, full_matrices=False)
    full = float((Ss ** 2).sum()) ** 0.5
    kept = float((Ss[:q] ** 2).sum()) ** 0.5
    scale = (full / kept) if kept > 1e-9 else 1.0            # energy match
    B = (Qb @ Uu[:, :q]) * (Ss[:q] * scale)
    A = (Qa @ Vt[:q].T).T
    return B, A


def build(name, target, arm, q):
    """All terms stay low-rank: the truncated partner residuals are emitted directly as factors."""
    ABs = {t: load_fold(t) for t in [target] + PARTNERS[target]}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        b, a = ABs[target][stem]                      # target residual: dense, untouched
        M = U.T @ b
        N = a @ V
        Bb += [b, -(U @ M)]
        Ab += [a, N @ V.T]
        c = ALPHA / len(PARTNERS[target])
        for p in PARTNERS[target]:                     # each partner: residual -> top-q -> add
            bp, ap = ABs[p][stem]
            Mp = U.T @ bp
            Np = ap @ V
            Rb = torch.cat([bp, -(U @ Mp)], 1)         # residual as factors: Rb @ Ra
            Ra = torch.cat([ap, Np @ V.T], 0)
            B, A = truncate(Rb, Ra, q)
            Bb += [c * B]
            Ab += [A]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def already_done(job, sc):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return False
    ok = {json.loads(l)["scene"] for l in open(f) if json.loads(l).get("status") == "ok"}
    return set(sc) <= ok


def gpus_free():
    return subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                          capture_output=True).returncode != 0


def report():
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]

    def tr(sr, t):
        ss = [s for s in scenes[t] if s in sr]
        return sum(sr[s] for s in ss) / len(ss) if ss else None

    R = {}
    for pat in ("e23_*.jsonl", "e24_*.jsonl"):
        for f in (REPO / "results/merge").glob(pat):
            for ln in open(f):
                d = json.loads(ln)
                if d.get("status") == "ok":
                    R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    L = ["=== Q5: asymmetric SPECTRAL truncation of PARTNER residuals (target untouched) ===",
         "W = S_16 + 1.0*r_target + 0.5*mean_p(top-q(r_p)), energy-matched; partners = all-76 draw", "",
         f"{'arm':<12}" + "".join(f"{'t'+str(T):>7}" for T in TARGETS) + f"{'meanD':>9}"]
    for lab, tmpl in [("dense (ref)", "e23_t{T}")] + [(a, "e24_t{T}_" + a) for a, _ in ARMS]:
        row = f"{lab:<12}"
        ds = []
        for T in TARGETS:
            v = tr(R.get(tmpl.format(T=T), {}), T)
            row += f"{'' if v is None else f'{v*100:.0f}%':>7}"
            if v is not None:
                ds.append((v - tr(base, T)) * 100)
        row += f"{np.mean(ds):>+8.1f}" if ds else ""
        if ds:
            row += f"   (sd {np.std(ds, ddof=1):.1f})"
        L.append(row)
    open(REPO / "results/analysis/dare_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def main():
    print("[stage] Q5 queued; waiting for GPUs", flush=True)
    while not gpus_free():
        time.sleep(120)
    print("[stage] starting", flush=True)
    for T in TARGETS:
        jobs, built = [], []
        for arm, param in ARMS:
            job = f"e24_t{T}_{arm}"
            if already_done(job, scenes[T]):
                continue
            path = build(f"q5_{job}", T, arm, param)
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        if not jobs:
            continue
        planp = REPO / "sweep/_q5_current.json"
        json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] t{T} — {len(jobs)} arms", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
    report()


if __name__ == "__main__":
    main()
