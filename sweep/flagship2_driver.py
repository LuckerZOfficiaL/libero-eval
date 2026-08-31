#!/usr/bin/env python3
"""FLAGSHIP-2: energy-matched partner sweep (resolves a confound in flagship-1).

Flagship-1 compared 'twin' (1 partner, coefficient alpha) against 'rand5' (mean of 5, so each
partner gets alpha/5). Because near-orthogonal residuals add in quadrature, the INJECTED residual
energy is alpha for twin but alpha/sqrt(5)=0.45*alpha for rand5 — i.e. the rand5 arms were ~2x
gentler, which alone could explain why they looked better. This run matches injected energy
E = alpha / sqrt(k) across partner counts k in {1, 5, 20}:

    E = 0.22 :  twin(k=1) a=0.22 | rand5 a=0.50 [done] | rand20 a=1.00
    E = 0.45 :  twin(k=1) a=0.45 [~= flagship a=0.5, done] | rand5 a=1.00 | rand20 a=2.00

Also re-runs the two winning cells of flagship-1 at a fresh 25 rollouts (winner's-curse check):
t6 twin a=0.25 (88%) and t19 rand5 a=0.25 (84%).
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al

OUTD = REPO / "results/analysis/decomp_adapters"
OUTD.mkdir(parents=True, exist_ok=True)
K = 16
TARGETS = [6, 19, 50, 15, 56]
PARTNER = {int(k): v for k, v in json.load(open(REPO / "sweep/flagship_partners.json")).items()}
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
US = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
_rng = np.random.default_rng(0)
RAND = {}
for _T in TARGETS:
    _pool = [i for i in range(76) if i != _T and i != PARTNER[_T]]
    r5 = sorted(int(x) for x in _rng.choice(_pool, 5, replace=False))          # same draw as flagship-1
    r20 = sorted(int(x) for x in np.random.default_rng(100 + _T).choice(_pool, 20, replace=False))
    RAND[_T] = {"rand5": r5, "rand20": r20}
# (strategy, alpha) -> injected energy alpha/sqrt(k)
ARMS = [("twinE", 0.22), ("rand20", 1.00),           # E = 0.22 band (rand5 a=0.5 already done)
        ("rand5H", 1.00), ("rand20H", 2.00)]         # E = 0.45 band (twin a=0.5 already done)
RERUN = {6: ("twin", 0.25), 19: ("rand5", 0.25)}     # winner's-curse re-checks
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


def build(name, target, partners, alpha):
    ABs = {target: load_fold(target)}
    for p in partners:
        ABs[p] = load_fold(p)
    npart = max(1, len(partners))
    tensors = {}
    for stem in stems:
        Usig = US[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in [(target, 1.0)] + [(p, alpha / npart) for p in partners]:
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


def partners_for(T, strat):
    if strat == "twinE":
        return [PARTNER[T]]
    if strat in ("rand5H",):
        return RAND[T]["rand5"]
    if strat in ("rand20", "rand20H"):
        return RAND[T]["rand20"]
    return [PARTNER[T]] if strat == "twin" else RAND[T]["rand5"]


def already_done(job, sc):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return False
    ok = {json.loads(l)["scene"] for l in open(f) if json.loads(l).get("status") == "ok"}
    return set(sc) <= ok


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
    for pat in ("e20_*.jsonl", "e21_*.jsonl"):
        for f in (REPO / "results/merge").glob(pat):
            for ln in open(f):
                d = json.loads(ln)
                if d.get("status") == "ok":
                    R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    # energy-matched view: (label, job-template, injected E)
    VIEW = [("twin k=1  a=0.22", "e21_t{T}_twinE_a22", 0.22),
            ("rand5 k=5 a=0.50", "e20_t{T}_rand5_a50", 0.22),
            ("rand20 k=20 a=1.0", "e21_t{T}_rand20_a100", 0.22),
            ("twin k=1  a=0.50", "e20_t{T}_twin_a50", 0.45),
            ("rand5 k=5 a=1.00", "e21_t{T}_rand5H_a100", 0.45),
            ("rand20 k=20 a=2.0", "e21_t{T}_rand20H_a200", 0.45)]
    lines = ["=== FLAGSHIP-2: ENERGY-MATCHED partner sweep (25 rollouts) ===",
             "injected residual energy E = alpha / sqrt(k)", ""]
    lines.append(f"{'arm':<20}{'E':>6}" + "".join(f"{'t'+str(T):>7}" for T in TARGETS) + f"{'meanD':>9}")
    for lab, tmpl, E in VIEW:
        row = f"{lab:<20}{E:>6.2f}"
        ds = []
        for T in TARGETS:
            v = tr(R.get(tmpl.format(T=T), {}), T)
            row += f"{'' if v is None else f'{v*100:.0f}%':>7}"
            if v is not None:
                ds.append((v - tr(base, T)) * 100)
        row += f"{np.mean(ds):>+8.1f}" if ds else f"{'':>9}"
        lines.append(row)
    lines.append("\nwinner's-curse re-checks (fresh 25 rollouts):")
    for T, (s, a) in RERUN.items():
        o = tr(R.get(f"e20_t{T}_{s}_a{int(a*100)}", {}), T)
        n = tr(R.get(f"e21_t{T}_rerun", {}), T)
        if o is not None and n is not None:
            lines.append(f"  t{T} {s} a={a}: first {o*100:.0f}%  re-run {n*100:.0f}%  "
                         f"(baseline {tr(base,T)*100:.0f}%)")
    open(REPO / "results/analysis/flagship2_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    for T in TARGETS:
        specs = [(s, a, f"e21_t{T}_{s}_a{int(a*100)}") for s, a in ARMS]
        if T in RERUN:
            s, a = RERUN[T]
            specs.append((s, a, f"e21_t{T}_rerun"))
        for i in range(0, len(specs), 3):                    # <=3 adapters live at once
            jobs, built = [], []
            for strat, alpha, job in specs[i:i + 3]:
                if already_done(job, scenes[T]):
                    continue
                path = build(f"f2_{job}", T, partners_for(T, strat), alpha)
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": scenes[T]})
            if not jobs:
                continue
            planp = REPO / "sweep/_f2_current.json"
            json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
            print(f"[run] t{T} batch of {len(jobs)}", flush=True)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                            str(planp)], cwd=REPO)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
    report()


if __name__ == "__main__":
    main()
