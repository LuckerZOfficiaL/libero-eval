#!/usr/bin/env python3
"""FLAGSHIP (week 24-08): asymmetric single-beneficiary merge operator.

Project goal: given a NEW LoRA + a pool, merge pool LoRAs INTO it to improve ITS performance.
Because there is only one beneficiary, the operator can be asymmetric — unlike every symmetric
operator tested so far (which had to dilute everyone's residual equally):

    W(target T, partners P, alpha) = S_16  +  1.0 * r_T  +  alpha * (mean over P of r_p)

  S_16   : the pool's shared component (Phase B proved it is interchangeable and FREE — for t6 it
           was +8pp on its own), so the target inherits pool knowledge at no cost.
  1.0*r_T: the target keeps its OWN residual at FULL strength (what plain averaging and every prior
           operator sacrificed to 1/N — the thing that kills fragile tasks).
  alpha  : how much partner residual to admit. alpha=0 is Phase-B interchange.

Partner strategies: 'twin'  = the target's best BENIGN twin (top residual cosine, confusable pairs
                              excluded per the week-17 merge-safety rules)
                    'pool'  = mean residual of all 75 others (selection-free)
Build -> eval (25 rollouts) -> DELETE per policy (adapters are ~750MB).
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
ARMS = [("base", 0.0)] + [("twin", a) for a in (0.25, 0.5, 1.0)] + [("rand5", a) for a in (0.25, 0.5)]
RNG = np.random.default_rng(0)   # fixed partner draw per target, logged in the report

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
_cache = {}
RAND5 = {}
_rng = np.random.default_rng(0)
for _T in TARGETS:
    _pool = [i for i in range(76) if i != _T and i != PARTNER[_T]]
    RAND5[_T] = sorted(int(x) for x in _rng.choice(_pool, 5, replace=False))


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


def resid_blocks(ABs, t, stem, U, V, coef):
    """factor blocks for coef * (D_t - P(D_t))"""
    b, a = ABs[t][stem]
    M = U.T @ b
    N = a @ V
    return [coef * b, -coef * (U @ M)], [a, N @ V.T]


def build(name, target, partners, alpha):
    ABs = {target: load_fold(target)}
    for p in partners:
        ABs[p] = load_fold(p)
    npart = max(1, len(partners))
    tensors = {}
    for stem in stems:
        Usig = US[stem]
        sig = torch.linalg.norm(Usig, dim=0)
        U = Usig / sig.clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]                       # S_16 = U Sigma V^T
        b, a = resid_blocks(ABs, target, stem, U, V, 1.0)
        Bb += b; Ab += a                             # own residual, full strength
        for p in partners:
            b, a = resid_blocks(ABs, p, stem, U, V, alpha / npart)
            Bb += b; Ab += a
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
    for f in (REPO / "results/merge").glob("e20_*.jsonl"):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    hdr = f"{'target':<8}{'base':>6}" + "".join(f"{(s+str(a)):>12}" for s, a in ARMS)
    lines = ["=== FLAGSHIP: asymmetric single-beneficiary operator (25 rollouts) ===",
             "W = S_16 + 1.0*r_target + alpha*mean(partner residuals)", "", hdr]
    for T in TARGETS:
        row = f"t{T:<7}{tr(base,T)*100:>5.0f}%"
        for strat, a in ARMS:
            v = tr(R.get(f"e20_t{T}_{strat}_a{int(a*100)}", {}), T)
            row += f"{'' if v is None else f'{v*100:.0f}%':>12}"
        lines.append(row)
    lines.append("")
    lines.append(f"  rand5 partner draws: {RAND5}")
    for strat, a in ARMS:
        ds = [tr(R.get(f"e20_t{T}_{strat}_a{int(a*100)}", {}), T) - tr(base, T)
              for T in TARGETS if tr(R.get(f"e20_t{T}_{strat}_a{int(a*100)}", {}), T) is not None]
        if ds:
            lines.append(f"  {strat} alpha={a}: mean {np.mean(ds)*100:+.1f}pp  (n={len(ds)})")
    open(REPO / "results/analysis/flagship_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


BATCHES = [ARMS[:4], ARMS[4:]]   # [base + 3 twin arms], then [2 rand5 arms]: keeps <=4 adapters on disk
                                 # AND keeps all 4 workers busy (one-policy-per-plan idled a whole GPU)


def main():
    for T in TARGETS:
        for batch in BATCHES:
            jobs, built = [], []
            for strat, alpha in batch:
                job = f"e20_t{T}_{strat}_a{int(alpha*100)}"
                if already_done(job, scenes[T]):
                    continue
                if strat == "base":
                    partners = []
                elif strat == "twin":
                    partners = [PARTNER[T]]
                else:                      # rand5: 5 random partners. (mean of ALL 76 residuals ~= 0 —
                    partners = RAND5[T]    # measured 1.4% of one residual's norm — that arm was a no-op)
                path = build(f"fs_t{T}_{strat}_a{int(alpha*100)}", T, partners, alpha)
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": scenes[T]})
            if not jobs:
                continue
            planp = REPO / "sweep/_fs_current.json"
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
