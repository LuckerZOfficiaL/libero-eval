#!/usr/bin/env python3
"""EXP-33 — is there anything for a LAYERWISE coefficient to find? (fixed patterns, no learning)

Before spending REINFORCE rollouts on a 5-dimensional action space (v2), ask the cheaper question:
does NON-UNIFORM dosing across layer bands beat uniform dosing at all? If uniform is already optimal
along this axis, a learned layerwise coefficient has nothing to discover and v2's budget is better
spent elsewhere. This is the same go/no-go logic that saved three GPU-days on synthetic targets.

PATTERNS (5 layer bands: vision / llm-early / llm-late / expert / action):
    uniform        (1,1,1,1,1)      the current operator, alpha=0.5 everywhere
    front          (2,2,.5,.5,.5)   dose the perceptual front end
    back           (.5,.5,2,2,2)    dose the action-producing back end
    expert_only    (0,0,0,1,0)      only the action expert receives the donor
    vision_only    (1,0,0,0,0)      only the vision tower receives it

ENERGY MATCHING IS ESSENTIAL HERE, for the same reason it was for external donors: these patterns
inject different total norms, and dose alone moves results by ~9pp (Q1). Each pattern is therefore
rescaled so the TOTAL injected Frobenius norm equals that of uniform alpha=0.5. What varies is WHERE
the donor's residual goes, not how much of it arrives.

Donor = each target's top-1 probe pick, so the comparison is on a donor already known to be useful.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import hostcfg
from reinforce_train import group_of, fold, stems, USIG, VS, scenes, afile, target_baseline

K = 16
ALPHA = 0.5
ROLLOUTS = 25
TARGETS = [6, 19, 50, 15, 56]
PATTERNS = {
    "uniform":     np.array([1., 1., 1., 1., 1.]),
    "front":       np.array([2., 2., .5, .5, .5]),
    "back":        np.array([.5, .5, 2., 2., 2.]),
    "expert_only": np.array([0., 0., 0., 1., 0.]),
    "vision_only": np.array([1., 0., 0., 0., 0.]),
}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META = hostcfg.apply(META)
CHUNK = max(6, 2 * META["n_workers"])
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
DONOR = {t: RANK[str(t)][0][0] for t in TARGETS}



def _clear_orphaned_builds():
    """Adapters in decomp_adapters/ are build->eval->delete transients. If no orchestrator is running,
    any that exist were abandoned by a killed run -- and 30+ of them is a blown disk quota (happened
    twice). Clear them at startup rather than trusting every kill path to clean up."""
    import subprocess as _sp
    if _sp.run(["pgrep", "-f", "merge_orchestrat[o]r.py"], capture_output=True).returncode != 0:
        for _f in (REPO / "results/analysis/decomp_adapters").glob("*.safetensors"):
            _f.unlink(missing_ok=True)


_clear_orphaned_builds()

def group_energies(donor):
    """||r_donor restricted to each band||^2 — needed to rescale a pattern to a fixed total norm."""
    f = fold(donor)
    e = np.zeros(5)
    for stem in stems:
        if stem not in f:
            continue
        b, a = f[stem]
        D = b @ a
        Un = USIG[stem] / torch.linalg.norm(USIG[stem], dim=0).clamp(min=1e-12)
        r = D - Un @ (Un.T @ D @ VS[stem]) @ VS[stem].T
        e[group_of(stem)] += float((r ** 2).sum())
    return e


def matched(pattern, energies):
    """Scale a pattern so sum_g (c_g^2 * E_g) equals the uniform alpha=0.5 case."""
    target = (ALPHA ** 2) * energies.sum()
    got = float((pattern ** 2 * energies).sum())
    if got <= 0:
        return pattern * 0.0
    return pattern * float(np.sqrt(target / got))


def build(name, target, donor, coefs):
    ft, fd = fold(target), fold(donor)
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        c = float(coefs[group_of(stem)])
        Bb, Ab = [Usig], [V.T]
        for f, k in ((ft, 1.0), (fd, c)):
            if stem not in f or k == 0.0:
                continue
            b, a = f[stem]
            M = U.T @ b
            N = a @ V
            Bb += [k * b, -k * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def get(job, t):
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


def main():
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(90)
    coefs = {}
    for t in TARGETS:
        E = group_energies(DONOR[t])
        coefs[t] = {nm: matched(p, E) for nm, p in PATTERNS.items()}
        print(f"[energy] t{t} donor t{DONOR[t]} band energies "
              f"{np.round(E / E.sum(), 3).tolist()}", flush=True)
    specs = [(f"e45_t{t}_{nm}", t, nm) for t in TARGETS for nm in PATTERNS]
    specs = [s for s in specs if get(s[0], s[1]) is None]
    for i in range(0, len(specs), CHUNK):
        jobs, built = [], []
        for job, t, nm in specs[i:i + CHUNK]:
            p = build(f"lw_{job}", t, DONOR[t], coefs[t][nm])
            built.append(p)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": scenes[t]})
        planp = REPO / "sweep/_lw_current.json"
        json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] chunk {i//CHUNK+1}/{(len(specs)+CHUNK-1)//CHUNK}", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        report(coefs)
    report(coefs)


def report(coefs):
    from scipy import stats
    names = list(PATTERNS)
    L = ["=== EXP-33: does NON-UNIFORM layer dosing beat uniform? (fixed patterns, energy-matched) ===",
         "bands: vision / llm-early / llm-late / expert / action; each pattern rescaled to the",
         "uniform alpha=0.5 injected norm, so only WHERE the donor lands varies", "",
         f"{'target':<8}{'donor':>7}{'base':>7}" + "".join(f"{n:>13}" for n in names)]
    acc = {n: [] for n in names}
    for t in TARGETS:
        row = f"t{t:<7}{'t'+str(DONOR[t]):>7}{target_baseline(t)*100:>6.0f}%"
        for n in names:
            v = get(f"e45_t{t}_{n}", t)
            if v is None:
                row += f"{'—':>13}"
            else:
                d = (v - target_baseline(t)) * 100
                acc[n].append(d)
                row += f"{v*100:>7.0f}% {d:>+4.0f}"
        L.append(row)
    L.append("")
    for n in names:
        if acc[n]:
            L.append(f"  {n:<12} mean {np.mean(acc[n]):+6.1f}pp  (n={len(acc[n])})")
    if len(acc["uniform"]) >= 3:
        best = max((n for n in names if n != "uniform"),
                   key=lambda n: np.mean(acc[n]) if acc[n] else -99)
        if acc[best]:
            k = min(len(acc["uniform"]), len(acc[best]))
            d = np.array(acc[best][:k]) - np.array(acc["uniform"][:k])
            t_ = stats.ttest_1samp(d, 0)
            L += ["", f"  best non-uniform ({best}) vs uniform: {d.mean():+.1f}pp "
                      f"(SE {stats.sem(d):.1f}, p={t_.pvalue:.3f}, n={k})",
                  "  >0 => a layerwise coefficient has something to find; v2 is justified.",
                  "  ~0 => uniform is already optimal along this axis; skip v2."]
    open(REPO / "results/analysis/layerwise_probe_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
