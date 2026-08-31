#!/usr/bin/env python3
"""BEHAVIOURAL PROBE SELECTOR — turn the asymmetric operator into a working method.

Every static selector feature failed (residual cosine; donor history LOOCV R^2 -0.075; activation
pair distance within-target r -0.043), and partner usefulness is a (target, donor) INTERACTION.
So we buy the interaction behaviourally, as cheaply as possible:

  STAGE 1 (probe, cheap): for each target T and each of 8 candidate partners p, build the pairwise
     operator W = S_16 + 1.0*r_T + 0.5*r_p and evaluate it on T's scenes with only PROBE_ROLLOUTS=3.
     3 rollouts is deliberately far too noisy to *measure* a partner — the bet is that it is enough
     to *rank* them coarsely (separate helpful from harmful).
  STAGE 2 (commit, accurate): take the top-3 probed partners, build the k=3 operator
     W = S_16 + 1.0*r_T + 0.5*mean(top3 residuals), evaluate at 25 rollouts.
     Also evaluate the BOTTOM-3 ("anti-selection") as the control that proves the probe carries signal:
     if top3 >> bottom3, selection works even if the absolute number is modest.

Reference points already measured: baseline; random-5 at alpha=.5 (e23, the all-76 screen);
per-target oracle from the flagship arms.

Cost per target: 8 probe evals @3 + 2 commit evals @25 = 8*3 + 50 = 74 rollouts (vs 8*25=200 for
exhaustive probing). 5 targets -> ~370 rollouts total.
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
PROBE_ROLLOUTS = 10
COMMIT_ROLLOUTS = 25
N_CAND = 8
N_PICK = 3
import json as _j
TARGETS = [6, 19, 50, 15, 56] + _j.load(open(REPO / "sweep/robust_targets.json"))
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
# candidate pools: fixed, seeded, and DISJOINT from nothing — plain random draws so the result is
# about the PROBE, not about a clever candidate list.
CAND = {T: sorted(int(x) for x in np.random.default_rng(2000 + T).choice(
    [i for i in range(76) if i != T], N_CAND, replace=False)) for T in TARGETS}
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


def build(name, target, partners):
    ABs = {t: load_fold(t) for t in [target] + list(partners)}
    npart = max(1, len(partners))
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in [(target, 1.0)] + [(p, ALPHA / npart) for p in partners]:
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


def results_for(job):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return {}
    out = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            out[d["scene"]] = d["success_rate"]
    return out


def rate(job, t):
    sr = results_for(job)
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if ss else None


def run_batch(jobs, n_trials):
    if not jobs:
        return
    planp = REPO / "sweep/_probe_current.json"
    json.dump(dict(META, n_trials=n_trials, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def gpus_free():
    return subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                          capture_output=True).returncode != 0


def main():
    while not gpus_free():
        time.sleep(120)
    picks = {}
    for T in TARGETS:
        # ---------- STAGE 1: cheap probes ----------
        for i in range(0, len(CAND[T]), 4):
            jobs, built = [], []
            for p in CAND[T][i:i + 4]:
                job = f"e26_probe_t{T}_p{p}"
                if rate(job, T) is not None:
                    continue
                path = build(f"pb10_t{T}_p{p}", T, [p])
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": scenes[T]})
            print(f"[probe] t{T}: {len(jobs)} candidates @{PROBE_ROLLOUTS} rollouts", flush=True)
            run_batch(jobs, PROBE_ROLLOUTS)
            for p_ in built:
                pathlib.Path(p_).unlink(missing_ok=True)
        scored = sorted(((rate(f"e26_probe_t{T}_p{p}", T), p) for p in CAND[T]
                         if rate(f"e26_probe_t{T}_p{p}", T) is not None), reverse=True)
        picks[T] = {"scored": [(p, v) for v, p in scored],
                    "top": [p for v, p in scored[:N_PICK]],
                    "bottom": [p for v, p in scored[-N_PICK:]]}
        print(f"[probe] t{T} ranking: {picks[T]['scored']}", flush=True)
        json.dump(picks, open(REPO / "results/analysis/probe_picks10.json", "w"), indent=1)
        # ---------- STAGE 2: commit + anti-selection control ----------
        jobs, built = [], []
        for tag, ps in (("top3", picks[T]["top"]), ("bot3", picks[T]["bottom"])):
            job = f"e26_commit_t{T}_{tag}"
            if rate(job, T) is not None:
                continue
            path = build(f"cm10_t{T}_{tag}", T, ps)
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        print(f"[commit] t{T}: top3={picks[T]['top']} bot3={picks[T]['bottom']}", flush=True)
        run_batch(jobs, COMMIT_ROLLOUTS)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
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

    L = ["=== BEHAVIOURAL PROBE SELECTOR v2: 10-rollout probes; weak AND robust targets ===",
         f"stage 1: {N_CAND} candidates x {PROBE_ROLLOUTS} rollouts (pairwise operator)",
         f"stage 2: top-{N_PICK} vs bottom-{N_PICK} committed at {COMMIT_ROLLOUTS} rollouts", "",
         f"{'target':<8}{'base':>6}{'rand5(ref)':>12}{'TOP3':>8}{'BOT3':>8}{'top-bot':>9}"]
    dt, db = [], []
    for T in TARGETS:
        b = brate(T)
        r5 = rate(f"e23_t{T}", T)
        tp = rate(f"e26_commit_t{T}_top3", T)
        bt = rate(f"e26_commit_t{T}_bot3", T)
        row = f"t{T:<7}{b*100:>5.0f}%"
        row += f"{'' if r5 is None else f'{r5*100:.0f}%':>12}"
        row += f"{'' if tp is None else f'{tp*100:.0f}%':>8}"
        row += f"{'' if bt is None else f'{bt*100:.0f}%':>8}"
        if tp is not None and bt is not None:
            row += f"{(tp-bt)*100:>+8.0f}"
            dt.append((tp - b) * 100)
            db.append((bt - b) * 100)
        L.append(row)
    WEAK = [6, 19, 50, 15, 56]
    for nm, grp in (("WEAK (headroom)", WEAK), ("ROBUST (>=96%, safety)", [t for t in TARGETS if t not in WEAK])):
        d1 = [(rate(f"e26_commit_t{T}_top3", T) - brate(T)) * 100 for T in grp
              if rate(f"e26_commit_t{T}_top3", T) is not None]
        d0 = [(rate(f"e26_commit_t{T}_bot3", T) - brate(T)) * 100 for T in grp
              if rate(f"e26_commit_t{T}_bot3", T) is not None]
        if d1:
            L.append("")
            L.append(f"  {nm}: TOP3 {np.mean(d1):+.1f}pp   BOT3 {np.mean(d0):+.1f}pp"
                     f"   selection effect {np.mean(d1)-np.mean(d0):+.1f}pp   (n={len(d1)})")
            L.append(f"    per-target TOP3 delta: {[round(x) for x in d1]}")
    L.append("")
    L.append(f"  probe rankings: {json.dumps({str(k): v['scored'] for k, v in picks.items()})}")
    open(REPO / "results/analysis/probe_selector10_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
