#!/usr/bin/env python3
"""EXP-24 — ITERATION: does probe-selected merging COMPOSE?

Every result so far is one-shot: baseline LoRA -> one probe round -> one merge. This asks whether the
method is a LOOP. Round-1 winners still have headroom (t19 84%, t56 80%, t6 84%, t50 76%, t15 80%);
treat each round-1 merged policy as the NEW target and run a second, identical probe round on top.

Round-2 target residual (everything stays linear in the factors):
    r' = r_T + (ALPHA/3) * (r_a + r_b + r_c)          for round-1 picks {a,b,c}
Round-2 probe:   W = S_16 + r' + ALPHA * r_cand      (8 fresh candidates, 10 rollouts)
Round-2 commit:  W = S_16 + r' + (ALPHA/3) * sum(top-3)   at 25 rollouts
GUARD: commit only if the best round-2 probe beats the round-1 committed score (else the answer for
that target is "iteration abstains", which is itself data).

Reads: round-2 committed score vs round-1 committed score, per target and mean. Climbing => the method
is a hill-climber and "keep merging while the probe finds improvers" becomes the deployment story.
Plateau => one round extracts what the pool offers; also worth knowing.

Candidates: 8 fresh donors per target, excluding the target, its twin, and round-1 picks; drawn from
the top of the hypernet donor-prior ranking? NO — drawn RANDOMLY (seeded), so round 2 tests the same
mechanism round 1 used, not a new selector.
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
PROBE_R = 10
COMMIT_R = 25
TWIN = {7: 18, 18: 7, 29: 62, 62: 29}
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
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
TARGETS = [19, 56, 6, 50, 15]
R1_PICKS = {T: [p for p, _ in RANK[str(T)][:3]] for T in TARGETS}
rng = np.random.default_rng(88)
CANDS = {}
for T in TARGETS:
    used = {T, TWIN.get(T, -1), *R1_PICKS[T]}
    CANDS[T] = sorted(int(x) for x in rng.choice([c for c in range(76) if c not in used],
                                                 8, replace=False))
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
    return _cache.setdefault(t, out)


def build(name, terms):
    """W = S_16 + sum(c * r_t) as LoRA factors; terms = [(task, coeff), ...]."""
    folds = {t: load_fold(t) for t, _ in terms}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in terms:
            b, a = folds[t][stem]
            M = U.T @ b
            N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def r2_terms(T):
    return [(T, 1.0)] + [(p, ALPHA / 3) for p in R1_PICKS[T]]


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


def run_plan(jobs, n_trials):
    if not jobs:
        return
    planp = REPO / "sweep/_iter_current.json"
    json.dump(dict(META, n_trials=n_trials, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(120)
    for T in TARGETS:
        jobs, built = [], []
        for c in CANDS[T]:
            job = f"e34_r2probe_t{T}_p{c}"
            if rate(job, T) is not None:
                continue
            path = build(f"r2p_t{T}_p{c}", r2_terms(T) + [(c, ALPHA)])
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        if jobs:
            print(f"[probe2] t{T}: {len(jobs)} candidates", flush=True)
            run_plan(jobs, PROBE_R)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        # guard + commit
        r1 = rate(f"e27_commit_t{T}_top3", T)
        s2 = sorted(((rate(f"e34_r2probe_t{T}_p{c}", T), c) for c in CANDS[T]
                     if rate(f"e34_r2probe_t{T}_p{c}", T) is not None), reverse=True)
        top2 = [c for _, c in s2[:3]]
        best = s2[0][0] if s2 else None
        print(f"[select2] t{T}: r1={r1*100:.0f}%  best probe {best*100:.0f}%  top3 {top2}"
              f"  -> {'COMMIT' if best and best > r1 else 'ABSTAIN'}", flush=True)
        if best and best > r1 and rate(f"e34_r2commit_t{T}", T) is None:
            path = build(f"r2c_t{T}", r2_terms(T) + [(p, ALPHA / 3) for p in top2])
            run_plan([{"name": f"e34_r2commit_t{T}", "method": "average", "params": {},
                       "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[T]}], COMMIT_R)
            pathlib.Path(path).unlink(missing_ok=True)
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

    L = ["=== EXP-24: DOES PROBE-SELECTED MERGING COMPOSE? (round 2 on top of round-1 winners) ===",
         "round-2 target = round-1 merged policy; fresh random candidates; guard: commit only if",
         "best round-2 probe beats the round-1 committed score", "",
         f"{'target':<8}{'base':>7}{'round1':>8}{'round2':>8}{'r2-r1':>7}{'verdict':>10}"]
    d = []
    for T in TARGETS:
        b, r1 = brate(T), rate(f"e27_commit_t{T}_top3", T)
        r2 = rate(f"e34_r2commit_t{T}", T)
        row = f"t{T:<7}{b*100:>6.0f}%{r1*100:>7.0f}%"
        if r2 is None:
            row += f"{'—':>8}{'—':>7}{'abstained':>10}"
        else:
            d.append((r2 - r1) * 100)
            row += f"{r2*100:>7.0f}%{(r2-r1)*100:>+7.0f}{'climbed' if r2 > r1 else 'flat/down':>10}"
        L.append(row)
    if d:
        L += ["", f"  committed targets: round2 - round1 = {np.mean(d):+.2f}pp (n={len(d)})"]
    L += ["", "  climbing => the method is a HILL-CLIMBER (keep merging while the probe finds improvers)",
          "  plateau  => one round extracts what the pool offers"]
    open(REPO / "results/analysis/iterate_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
