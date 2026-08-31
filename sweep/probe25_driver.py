#!/usr/bin/env python3
"""PHASE 2 — does a LESS NOISY probe select better?

EXP-22.4 measured the probe's test-retest reliability at 10 rollouts: rho = 0.41. The affinity
differences the probe must rank are 10-30pp while its own noise is ~13pp, so selection has been running
at signal-to-noise ~1 and the deployed top-3 is not the best available top-3. That makes every number we
have reported a LOWER bound rather than a limit.

Two independent reasons to expect a real gain from more rollouts per probe:
  * empirical -- raising probes 3 -> 10 rollouts took weak-target gain from +5.6pp to +14.4pp;
  * Spearman-Brown -- 10 -> 25 rollouts should lift reliability 0.41 -> 0.63, and the ceiling on any
    selector from 0.64 -> 0.80.

DESIGN. For each weak target, re-probe the SAME 8 candidates e27 used, now at 25 rollouts, and compare
the resulting top-3 against the 10-rollout top-3.
  * Where the two selections AGREE, nothing needs running: e27's 25-rollout commit already measures it.
  * Where they DIFFER, commit the new top-3 at 25 rollouts. The e27 commit is the paired control, so the
    comparison is like-for-like at the same commit budget -- the only thing that changed is probe quality.
This makes the run cheap in the good case and informative in the bad one: if richer probes never change
the selection, that is itself the answer (the 10-rollout probe was already picking what a 25 would).
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
PROBE_ROLLOUTS = 25
COMMIT_ROLLOUTS = 25
N_PICK = 3
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
# tmpfs build cache is 32G, each merged policy 6.8G -> 4 workers overflow it. See EXPERIMENTS.md.
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
base = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    r = json.loads(ln)
    if r.get("status") == "ok":
        base[r["scene"]] = r["success_rate"]


def brate(t):
    ss = [s for s in scenes[t] if s in base]
    return sum(base[s] for s in ss) / len(ss)


# weak targets = where affinity is measurable and the method actually matters
TARGETS = sorted([t for t in range(76) if brate(t) < 0.90],
                 key=lambda t: brate(t))[:10]
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


def run_batch(jobs, n_trials):
    if not jobs:
        return
    planp = REPO / "sweep/_p25_current.json"
    json.dump(dict(META, n_trials=n_trials, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    print(f"[plan] targets {TARGETS}", flush=True)
    while not gpus_free():
        time.sleep(120)
    for T in TARGETS:
        cands = [p for p, _ in RANK[str(T)]]
        todo = [p for p in cands if rate(f"e31_probe_t{T}_p{p}", T) is None]
        # Submit ALL candidates as ONE plan. Batching them in groups of n_workers inserts a barrier
        # after each group, so the trailing group (8 = 3+3+2) and every within-group straggler leave
        # workers -- and therefore a whole GPU -- idle. One plan lets the orchestrator pull the next
        # job the instant a worker frees up. Peak tmpfs is unchanged: it is bounded by n_workers
        # concurrent policies (3 x 6.8G + 6.8G building = 27.2G), not by the number of queued jobs.
        jobs, built = [], []
        for p in todo:
            path = build(f"p25_t{T}_p{p}", T, [p])
            built.append(path)
            jobs.append({"name": f"e31_probe_t{T}_p{p}", "method": "average", "params": {},
                         "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[T]})
        if jobs:
            print(f"[probe25] t{T}: {len(jobs)} candidates in one plan", flush=True)
            run_batch(jobs, PROBE_ROLLOUTS)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        # selection at 25 rollouts vs the 10-rollout selection
        s25 = sorted(((rate(f"e31_probe_t{T}_p{p}", T), p) for p in cands
                      if rate(f"e31_probe_t{T}_p{p}", T) is not None), reverse=True)
        top25 = [p for _, p in s25[:N_PICK]]
        top10 = [p for p, _ in RANK[str(T)][:N_PICK]]
        print(f"[select] t{T}: 10-roll {top10} -> 25-roll {top25}"
              f"  {'SAME' if set(top10) == set(top25) else 'DIFFERENT'}", flush=True)
        if set(top10) != set(top25) and rate(f"e31_commit_t{T}", T) is None:
            path = build(f"p25c_t{T}", T, top25)
            run_batch([{"name": f"e31_commit_t{T}", "method": "average", "params": {}, "adapters": [],
                        "adapter_paths": [path], "eval_scenes": scenes[T]}], COMMIT_ROLLOUTS)
            pathlib.Path(path).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
        report()
    report()


def report():
    L = ["=== PHASE 2: does a 25-rollout probe select better than a 10-rollout probe? ===",
         "same 8 candidates, same operator, same 25-rollout commit budget; only PROBE quality differs", "",
         f"{'target':<8}{'base':>6}{'10-roll pick':>26}{'25-roll pick':>26}{'same':>7}"
         f"{'e27 commit':>12}{'new commit':>12}{'delta':>8}"]
    same = 0
    n = 0
    d10, d25 = [], []
    for T in TARGETS:
        cands = [p for p, _ in RANK[str(T)]]
        s25 = sorted(((rate(f"e31_probe_t{T}_p{p}", T), p) for p in cands
                      if rate(f"e31_probe_t{T}_p{p}", T) is not None), reverse=True)
        if len(s25) < len(cands):
            continue
        n += 1
        top25 = [p for _, p in s25[:N_PICK]]
        top10 = [p for p, _ in RANK[str(T)][:N_PICK]]
        eq = set(top10) == set(top25)
        same += eq
        c10 = rate(f"e27_commit_t{T}_top3", T)
        c25 = c10 if eq else rate(f"e31_commit_t{T}", T)
        row = (f"t{T:<7}{brate(T)*100:>5.0f}%{str(top10):>26}{str(top25):>26}"
               f"{'yes' if eq else 'NO':>7}")
        row += f"{'' if c10 is None else f'{c10*100:.0f}%':>12}"
        row += f"{'' if c25 is None else f'{c25*100:.0f}%':>12}"
        if c10 is not None and c25 is not None:
            d10.append((c10 - brate(T)) * 100)
            d25.append((c25 - brate(T)) * 100)
            row += f"{(c25-c10)*100:>+8.0f}"
        L.append(row)
    if n:
        L += ["", f"  selection agreement: {same}/{n} targets picked the same top-3"]
    if d10:
        L += [f"  10-rollout selection: {np.mean(d10):+.2f}pp vs baseline  (n={len(d10)})",
              f"  25-rollout selection: {np.mean(d25):+.2f}pp vs baseline",
              f"  GAIN FROM A BETTER PROBE: {np.mean(d25)-np.mean(d10):+.2f}pp"]
    open(REPO / "results/analysis/probe25_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
