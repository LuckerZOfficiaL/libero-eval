#!/usr/bin/env python3
"""E3 — is affinity RECIPROCAL? Is B a good donor for A when A is a good donor for B?

Offline from last week's 604 probes the answer is already "no": among 25 reciprocal pairs the raw
correlation is r = -0.524, but that is a CEILING ARTIFACT (a 100%-baseline task can only lose, a 64%
task can only gain, so any mismatched pair is forced anti-correlated). Restricted to the 17 pairs whose
baselines are within 10pp, r = -0.005 -- no reciprocity in either direction.

n=17 is thin, so this run measures it by design instead of by accident:

  * Only tasks WITH HEADROOM are eligible (baseline <= CEIL). On a saturated task affinity is bounded
    at ~0 and reciprocity is not measurable even in principle -- including such pairs is what produced
    the spurious -0.52.
  * Pairs are sampled among eligible tasks, and BOTH directions are probed with the same operator
    W = S_16 + 1.0*r_target + 0.5*r_donor at 10 rollouts.
  * Directions already measured in e27 are REUSED, so we only pay for the missing halves.

Why it matters: a cosine/distance in any latent space is symmetric by construction. If affinity is not
reciprocal, no metric can represent it regardless of the embedding -- which is what routes the design to
a learned asymmetric model rather than "compute alphas from a latent cosine".
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
ROLLOUTS = 10
CEIL = 0.96              # eligibility: need room for affinity to be measurable in BOTH directions
N_PAIRS = 30
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
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
base = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    r = json.loads(ln)
    if r.get("status") == "ok":
        base[r["scene"]] = r["success_rate"]
_cache = {}


def brate(t):
    ss = [s for s in scenes[t] if s in base]
    return sum(base[s] for s in ss) / len(ss)


def known():
    """(target, donor) -> affinity in pp, from e27 probes."""
    out = {}
    for t, v in RANK.items():
        T = int(t)
        for p, s in v:
            out[(T, int(p))] = (s - brate(T)) * 100
    return out


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


def build(name, target, donor):
    ABs = {t: load_fold(t) for t in (target, donor)}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in ((target, 1.0), (donor, ALPHA)):
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


def aff(T, p, K_):
    """affinity(T,p) in pp -- from this run if measured here, else from e27."""
    v = rate(f"e29_sym_t{T}_p{p}", T)
    if v is not None:
        return (v - brate(T)) * 100
    return K_.get((T, p))


def choose_pairs():
    K_ = known()
    elig = [t for t in range(76) if brate(t) <= CEIL]
    rng = np.random.default_rng(31)
    # prefer pairs where ONE direction is already measured (half price), then fill with fresh pairs
    half = [(a, b) for (a, b) in K_ if a in elig and b in elig and a < b and (b, a) not in K_]
    rng.shuffle(half)
    pairs = half[:N_PAIRS]
    while len(pairs) < N_PAIRS:
        a, b = sorted(rng.choice(elig, 2, replace=False).tolist())
        if a != b and (a, b) not in pairs:
            pairs.append((a, b))
    return pairs, elig, K_


def gpus_free():
    return subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                          capture_output=True).returncode != 0


def main():
    pairs, elig, K_ = choose_pairs()
    print(f"[plan] {len(elig)} eligible tasks (baseline <= {CEIL}); {len(pairs)} pairs", flush=True)
    todo = []
    for a, b in pairs:
        for T, p in ((a, b), (b, a)):
            if aff(T, p, K_) is None:
                todo.append((T, p))
    print(f"[plan] {len(todo)} directions to measure ({2*len(pairs)-len(todo)} reused from e27)", flush=True)
    print("[stage] waiting for GPUs", flush=True)
    while not gpus_free():
        time.sleep(120)
    print("[stage] starting", flush=True)
    for i in range(0, len(todo), 4):
        jobs, built = [], []
        for T, p in todo[i:i + 4]:
            path = build(f"sym_t{T}_p{p}", T, p)
            built.append(path)
            jobs.append({"name": f"e29_sym_t{T}_p{p}", "method": "average", "params": {},
                         "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[T]})
        planp = REPO / "sweep/_sym_current.json"
        json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] {[j['name'] for j in jobs]}", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        print(f"[done] {i+len(jobs)}/{len(todo)}", flush=True)
    report(pairs, K_)


def report(pairs, K_):
    from scipy import stats
    ab, ba, rows = [], [], []
    for a, b in pairs:
        x, y = aff(a, b, K_), aff(b, a, K_)
        if x is None or y is None:
            continue
        ab.append(x); ba.append(y); rows.append((a, b, x, y))
    ab, ba = np.array(ab), np.array(ba)
    L = ["=== E3: IS AFFINITY RECIPROCAL? (designed matched-headroom pairs, both directions) ===",
         f"operator W = S_16 + 1.0*r_target + {ALPHA}*r_donor, {ROLLOUTS} rollouts, "
         f"eligibility baseline <= {CEIL}", ""]
    if len(ab) > 2:
        r, p = stats.pearsonr(ab, ba)
        rs, ps = stats.spearmanr(ab, ba)
        L += [f"  n = {len(ab)} pairs",
              f"  pearson  r = {r:+.3f}  (p = {p:.4f})",
              f"  spearman r = {rs:+.3f}  (p = {ps:.4f})",
              f"  sign agreement: {int((np.sign(ab) == np.sign(ba)).sum())}/{len(ab)} "
              f"= {100*np.mean(np.sign(ab) == np.sign(ba)):.0f}%  (chance ~50%)",
              f"  mean |a->b  minus  b->a| = {np.abs(ab-ba).mean():.1f}pp", "",
              "  INTERPRETATION: r ~ +1 => affinity is a distance, a latent cosine can produce alphas.",
              "                  r ~  0 => not reciprocal; no symmetric metric can represent it.", "",
              "  most asymmetric pairs:"]
        for a, b, x, y in sorted(rows, key=lambda z: -abs(z[2]-z[3]))[:8]:
            L.append(f"    t{a}<->t{b}:  t{a}->t{b} {y:+5.0f}pp   t{b}->t{a} {x:+5.0f}pp   gap {abs(x-y):.0f}")
    else:
        L.append("  insufficient completed pairs")
    open(REPO / "results/analysis/symmetry_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
