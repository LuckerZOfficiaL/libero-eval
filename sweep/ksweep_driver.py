#!/usr/bin/env python3
"""Q3: does the SHARED/RESIDUAL boundary placement (k) matter behaviourally?

k=32 is a mathematical no-op (the soup is exactly rank 16 — directions 17+ have ~zero singular
values, so P_32 == P_16). The meaningful sweep is BELOW saturation, where lowering k moves real
content out of "shared" and into the task's own residual:

  capture@k (fraction of an adapter's energy called "shared"):  k=4 -> 0.55 | 8 -> 0.67 | 16 -> 0.80

Two arms per k (both reuse the saved top-64 soup factors, sliced — no recomputation):
  A 'swap'  : W = S_k + (D_t - P_k(D_t))            the Phase-B interchange at rank k
              -> is the pool's shared component still interchangeable when we take more/less of it?
  B 'op'    : W = S_k + r_k,own + 0.5*mean(rand5 r_k)   the best flagship arm, at rank k
              -> does the operator's benefit depend on where we cut?
k=16 for both arms already exists (e20_*_base_a0 and e20_*_rand5_a50), so only k in {4,8,12} runs
for 'swap' and k in {4,8} for 'op'.

Staged: waits for the GPUs to free (flagship-2) before evaluating; builds are CPU and overlap.
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
TARGETS = [6, 19, 50, 15, 56]
SWAP_KS = [4, 8, 12]
OP_KS = [4, 8]
PARTNER = {int(k): v for k, v in json.load(open(REPO / "sweep/flagship_partners.json")).items()}
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
USIG = {s: torch.from_numpy(z[f"U_{i}"]).float() for i, s in enumerate(stems)}   # U*Sigma, 64 cols
VFULL = {s: torch.from_numpy(z[f"V_{i}"]).float() for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
_rng = np.random.default_rng(0)
RAND5 = {}
for _T in TARGETS:                                   # SAME draw as flagship-1/2 (seed 0, same order)
    _pool = [i for i in range(76) if i != _T and i != PARTNER[_T]]
    RAND5[_T] = sorted(int(x) for x in _rng.choice(_pool, 5, replace=False))
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


def build(name, target, partners, alpha, k):
    ABs = {target: load_fold(target)}
    for p in partners:
        ABs[p] = load_fold(p)
    npart = max(1, len(partners))
    tensors = {}
    for stem in stems:
        Usig = USIG[stem][:, :k]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VFULL[stem][:, :k]
        Bb, Ab = [Usig], [V.T]                                  # S_k
        for t, c in [(target, 1.0)] + [(p, alpha / npart) for p in partners]:
            b, a = ABs[t][stem]
            M = U.T @ b
            N = a @ V
            Bb += [c * b, -c * (U @ M)]                          # c * (D_t - P_k(D_t))
            Ab += [a, N @ V.T]
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
    r = subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"], capture_output=True, text=True)
    return r.returncode != 0


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
    for pat in ("e20_*.jsonl", "e22_*.jsonl"):
        for f in (REPO / "results/merge").glob(pat):
            for ln in open(f):
                d = json.loads(ln)
                if d.get("status") == "ok":
                    R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    CAP = {4: 0.55, 8: 0.67, 12: 0.74, 16: 0.80}
    lines = ["=== Q3: does the shared/residual boundary k matter? (25 rollouts) ===",
             "capture@k = fraction of an adapter's energy assigned to 'shared'", ""]
    for arm, ks, tmpl16 in (("swap  (S_k + own residual)", SWAP_KS + [16], "e20_t{T}_base_a0"),
                            ("op    (S_k + own + .5*rand5)", OP_KS + [16], "e20_t{T}_rand5_a50")):
        lines.append(f"{arm}")
        lines.append(f"  {'k':<4}{'capture':>9}" + "".join(f"{'t'+str(T):>7}" for T in TARGETS) + f"{'meanD':>9}")
        for k in ks:
            job = tmpl16 if k == 16 else ("e22_t{T}_swap_k%d" % k if arm.startswith("swap")
                                          else "e22_t{T}_op_k%d" % k)
            row = f"  {k:<4}{CAP[k]:>9.2f}"
            ds = []
            for T in TARGETS:
                v = tr(R.get(job.format(T=T), {}), T)
                row += f"{'' if v is None else f'{v*100:.0f}%':>7}"
                if v is not None:
                    ds.append((v - tr(base, T)) * 100)
            row += f"{np.mean(ds):>+8.1f}" if ds else f"{'':>9}"
            lines.append(row)
        lines.append("")
    open(REPO / "results/analysis/ksweep_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    specs = []
    for T in TARGETS:
        for k in SWAP_KS:
            specs.append((T, f"e22_t{T}_swap_k{k}", [], 0.0, k))
        for k in OP_KS:
            specs.append((T, f"e22_t{T}_op_k{k}", RAND5[T], 0.5, k))
    print(f"[stage] {len(specs)} policies queued; waiting for GPUs", flush=True)
    while not gpus_free():
        time.sleep(120)
    print("[stage] GPUs free — starting", flush=True)
    for i in range(0, len(specs), 3):
        jobs, built = [], []
        for T, job, partners, alpha, k in specs[i:i + 3]:
            if already_done(job, scenes[T]):
                continue
            path = build(f"ks_{job}", T, partners, alpha, k)
            built.append(path)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        if not jobs:
            continue
        planp = REPO / "sweep/_ks_current.json"
        json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] batch of {len(jobs)}", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
    report()


if __name__ == "__main__":
    main()
