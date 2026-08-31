#!/usr/bin/env python3
"""ALL-76 deployment screen for the winning asymmetric operator (per user reporting rule).

The flagship measured 5 WEAK targets (+4.0pp). A deployable policy is applied to whatever LoRA
arrives — 61 of our 76 tasks are already >=90%, where the operator can only hold or harm. This run
applies the winning selection-free recipe to ALL 76 targets and reports the deployment-grade table
(all / weak / strong split + count harmed), the same shape as the EXP-13 safety screen.

  recipe: W = S_16 + 1.0*r_target + 0.5 * mean(5 RANDOM pool residuals)

Random partners are drawn per target with a fixed seed (logged). Build -> eval (25 rollouts) ->
delete, 3 policies live at a time. Waits for the GPUs (Q3 k-sweep) before starting.
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
NPART = 5
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
PARTNERS = {T: sorted(int(x) for x in np.random.default_rng(1000 + T).choice(
    [i for i in range(76) if i != T], NPART, replace=False)) for T in range(76)}
_cache = {}


def load_fold(t):
    if t in _cache:
        return _cache[t]
    if len(_cache) > 24:                    # bound RAM: keep the working set small
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


def build(name, target):
    ABs = {t: load_fold(t) for t in [target] + PARTNERS[target]}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in [(target, 1.0)] + [(p, ALPHA / NPART) for p in PARTNERS[target]]:
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
    for f in (REPO / "results/merge").glob("e23_*.jsonl"):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    rows = []
    for T in range(76):
        v = tr(R.get(f"e23_t{T}", {}), T)
        if v is not None:
            rows.append((T, tr(base, T), v, (v - tr(base, T)) * 100))
    d = np.array([r[3] for r in rows])
    b = np.array([r[1] for r in rows])
    weak, strong = b < 0.90, b >= 0.90
    L = ["=== ALL-76 DEPLOYMENT SCREEN: S_16 + 1.0*r_target + 0.5*mean(5 random) (25 rollouts) ===", ""]
    L.append(f"ALL {len(rows)} targets : mean {d.mean():+.2f}pp  (sd {d.std():.1f}, SE {d.std(ddof=1)/np.sqrt(len(d)):.2f})")
    for nm, m in (("WEAK  (base<90%)", weak), ("STRONG(base>=90%)", strong)):
        if m.sum():
            L.append(f"{nm} n={int(m.sum()):<3}: mean {d[m].mean():+.2f}pp  "
                     f"lifted(>+5) {int((d[m]>5).sum())}  flat {int((abs(d[m])<=5).sum())}  hurt(<-5) {int((d[m]<-5).sum())}")
    L.append(f"\nfleet-wide success: {b.mean()*100:.1f}% -> {np.array([r[2] for r in rows]).mean()*100:.1f}%")
    L.append("\nbiggest gains:")
    for T, bb, v, dd in sorted(rows, key=lambda r: -r[3])[:8]:
        L.append(f"  t{T:<3} {bb*100:3.0f}% -> {v*100:3.0f}%  ({dd:+.0f}pp)")
    L.append("biggest losses:")
    for T, bb, v, dd in sorted(rows, key=lambda r: r[3])[:8]:
        L.append(f"  t{T:<3} {bb*100:3.0f}% -> {v*100:3.0f}%  ({dd:+.0f}pp)")
    open(REPO / "results/analysis/all76_report.txt", "w").write("\n".join(L) + "\n")
    json.dump({str(T): {"base": bb, "merged": v, "delta": dd} for T, bb, v, dd in rows},
              open(REPO / "results/analysis/all76.json", "w"), indent=1)
    print("\n".join(L), flush=True)


def main():
    print("[stage] all-76 queued; waiting for GPUs", flush=True)
    while not gpus_free():
        time.sleep(120)
    print("[stage] GPUs free — starting", flush=True)
    todo = [T for T in range(76) if not already_done(f"e23_t{T}", scenes[T])]
    for i in range(0, len(todo), 3):
        jobs, built = [], []
        for T in todo[i:i + 3]:
            path = build(f"a76_t{T}", T)
            built.append(path)
            jobs.append({"name": f"e23_t{T}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[T]})
        planp = REPO / "sweep/_a76_current.json"
        json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] targets {[j['name'] for j in jobs]}", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
    report()


if __name__ == "__main__":
    main()
