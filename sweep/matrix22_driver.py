#!/usr/bin/env python3
"""Self-driving full 2x2 matrix run (EXP-18 closing experiment).

For each pair, loops: build the 4 factorial policies -> eval them (25 rollouts) ->
DELETE the adapters (each is ~750MB; building all 64 upfront filled the disk).
Resumable: pairs whose results already exist are skipped. Writes the final matrix
to results/analysis/matrix_2x2_full.txt when done.

Cells: {shared merged, residual merged} x {similar (twin), dissimilar}.
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
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
US = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
PAIRS = json.load(open(REPO / "sweep/matrix22_pairs.json"))
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}


def load_fold(t):
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
    return out


def build(name, Ao, Ap, mode):
    tensors = {}
    for stem in stems:
        Usig = US[stem]
        sig = torch.linalg.norm(Usig, dim=0)
        U = Usig / sig.clamp(min=1e-12)
        V = VS[stem]
        bo, ao = Ao[stem]
        bp, ap = Ap[stem]
        Mo, No = U.T @ bo, ao @ V
        Mp, Np = U.T @ bp, ap @ V
        if mode == "sh":                       # mean(P_own,P_partner) + r_own
            Bb = [0.5 * (U @ Mo), 0.5 * (U @ Mp), bo, -(U @ Mo)]
            Ab = [No @ V.T, Np @ V.T, ao, No @ V.T]
        else:                                  # P_own + mean(r_own, r_partner)
            Bb = [U @ Mo, 0.5 * bo, -0.5 * (U @ Mo), 0.5 * bp, -0.5 * (U @ Mp)]
            Ab = [No @ V.T, ao, No @ V.T, ap, Np @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p))
    return str(p)


def already_done(job, sc):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return False
    ok = {json.loads(l)["scene"] for l in open(f) if json.loads(l).get("status") == "ok"}
    return set(sc) <= ok


def main():
    for tag in ("twin", "dissim"):
        for a, b in PAIRS[tag]:
            jobs, built = [], []
            for own, partner in ((a, b), (b, a)):
                for mode in ("sh", "re"):
                    job = f"e18f_{mode}_{tag}_t{own}p{partner}"
                    if already_done(job, scenes[own]):
                        continue
                    if not built:
                        Aa, Ab_ = load_fold(a), load_fold(b)
                    Ao, Ap = (Aa, Ab_) if own == a else (Ab_, Aa)
                    path = build(f"m22f_{mode}_{tag}_t{own}p{partner}", Ao, Ap, mode)
                    built.append(path)
                    jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                                 "adapter_paths": [path], "eval_scenes": scenes[own]})
            if not jobs:
                print(f"[skip] {tag} {a}:{b} already complete", flush=True)
                continue
            planp = REPO / "sweep/_m22_current.json"
            json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
            print(f"[run] {tag} {a}:{b} — {len(jobs)} policies", flush=True)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                            str(planp)], cwd=REPO)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)      # reclaim ~750MB each
            print(f"[done] {tag} {a}:{b}", flush=True)
    report()


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
    for f in (REPO / "results/merge").glob("e18f_*.jsonl"):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    lines = ["=== FULL 2x2 MATRIX (all pairs, 25 rollouts, delta vs own baseline) ==="]
    cells = {}
    for mode, mn in (("sh", "SHARED merged"), ("re", "RESIDUAL merged")):
        for tag, lab in (("twin", "similar"), ("dissim", "dissimilar")):
            ds = []
            for a, b in PAIRS[tag]:
                for own, partner in ((a, b), (b, a)):
                    v = tr(R.get(f"e18f_{mode}_{tag}_t{own}p{partner}", {}), own)
                    if v is not None:
                        ds.append(v - tr(base, own))
            if ds:
                m, se = float(np.mean(ds)) * 100, float(np.std(ds, ddof=1) / np.sqrt(len(ds))) * 100
                cells[(mode, tag)] = (m, se, len(ds))
                lines.append(f"  {mn:<16} x {lab:<11}: {m:+6.1f}pp  +-{se:4.1f} (SE)  n={len(ds)} task-instances")
    open(REPO / "results/analysis/matrix_2x2_full.txt", "w").write("\n".join(lines) + "\n")
    json.dump({f"{k[0]}_{k[1]}": v for k, v in cells.items()},
              open(REPO / "results/analysis/matrix_2x2_full.json", "w"), indent=1)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
