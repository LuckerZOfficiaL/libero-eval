#!/usr/bin/env python3
"""N=3 merge-safety factorial (EXP-18 compositionality test).

For each triplet (a,b,c) and each member `own`, two policies (build -> eval -> delete):
  sh: mean(P(D_a),P(D_b),P(D_c)) + r_own          (shared merged across 3, own residual kept)
  re: P(D_own) + mean(r_a, r_b, r_c)              (own shared kept, residuals merged across 3)
25 rollouts on own's scenes. Resumable. Writes results/analysis/matrix_n3_report.txt at the end.
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
TRIPLETS = {k: tuple(v) for k, v in json.load(
    open(pathlib.Path("/home/lzhou/libero-eval/sweep/matrix3_triplets.json"))).items()}
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


def build(name, ABs, own, grp, mode):
    tensors = {}
    for stem in stems:
        Usig = US[stem]
        sig = torch.linalg.norm(Usig, dim=0)
        U = Usig / sig.clamp(min=1e-12)
        V = VS[stem]
        parts = {}
        for t in grp:
            b, a = ABs[t][stem]
            parts[t] = (b, a, U.T @ b, a @ V)          # b, a, M, N
        bo, ao, Mo, No = parts[own]
        if mode == "sh":
            # mean of the three projections + own residual
            Bb = [(1 / 3) * (U @ parts[t][2]) for t in grp] + [bo, -(U @ Mo)]
            Ab = [parts[t][3] @ V.T for t in grp] + [ao, No @ V.T]
        else:
            # own projection + mean of the three residuals
            Bb = [U @ Mo]
            Ab = [No @ V.T]
            for t in grp:
                b, a, M, Nv = parts[t]
                Bb += [(1 / 3) * b, -(1 / 3) * (U @ M)]
                Ab += [a, Nv @ V.T]
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
    for tag, grp in TRIPLETS.items():
        jobs, built = [], []
        ABs = None
        for own in grp:
            for mode in ("sh", "re"):
                job = f"e18g_{mode}_{tag}_t{own}"
                if already_done(job, scenes[own]):
                    continue
                if ABs is None:
                    ABs = {t: load_fold(t) for t in grp}
                path = build(f"m3_{mode}_{tag}_t{own}", ABs, own, grp, mode)
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": scenes[own]})
        if not jobs:
            print(f"[skip] {tag} {grp} complete", flush=True)
            continue
        planp = REPO / "sweep/_m3_current.json"
        json.dump(dict(META, n_trials=25, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] {tag} {grp} — {len(jobs)} policies", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        print(f"[done] {tag} {grp}", flush=True)
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
    for f in (REPO / "results/merge").glob("e18g_*.jsonl"):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("status") == "ok":
                R.setdefault(d["job"], {})[d["scene"]] = d["success_rate"]
    lines = ["=== N=3 MERGE-SAFETY FACTORIAL (25 rollouts, delta vs own baseline) ==="]
    for tag, grp in TRIPLETS.items():
        lines.append(f"\n[{tag}] {grp}")
        for mode, mn in (("sh", "shared merged"), ("re", "residual merged")):
            row = []
            for own in grp:
                v = tr(R.get(f"e18g_{mode}_{tag}_t{own}", {}), own)
                row.append(f"t{own}:{'' if v is None else f'{tr(base,own)*100:.0f}->{v*100:.0f}%'}")
            ds = [tr(R.get(f"e18g_{mode}_{tag}_t{own}", {}), own) - tr(base, own)
                  for own in grp if tr(R.get(f"e18g_{mode}_{tag}_t{own}", {}), own) is not None]
            m = np.mean(ds) * 100 if ds else float("nan")
            lines.append(f"  {mn:<16} mean {m:+6.1f}pp   " + "  ".join(row))
    open(REPO / "results/analysis/matrix_n3_report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
