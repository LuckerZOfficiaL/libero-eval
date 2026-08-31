#!/usr/bin/env python3
"""EXP-26 — POOL-INCONSISTENT DONORS: does the method absorb LoRAs from outside its pool?

The deployment question we have never tested: someone hands you a LoRA trained elsewhere. Can it serve
as a donor for your target, or does the method only work inside the fleet it was built from?

WHAT WE ALREADY KNOW (free CPU check, run before this):
  capture@16 by OUR soup basis  --  our adapters 0.84-0.87   external experts 0.19-0.39
So external LoRAs live largely OUTSIDE the shared subspace S_16 is built from. Consequences:
  * their "residual" r = D - P_16(D) is ~70-80% of the full delta (ours is ~13%), so
  * at the same alpha an external donor injects ~2.4x the energy -- a DOSING confound, not a
    partner-quality difference. Every arm here is therefore ENERGY-MATCHED: each external donor's
    coefficient is scaled so ||c*r_ext|| equals the mean ||0.5*r_internal|| for that target's pool.
  * a naive (un-matched) arm is run on a subset to demonstrate the trap rather than assume it.

ARMS per weak target (5 targets, 8 external donors):
  ext_matched  : W = S_16 + r_T + c_matched * r_ext        (9 donors, 10 rollouts)  <- the real test
  ext_naive    : W = S_16 + r_T + 0.5 * r_ext              (2 donors, 10 rollouts)  <- the dosing trap
  commit       : top-3 external donors, energy-matched, 25 rollouts
Paired control: e27's internal top-3 commit at the same budget.
"""
import glob
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
TARGETS = [19, 56, 6, 50, 15]
NAIVE_ON = [19, 6]          # subset for the un-matched dosing arm
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
EXT = sorted({p.split("/")[-3]: p for p in
              glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/adapter_model.safetensors")
              }.items())
print(f"[ext] {len(EXT)} external experts: {[n for n, _ in EXT]}", flush=True)
_cache = {}


def fold_path(p):
    key = str(p)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 20:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(p))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return _cache.setdefault(key, out)


def resid_norm(f):
    """Frobenius norm of the residual r = D - P_16(D), summed over modules."""
    tot = 0.0
    for s in stems:
        if s not in f:
            continue
        b, a = f[s]
        D = b @ a
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        P = Un @ (Un.T @ D @ VS[s]) @ VS[s].T
        tot += float(((D - P) ** 2).sum())
    return tot ** 0.5


def build(name, terms):
    """W = S_16 + sum(c * r) for terms = [(fold, coeff), ...]; emitted as LoRA factors."""
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for f, c in terms:
            if stem not in f:
                continue
            b, a = f[stem]
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


def run_plan(jobs, n):
    if not jobs:
        return
    planp = REPO / "sweep/_ext_current.json"
    json.dump(dict(META, n_trials=n, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    # internal reference energy: mean ||0.5 * r_internal|| over the fleet
    ref = ALPHA * float(np.mean([resid_norm(fold_path(afile[t])) for t in range(0, 76, 8)]))
    print(f"[energy] internal reference ||0.5*r|| = {ref:.2f}", flush=True)
    coef = {}
    for nm, p in EXT:
        rn = resid_norm(fold_path(p))
        coef[nm] = ref / rn
        print(f"[energy] {nm:<32} ||r_ext||={rn:8.2f}  matched coeff={coef[nm]:.3f} "
              f"(naive would be {ALPHA})", flush=True)
    json.dump(coef, open(REPO / "results/analysis/ext_coeffs.json", "w"), indent=1)
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(120)
    for T in TARGETS:
        tf = fold_path(afile[T])
        jobs, built = [], []
        for nm, p in EXT:
            job = f"e36_ext_t{T}_{nm}"
            if rate(job, T) is None:
                path = build(f"ex_t{T}_{nm}", [(tf, 1.0), (fold_path(p), coef[nm])])
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": scenes[T]})
        if T in NAIVE_ON:
            for nm, p in EXT[:2]:
                job = f"e36_naive_t{T}_{nm}"
                if rate(job, T) is None:
                    path = build(f"exn_t{T}_{nm}", [(tf, 1.0), (fold_path(p), ALPHA)])
                    built.append(path)
                    jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                                 "adapter_paths": [path], "eval_scenes": scenes[T]})
        if jobs:
            print(f"[probe] t{T}: {len(jobs)} external arms", flush=True)
            run_plan(jobs, PROBE_R)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        # commit the best 3 external donors, energy-matched
        sc = sorted(((rate(f"e36_ext_t{T}_{nm}", T), nm) for nm, _ in EXT
                     if rate(f"e36_ext_t{T}_{nm}", T) is not None), reverse=True)
        top = [nm for _, nm in sc[:3]]
        print(f"[select] t{T}: external top-3 {top}", flush=True)
        if top and rate(f"e36_extcommit_t{T}", T) is None:
            d = dict(EXT)
            terms = [(tf, 1.0)] + [(fold_path(d[nm]), coef[nm] / 3) for nm in top]
            path = build(f"exc_t{T}", terms)
            run_plan([{"name": f"e36_extcommit_t{T}", "method": "average", "params": {},
                       "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[T]}], COMMIT_R)
            pathlib.Path(path).unlink(missing_ok=True)
        print(f"[done] t{T}", flush=True)
        report(coef)
    report(coef)


def report(coef):
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]

    def brate(t):
        ss = [s for s in scenes[t] if s in base]
        return sum(base[s] for s in ss) / len(ss)

    L = ["=== EXP-26: POOL-INCONSISTENT DONORS (external experts as partners) ===",
         "our S_16 captures 0.84-0.87 of OUR adapters but only 0.19-0.39 of external ones,",
         "so every external arm is ENERGY-MATCHED to the internal residual dose.", "",
         f"{'target':<8}{'base':>6}{'internal top3':>15}{'EXTERNAL top3':>15}{'ext-int':>9}"
         f"{'best ext probe':>16}"]
    di, de = [], []
    for T in TARGETS:
        b = brate(T)
        i3 = rate(f"e27_commit_t{T}_top3", T)
        e3 = rate(f"e36_extcommit_t{T}", T)
        probes = [rate(f"e36_ext_t{T}_{nm}", T) for nm, _ in EXT]
        probes = [p for p in probes if p is not None]
        row = f"t{T:<7}{b*100:>5.0f}%"
        row += f"{'' if i3 is None else f'{i3*100:.0f}%':>15}"
        row += f"{'' if e3 is None else f'{e3*100:.0f}%':>15}"
        if i3 is not None and e3 is not None:
            di.append((i3 - b) * 100); de.append((e3 - b) * 100)
            row += f"{(e3-i3)*100:>+9.0f}"
        row += f"{'' if not probes else f'{max(probes)*100:.0f}%':>16}"
        L.append(row)
    if di:
        L += ["", f"  internal donors: {np.mean(di):+.2f}pp vs baseline",
              f"  EXTERNAL donors: {np.mean(de):+.2f}pp vs baseline",
              f"  gap: {np.mean(de)-np.mean(di):+.2f}pp"]
    nv = [(T, nm, rate(f"e36_naive_t{T}_{nm}", T), rate(f"e36_ext_t{T}_{nm}", T))
          for T in NAIVE_ON for nm, _ in EXT[:2]]
    nv = [x for x in nv if x[2] is not None and x[3] is not None]
    if nv:
        L += ["", "  DOSING TRAP (same donor, un-matched alpha=0.5 vs energy-matched):"]
        for T, nm, a, b_ in nv:
            L.append(f"    t{T} {nm:<32} naive {a*100:3.0f}%   matched {b_*100:3.0f}%   "
                     f"{(a-b_)*100:+.0f}pp")
    open(REPO / "results/analysis/external_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
