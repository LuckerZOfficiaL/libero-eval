#!/usr/bin/env python3
"""EXP-109 — clean scaling curve of multi-task merging, and a structure-derived operator.

Design (fixed before any rollout):
  tasks    the 74 unique LIBERO-90 fleet tasks; for each k in KS a seeded random PARTITION into groups of k
  panel    24 seeded-random tasks, scored at every k in the group that contains them (25 episodes, states 0..24)
  baseline the same 24 single-task adapters, same period (e109_base_t{t}), 25 episodes, states 0..24
  schemes  avg  : W = base + (1/k) sum_i D_i                          (plain averaging; orchestrator merge)
           sr{a}: W = base + S16 + alpha * sum_i r_i, r_i = D_i - U U' D_i V V'  (shared part once + residuals)
                  alpha in {1 ("sr1"), 1/sqrt(k) ("srq"), 1/k ("srk")} for k in SR_KS; prebuilt dense checkpoints
  metric   retention = merged success / single-task success on the panel (pooled), and per-task gaps
Env: E109_PART = base | avg | sr ; E109_KS (comma list, default per part); E109_ROUND (prebuilt per round, 4).
Records e109_base_t{t}, e109_avg_k{k}_g{i}, e109_sr{a}_k{k}_g{i} (multi-scene jobs: one record line per scene).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import unified_pool_driver as U
from reinforce_train import scenes as SC
UNIQUE = [t for t in range(76) if t not in (18, 62)]
PANEL = sorted(int(x) for x in np.random.default_rng(109).choice(UNIQUE, 24, replace=False))
KS = [2, 4, 8, 16, 37, 74]
SR_KS = []                                                   # 10-07: operator arms moved to EXP-110 (validated benchmark)
ALPHA = {"sr1": lambda k: 1.0, "srq": lambda k: 1.0 / np.sqrt(k), "srk": lambda k: 1.0 / k}
BASE_DIR = pathlib.Path("/media/pinas/Chris/models/pi05_base_pytorch")
PRE = pathlib.Path("/dev/shm/mergesweep/prebuilt")
ROUND = int(os.environ.get("E109_ROUND", "4"))

def groups(k):
    perm = [int(x) for x in np.random.default_rng(1090 + k).permutation(UNIQUE)]
    return [sorted(perm[i:i + k]) for i in range(0, len(perm), k)]
def panel_groups(k):
    return [(i, g, [t for t in g if t in PANEL]) for i, g in enumerate(groups(k)) if any(t in PANEL for t in g)]

def rec(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists(): return None
    for ln in open(f):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0]: return r
    return None

def orchestrate(jobs, tag, n=25):
    if not jobs: return
    for j in jobs: j.update(suite="libero_90", norm_stats=U.FLEET_NORM)
    planp = REPO / f"sweep/_e109_{tag}_{os.getpid()}.json"
    json.dump(dict(U.META, n_trials=n, jobs=jobs, eval_script="examples/libero/main_dense.py", save_videos="0",
                   heavy_parallel=2), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)

def build_sr(out, members, alpha, base_sd, ck_key):
    """dense W = base + S16 + alpha * (M - Un Un' M V V'), M = sum_i D_i, written as a full checkpoint in /dev/shm"""
    sd = dict(base_sd)
    folds = [U.fold(t) for t in members]
    with torch.no_grad():
        for s_, ck in ck_key.items():
            Bs = [f[s_][0] for f in folds if s_ in f]; As = [f[s_][1] for f in folds if s_ in f]
            Usig, V = U.USIG[s_], U.VS[s_]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
            D = Usig @ V.T
            if Bs:
                M = torch.cat(Bs, 1) @ torch.cat(As, 0)
                D = D + alpha * (M - Un @ (Un.T @ M @ V) @ V.T)
            W = base_sd[ck].float()
            sd[ck] = (W + (D if D.shape == W.shape else D.T)).to(base_sd[ck].dtype)
    shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
    st.save_file(sd, str(out / "model.safetensors"))
    shutil.copy2(BASE_DIR / "config.json", out / "config.json")
    shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")

def main():
    part = os.environ.get("E109_PART", "base"); U._clear_orphaned_builds()
    ks = [int(x) for x in os.environ["E109_KS"].split(",")] if os.environ.get("E109_KS") else (KS if part == "avg" else SR_KS)
    print(f"[e109] part {part}; panel {PANEL}; ks {ks}", flush=True)
    if part == "base":
        orchestrate([{"name": f"e109_base_t{t}", "method": "average", "params": {}, "adapters": [],
                      "adapter_paths": [str(U.afile[t])], "eval_scenes": [SC[t][0]]}
                     for t in PANEL if rec(f"e109_base_t{t}", t) is None], "base")
    elif part == "avg":
        for k in ks:
            # one job per (group, panel task): the orchestrator reuses the group's cached build across workers
            # (10-07: multi-scene jobs ran their scenes serially on one worker -- 3 h for the 74-way model)
            J = [{"name": f"e109_avg_k{k}_g{i}_t{t}", "method": "average", "params": {}, "adapters": [],
                  "adapter_paths": [str(U.afile[x]) for x in g], "eval_scenes": [SC[t][0]]}
                 for i, g, pt in panel_groups(k) for t in pt
                 if rec(f"e109_avg_k{k}_g{i}", t) is None and rec(f"e109_avg_k{k}_g{i}_t{t}", t) is None]
            print(f"[e109] avg k={k}: {len(J)} merged models", flush=True); orchestrate(J, f"avg{k}")
    else:
        base_sd = st.load_file(str(BASE_DIR / "model.safetensors"))
        ck_key = {s_: al.resolve_base_key({k: None for k in base_sd}, s_) for s_ in U.stems}
        ck_key = {s_: k for s_, k in ck_key.items() if k is not None}
        todo = [(sch, k, i, g, pt) for k in ks for sch in ALPHA for i, g, pt in panel_groups(k)
                if any(rec(f"e109_{sch}_k{k}_g{i}", t) is None for t in pt)]
        print(f"[e109] sr: {len(todo)} prebuilt models, {len(ck_key)} modules", flush=True)
        for r0 in range(0, len(todo), ROUND):
            jobs = []
            for sch, k, i, g, pt in todo[r0:r0 + ROUND]:
                t1 = time.time(); out = PRE / f"e109_{sch}_k{k}_g{i}"
                build_sr(out, g, float(ALPHA[sch](k)), base_sd, ck_key)
                print(f"[e109] built {sch} k={k} g{i} ({len(g)} tasks, alpha {ALPHA[sch](k):.3f}) in {time.time()-t1:.0f}s", flush=True)
                jobs.append({"name": f"e109_{sch}_k{k}_g{i}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [],
                             "prebuilt_dir": str(out), "eval_scenes": [SC[t][0] for t in pt]})
            orchestrate(jobs, f"sr{r0}")
            for j in jobs: shutil.rmtree(j["prebuilt_dir"], ignore_errors=True)
            print(f"[e109] sr round {r0 // ROUND + 1}/{(len(todo) + ROUND - 1) // ROUND} done", flush=True)
    print(f"[e109] part {part} DONE", flush=True)

if __name__ == "__main__":
    main()
