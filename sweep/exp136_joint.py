#!/usr/bin/env python3
"""EXP-136 — joint-training upper bound (user plan 10-21).

One LoRA trained JOINTLY on all 24 skills of an order (its 16 library skills + its 8 new skills), from the base model, on
their demos — the standard continual-learning upper bound (all data available at once). One model per order (EXP-116
sequences 0-3). Recipe: rank 64 (4x a single skill; rank 64 batch 8 fits an A6000), 12,000 steps, batch 8, each step draws
one skill uniformly and a batch from its demos with its bare-instruction prompt, frame stride 4, AdamW 1e-4, plain LoRA init.

PRE-REGISTERED (10-21, before any training): evaluate each joint model on all 24 of its skills (10 eps, states 0..9);
retention vs each skill's single-task success (library: fleet adapters; new: write-protected adapters) — compared with the
t = 8 merged models of EXP-134 (same orders, same 24 skills). Prediction: joint training is the best single model on
average (library and new >= 80% retention); our merges stay below it, most on new skills.
Env: E136_STAGE train|eval|report; E136_ORDERS (default 0,1,2,3); E136_GPU (train, physical index); E113_* (eval).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp124_wpcontinual as W                                         # registers WP adapters (single-task references)
import exp118_anchored as A
import exp112_bench as E
import exp113_controls as X
import exp120_privileged as P
import unified_pool_driver as U
ORDERS = [int(x) for x in os.environ.get("E136_ORDERS", "0,1,2,3").split(",")]
SFX, STEPS = os.environ.get("E136_SFX", ""), os.environ.get("E136_STEPS", "12000")   # 11-08 EXP-150 fairness: E136_SFX=24k E136_STEPS=24000
ADIR = REPO / "results/analysis/joint_loras"
JX = {}
for s in list(range(7)) + [32]:                                        # 11-06: s4 = scaling pseudo-order (EXP-148, 38 skills); 11-07: s5 / s6 = EXP-150 library 32 / 74; 11-22: s32 = EXP-170 (32 + 22)
    path = ADIR / f"joint{SFX}_s{s}" / "adapter_model.safetensors"; key = f"joint{SFX}:s{s}"
    U.EXTPATH[key] = str(path)
    JX[s] = len(E.LIB); E.LIB.append({"key": key, "ext": True, "suite": "joint", "scene": f"joint{SFX}_s{s}", "tid": f"j{s}{SFX}", "path": str(path)})

def skills(s):
    old, new = W.sequence(s); return list(old) + list(new)

def train():
    for s in ORDERS:
        out = ADIR / f"joint{SFX}_s{s}"
        if (out / "adapter_model.safetensors").exists(): print(f"[e136] s{s} already trained", flush=True); continue
        out.mkdir(parents=True, exist_ok=True); t0 = time.time()
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=os.environ.get("E136_GPU", "0"))
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--tasks_file", str(REPO / f"results/analysis/exp136_joint_tasks_s{s}.txt"),
               "--rank", "64", "--stride", "4", "--steps", STEPS, "--batch", "8", "--device", "cuda:0", "--out", str(out)]
        with open(REPO / f"results/merge_logs/e136_joint{SFX}_s{s}.log", "w") as lg:
            rc = subprocess.run(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, cwd=REPO).returncode
        print(f"[e136] s{s} trained rc={rc} in {(time.time()-t0)/3600:.1f} h", flush=True)

def jobs(s):
    return [(f"e136_joint{SFX}_s{s}_{E.LIB[i]['tid']}", E.LIB[i]["suite"], E.LIB[i]["scene"]) for i in skills(s)]

def evaluate():
    todo = [s for s in ORDERS if pathlib.Path(E.LIB[JX[s]]["path"]).exists() and any(E.rec(n, sc) is None for n, su, sc in jobs(s))]
    print(f"[e136] eval: {len(todo)} joint models", flush=True)
    B = P.PBuilder()
    for s in todo:
        out = E.PRE / f"e136_joint{SFX}_s{s}"; B.build("ta", {"lam": 1.0}, None, [JX[s]], out)
        js = [{"name": n, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
               "eval_scenes": [sc], "suite": su, "norm_stats": U.FLEET_NORM} for n, su, sc in jobs(s) if E.rec(n, sc) is None]
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(js, 10, 0, f"e136{SFX}_s{s}"))], cwd=REPO)
        shutil.rmtree(out, ignore_errors=True); print(f"[e136] s{s} evaluated", flush=True)

def report():
    sg = W.single(); L = ["=== EXP-136 joint-training upper bound (one rank-64 LoRA on all 24 skills per order), retention ==="]
    lib, new = [], []
    for s in ORDERS:
        old, nw = W.sequence(s)
        for i in skills(s):
            r = E.rec(f"e136_joint{SFX}_s{s}_{E.LIB[i]['tid']}", E.LIB[i]["scene"])
            if r is None or sg.get(i, 0) <= 0: continue
            (lib if i in old else new).append(r["success_rate"] / sg[i])
    L.append(f"library skills: {100*np.mean(lib):.1f}% (n={len(lib)}) | new skills: {100*np.mean(new):.1f}% (n={len(new)}) | overall {100*np.mean(lib+new):.1f}%")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp136_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E136_STAGE", "train"); U._clear_orphaned_builds()
    {"train": train, "eval": evaluate, "report": report}[st]()
    print(f"[e136] {st} DONE", flush=True)
