#!/usr/bin/env python3
"""EXP-81 -- OOD robustness of probe-merged targets (RETAIN's axis), 4 arms x 2 shift levels.

RETAIN (arXiv 2512.08333): interpolating a VLA before/after finetuning buys OOD robustness of the finetuned
task with ID unchanged; the LLM backbone at alpha~0.8 carries it. Every number we have is fixed-state ID.
Whether probe-selected donors buy OOD robustness -- especially on the strong targets that show ~0 ID gain --
has never been measured. OOD here = every free-joint object translated by 5 / 10 cm along one episode-seeded
direction (canonical states span ~+-2.4 cm), via main_dense.py --args.ood-shift.

arms (target residual on LLM bands / donors):
  A  1.0 / none                       the target alone under OOD
  B  0.8 / none                       RETAIN on a LoRA target, no donors
  C  1.0 / e27 top-3 at c/3           our deployed commit under OOD
  D  0.8 / e27 top-3 at c/3           do they stack
20 fleet targets: 10 weakest + 10 strongest with e27 commits. 25 rollouts per cell. Records e90_{arm}_t{t}_s{cm}.
  python sweep/ood_robust.py --dry
"""
import json, pathlib, subprocess, sys
import numpy as np, torch
from safetensors import torch as st
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC
from reinforce_train import group_of          # 0 vision, 1 llm-early, 2 llm-late, 3 expert, 4 action

WEAK   = [19, 56, 6, 50, 8, 15, 14, 60, 9, 33]
STRONG = [0, 1, 2, 4, 5, 13, 16, 21, 22, 24]
TARGETS = WEAK + STRONG
SHIFTS = [0.05, 0.10]
ARMS = {"A": (1.0, False), "B": (0.8, False), "C": (1.0, True), "D": (0.8, True)}
LLM_BANDS = {1, 2}

def build(name, t, beta_llm, donors):
    """S16 form with the TARGET residual scaled by beta on LLM-band stems (1.0 elsewhere); donors uniform."""
    ft = U.fold(t); tensors = {}
    for stem in U.stems:
        Usig = U.USIG[stem]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[stem]
        Bb, Ab = [Usig], [V.T]
        kt = beta_llm if group_of(stem) in LLM_BANDS else 1.0
        parts = [(ft, kt)] + [(U.fold(d), c) for d, c in donors]
        for f, k in parts:
            if stem not in f or k == 0.0: continue
            b, a = f[stem]; M = Un.T @ b; N = a @ V
            Bb += [k * b, -k * (Un @ M)]; Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    U._ensure_space(); st.save_file(tensors, str(p)); return str(p)

def main(dry):
    if not dry: U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for s in SHIFTS:
        for arm, (beta, use_d) in ARMS.items():
            for t in TARGETS:
                job = f"e90_{arm}_t{t}_s{int(s*100)}"
                if U.get(job, SC[t][0]) is None: todo.append((job, t, s, beta, use_d))
    print(f"[ood] {len(todo)} cells to run of {len(SHIFTS)*len(ARMS)*len(TARGETS)}; ~{len(todo)*25} rollouts", flush=True)
    if dry: return
    for i in range(0, len(todo), U.CHUNK):
        chunk = todo[i:i + U.CHUNK]
        # one plan per shift value (orchestrator takes ood_shift plan-wide); chunks are built shift-major
        for s in sorted(set(c[2] for c in chunk)):
            jobs, built = [], []
            for job, t, ss, beta, use_d in chunk:
                if ss != s: continue
                donors = [(int(d), (ref / U.resid_norm(int(d))) / 3) for d, _ in U.E27[str(t)][:3]] if use_d else []
                p = build(f"ood_{job}", t, beta, donors); built.append(p)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": SC[t]})
            # eval_script MUST be main_dense.py: only that copy has --args.ood-shift. The orchestrator defaults
            # to main.py, which rejects the flag and dies in 3s with status=parse_fail (cost 40 cells, 09-16).
            json.dump(dict(U.META, n_trials=25, ood_shift=s, eval_script="examples/libero/main_dense.py",
                           jobs=jobs), open(REPO / "sweep/_ood.json", "w"), indent=1)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_ood.json")], cwd=REPO)
            for p in built: pathlib.Path(p).unlink(missing_ok=True)
        print(f"[ood] chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK}", flush=True)
    print("[ood] done")

if __name__ == "__main__":
    main("--dry" in sys.argv)
