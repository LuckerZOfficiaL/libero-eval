#!/usr/bin/env python3
"""EXP-84/85 stage 1 -- per-episode outcomes for every fleet target's OWN adapter on the 25 fixed states.

Both planned selectors need what the archives lack: not a success RATE but WHICH initial states succeed
and how far each episode gets (main_dense.py logs PROGRESS and SUCCESS_STEP). EXP-84 treats states as
test items and needs per-state outcomes across candidates; EXP-85 describes a target by WHERE it fails.
This pass is build-free -- it evaluates the raw fleet adapters as-is -- so it is safe on odin's full disk.
Records: e92_base_f{t}, 25 episodes, suite libero_90.
"""
import json, pathlib, subprocess, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, afile

def main(dry=False):
    todo = [t for t in range(76) if U.get(f"e92_base_f{t}", SC[t][0]) is None]
    # FP_RANGE="62-75" restricts to a slice (second machine takes the tail); FP_REVERSE=1 walks it backwards
    # so two machines working toward each other rarely evaluate the same target twice.
    import os
    if os.environ.get("FP_RANGE"):
        lo, hi = map(int, os.environ["FP_RANGE"].split("-")); todo = [t for t in todo if lo <= t <= hi]
    if os.environ.get("FP_REVERSE"): todo = todo[::-1]
    print(f"[fp] {len(todo)}/76 fleet targets need a per-episode baseline (25 rollouts each, no builds)", flush=True)
    if dry: return
    for i in range(0, len(todo), 8):
        jobs = [{"name": f"e92_base_f{t}", "method": "average", "params": {}, "adapters": [],
                 "adapter_paths": [str(afile[t])], "eval_scenes": SC[t]} for t in todo[i:i + 8]]
        json.dump(dict(U.META, n_trials=25, eval_script="examples/libero/main_dense.py", jobs=jobs),
                  open(REPO / "sweep/_fp.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_fp.json")], cwd=REPO)
        print(f"[fp] chunk {i//8+1}/{(len(todo)+7)//8}", flush=True)
    print("[fp] done", flush=True)

if __name__ == "__main__":
    main("--dry" in sys.argv)
