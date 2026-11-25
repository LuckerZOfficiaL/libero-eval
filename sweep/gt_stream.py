#!/usr/bin/env python3
"""EXP-95 ground-truth sweep, STREAMING version (same records e98_t{t}_d{d} as gt_sweep.py).

Chunked evaluation wasted workers: every chunk waited for its slowest job, and workers start ~65 s apart,
so shiva ran at ~13-17 pairs/h with 7 workers that could do ~40. Here ONE orchestrator gets the whole job
list (FIFO), a builder thread keeps up to AHEAD adapters built in the same order, and a cleaner deletes
each adapter as soon as its record exists. A build is ~40 s, a worker consumes one job per ~80 s, so the
builder stays ahead. If a job ever finds its file missing it records an error and a final refill pass
(chunked gt_sweep.py) redoes it.   Env: GT_TARGETS, GT_N (as gt_sweep.py), GT_AHEAD (default 14).
"""
import json, os, pathlib, subprocess, sys, threading, time
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
import gt_sweep as G
from reinforce_train import scenes as SC
AHEAD = int(os.environ.get("GT_AHEAD", "14"))

def path(t, d): return REPO / f"results/analysis/decomp_adapters/gt_{t}_{d}.safetensors"

def main():
    U._clear_orphaned_builds()
    todo = [(t, d) for t in G.TGTS for d in G.UNIQUE if d != t and not G.done(t, d)]
    print(f"[gs] {len(todo)} pairs @ {G.NR} rollouts, streaming, {AHEAD} builds ahead", flush=True)
    if not todo: return
    built = set(); lock = threading.Lock(); stop = {"v": False}
    def live():
        with lock: return [p for p in built if not G.done(*p)]
    def builder():
        for t, d in todo:
            while len(live()) >= AHEAD and not stop["v"]: time.sleep(5)
            if stop["v"]: return
            for (t2, d2) in live():                       # keep in-flight files young (see gt_sweep.build_chunk)
                f = path(t2, d2)
                if f.exists(): os.utime(f, None)
            U.build(f"gt_{t}_{d}", t, [(d, G.REF / U.resid_norm(d))])
            with lock: built.add((t, d))
    th = threading.Thread(target=builder, daemon=True); th.start()
    while len(built) < min(AHEAD, len(todo)): time.sleep(5)
    jobs = [{"name": f"e98_t{t}_d{d}", "method": "average", "params": {}, "adapters": [],
             "adapter_paths": [str(path(t, d))], "eval_scenes": [SC[t][0]]} for t, d in todo]
    proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(G.plan_file(jobs, "stream"))], cwd=REPO)
    last = 0
    while proc.poll() is None:
        time.sleep(30)
        with lock: snapshot = list(built)
        for t, d in snapshot:                              # delete as soon as the record exists
            if G.done(t, d): path(t, d).unlink(missing_ok=True)
        n = sum(1 for t, d in todo if G.done(t, d))
        if n - last >= 10 or n == len(todo):
            print(f"[gs] {n}/{len(todo)} pairs done ({time.strftime('%H:%M')})", flush=True); last = n
    stop["v"] = True
    for t, d in todo:
        path(t, d).unlink(missing_ok=True)
    print(f"[gs] orchestrator exited; {sum(1 for t, d in todo if G.done(t, d))}/{len(todo)} done", flush=True)

if __name__ == "__main__":
    main()
