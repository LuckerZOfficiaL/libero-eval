#!/usr/bin/env python3
"""EXP-70 — does DOUBLING the candidates at full depth help? Nearly free, using EXP-69's probes.

EXP-66 showed 16 candidates x 5 rollouts does not beat 8 x 10 at equal budget. That is a budget
question. The different question -- worth more rollouts if the answer is yes -- is whether 16 candidates
at FULL depth beats 8, given draw variance of 13.6pp: more candidates should make an unlucky draw less
likely. EXP-32's external targets hinted at it (+12.00 with 16 vs +7.56 with 8, p=0.17).

EXP-69 already probed 4 independent sets of 8 per target at 10 rollouts. Unioning two sets gives a
genuine 16-candidate probe at full depth for free; only the commits are new. Two disjoint unions per
target (draws 0+1 and 2+3) give two independent replicates.
"""
import json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import scenes as SC, target_baseline
from draw_variance import TARGETS, cands
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
PAIRS = [("A", 0, 1), ("B", 2, 3)]

def scores(t, k):
    if k == 0:
        return {int(p): s for p, s in RANK[str(t)]}
    out = {}
    for d in cands(t, k):
        v = U.get(f"e75_t{t}_d{k}_p{d}", SC[t][0])
        if v is not None: out[d] = v
    return out

def main():
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = []
    for t in TARGETS:
        for nm, k1, k2 in PAIRS:
            sc = {**scores(t, k1), **scores(t, k2)}
            if len(sc) < 14 or U.get(f"e77_t{t}_{nm}", SC[t][0]) is not None: continue
            if max(sc.values()) <= target_baseline(t):
                print(f"[dc] t{t} union {nm}: guard abstains", flush=True); continue
            rng = np.random.default_rng(11000 + t)
            jit = {c: sc[c] + 1e-6 * rng.random() for c in sc}     # fair ties, per EXP-66
            todo.append((t, nm, sorted(jit, key=jit.get, reverse=True)[:3], len(sc)))
    print(f"[dc] {len(todo)} commits from 16-candidate unions at full depth", flush=True)
    for i in range(0, len(todo), U.CHUNK):
        jobs, built = [], []
        for t, nm, top, n in todo[i:i + U.CHUNK]:
            p = U.build(f"dc_{t}_{nm}", t, [(c, (ref / U.resid_norm(c)) / 3) for c in top]); built.append(p)
            jobs.append({"name": f"e77_t{t}_{nm}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": SC[t]})
        json.dump(dict(U.META, n_trials=25, jobs=jobs), open(REPO / "sweep/_dc.json", "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(REPO / "sweep/_dc.json")], cwd=REPO)
        for p in built: pathlib.Path(p).unlink(missing_ok=True)
    print("[dc] done")

if __name__ == "__main__":
    main()
