#!/usr/bin/env python3
"""EXP-80 -- the augmented fleet, PROVENANCE-BLIND: one pool of 84, 8 candidates drawn uniformly.

EXP-32 stratified its candidate list: 8 fleet + 8 external per target. That bakes provenance INTO the
design -- external adapters are 8 of 84 in the pool (under 10%) yet fill half the candidate slots -- and
it doubles the method's canonical 8-candidate budget. It is therefore not a test of "the probe method on
one pool of everything we own"; it is a test that already knows which adapters are which.

This is the honest version. The 84 adapters are ONE list. For every target, 8 candidates are drawn
uniformly from the other 83 (per-target seeded draw, never shared -- EXP-29's failure mode), probed at 10
rollouts, top-3 committed at 25 under the guard. Same operator, dose, guard, baselines (e44_base) as
EXP-32. Where a drawn (target, donor) pair was already probed in EXP-32 -- identical build, identical
10-rollout probe -- that score is reused rather than re-measured. Weak targets run first so the slice
that carries the signal lands first.

  python sweep/blind_pool.py --dry     # plan and cost only
"""
import hashlib, json, os, pathlib, subprocess, sys
import numpy as np
# DRAW=2,3,4 = additional independent candidate draws for draw-averaging (EXP-80 follow-up); DRAW=1 is the
# original run (names unchanged). ONLY_WEAK_FLEET=1 restricts to LIBERO-90 receivers with base < 90%.
DRAW = int(os.environ.get("DRAW", "1")); SFX = "" if DRAW == 1 else f"d{DRAW}"
ONLY_WEAK_FLEET = os.environ.get("ONLY_WEAK_FLEET") == "1"
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]
key = lambda c: c if isinstance(c, int) else c[11:]

def cands(tag, tk, pool):
    rng = np.random.default_rng(int(h6(f"{tag}_blind80" + SFX), 16))
    others = [c for c in pool if c != tk]
    return [others[i] for i in sorted(rng.choice(len(others), 8, replace=False))]

def score(tag, task, c):
    """new probe if present, else the identical EXP-32 probe of the same pair."""
    for pre in (f"e89{SFX}_p", "e89_p", "e89d2_p", "e89d3_p", "e89d4_p", "e44_p"):   # identical build, identical probe
        v = U.get(f"{pre}_{tag}_{key(c)}", task)
        if v is not None: return v
    return None

def plan():
    pool = list(range(76)) + sorted(U.EXTPATH)
    tg = []
    for tag, tk, task, suite, norm in U.targets():
        b = U.get(f"e44_base_{tag}", task)
        if b is None: continue
        if ONLY_WEAK_FLEET and not (isinstance(tk, int) and b < 0.9): continue
        tg.append((b, tag, tk, task, suite, norm))
    tg.sort()                                   # weakest first
    out = []
    for b, tag, tk, task, suite, norm in tg:
        cs = cands(tag, tk, pool)
        need = [c for c in cs if score(tag, task, c) is None]
        out.append((tag, tk, task, suite, norm, b, cs, need))
    return out

def main(dry):
    if not dry: U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    P = plan()
    nx = sum(sum(1 for c in cs if not isinstance(c, int)) for *_, cs, _ in P)
    need = sum(len(n) for *_, n in P)
    done_c = sum(1 for tag, tk, task, *_ in P if U.get(f"e89{SFX}_c_{tag}", task) is not None)
    print(f"[blind] {len(P)} targets | {nx} external among {8*len(P)} slots ({100*nx/(8*len(P)):.1f}%, pool share 9.5%) | "
          f"{need} new probes, {8*len(P)-need} reused from EXP-32 | {done_c} commits done", flush=True)
    if dry:
        w = sum(1 for *_, b, _, _ in P if b < 0.9)
        print(f"[blind] weak targets first: {w} of {len(P)} | rollouts ~ {need*10 + (len(P)-done_c)*25}")
        return
    for tag, tk, task, suite, norm, b, cs, need in P:
        for i in range(0, len(need), U.CHUNK):
            jobs, built = [], []
            for c in need[i:i + U.CHUNK]:
                p = U.build(f"b80{SFX}_{tag}_{key(c)}", tk, [(c, ref / U.resid_norm(c))]); built.append(p)
                jobs.append({"name": f"e89{SFX}_p_{tag}_{key(c)}", "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": [task]})
            U.run(jobs, U.PROBE_R, suite, norm)
            for p in built: pathlib.Path(p).unlink(missing_ok=True)
        sc = sorted(((score(tag, task, c), c) for c in cs if score(tag, task, c) is not None), key=lambda z: -z[0])
        if len(sc) < 6: continue
        if sc[0][0] <= b:
            print(f"[{tag}] base {b*100:3.0f}% best {sc[0][0]*100:3.0f}% -> ABSTAIN", flush=True); continue
        if U.get(f"e89{SFX}_c_{tag}", task) is None:
            top = [c for _, c in sc[:3]]
            p = U.build(f"b80c{SFX}_{tag}", tk, [(c, (ref / U.resid_norm(c)) / 3) for c in top])
            U.run([{"name": f"e89{SFX}_c_{tag}", "method": "average", "params": {}, "adapters": [],
                    "adapter_paths": [p], "eval_scenes": [task]}], U.COMMIT_R, suite, norm)
            pathlib.Path(p).unlink(missing_ok=True)
        cm = U.get(f"e89{SFX}_c_{tag}", task)
        print(f"[{tag}] base {b*100:3.0f}% -> {'--' if cm is None else f'{cm*100:3.0f}%'}  "
              f"ext-in-draw {sum(1 for c in cs if not isinstance(c,int))}/8", flush=True)
    print("[blind] done")

if __name__ == "__main__":
    main("--dry" in sys.argv)
