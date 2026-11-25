#!/usr/bin/env python3
"""EXP-39 — the probe method on the S16-initialised new experts (targets), fleet adapters as donors.

The week's boundary results said the method fails whenever target and donors do not share the fleet's
shared subspace. EXP-37 showed that subspace is the fleet's warm-start init and that new experts trained
FROM it are pool members by construction (capture 0.885, cos-to-fleet 0.92). This is the payoff test:
does manufactured consistency let the method cross what used to be the boundary?

Protocol = the fleet protocol exactly (probe_selector76 / EXP-32): baseline @25, 8 random fleet
candidates (seeded per target, plain random draws so the result is about the probe, not the list)
probed pairwise @10 with energy-matched dose, commit top-3 @25 under the abstention guard
"best probe > baseline". Operator form by capture gate (these targets are S16-form). All 30 targets
are run, not only the weak ones: that is the reporting rule, and it also yields the (target, donor)
labels that grow the affinity training set from 76 to 106 targets.

Reusable: --targets is a glob over adapter dirs and --tag prefixes job names, so the same driver
scores the EXP-38 seed variants.
"""
import argparse
import glob
import hashlib
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
import unified_pool_driver as U

SUITE = {"libero_goal_no_noops": "libero_goal", "libero_object_no_noops": "libero_object",
         "libero_spatial_no_noops": "libero_spatial", "libero_90_no_noops": "libero_90"}


def h6(name):
    """Stable 6-digit id for a target directory name.

    NOT Python's hash(): that is salted per process, so job names would change between runs and every
    already-paid rollout would be re-run instead of resumed. Resumability is the whole reason these
    drivers key results by job name.
    """
    return hashlib.md5(name.encode()).hexdigest()[:6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="results/analysis/new_loras_s16/libero_*__*")
    ap.add_argument("--tag", default="e51")
    ap.add_argument("--report", default="results/analysis/newpool_probe_report.txt")
    a = ap.parse_args()
    dirs = sorted(pathlib.Path(p) for p in glob.glob(str(REPO / a.targets)))
    tg = []
    for d in dirs:
        ad = d / "adapter_model.safetensors"
        if not ad.exists():
            continue
        key = "np:" + d.name                       # str key -> routed through U.EXTPATH by U.fold
        U.EXTPATH[key] = str(ad)
        suite_dir, task = d.name.split("__", 1)
        task = task.split("__seed")[0]              # seed variants share the base task
        tg.append((d.name, key, task, SUITE[suite_dir]))
    if not tg:
        sys.exit("no targets matched")
    norm = U.FLEET_NORM
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    print(f"[newpool] {len(tg)} targets | donors = 76 fleet | 8 candidates each | ref {ref:.2f}", flush=True)
    for name, tk, task, suite in tg:
        tag = f"{a.tag}_{h6(name)}"
        cap = U.capture(tk)
        if U.get(f"{tag}_base", task) is None:
            p = U.build(f"np_base_{tag}", tk, [])
            U.run([{"name": f"{tag}_base", "method": "average", "params": {}, "adapters": [],
                    "adapter_paths": [p], "eval_scenes": [task]}], U.BASE_R, suite, norm)
            pathlib.Path(p).unlink(missing_ok=True)
        base = U.get(f"{tag}_base", task)
        if base is None:
            print(f"[skip] {name}: no baseline", flush=True); continue
        # candidates are drawn from the BASE task name, so seed variants (dir '...__seedK') probe the
        # SAME 8 fleet donors as their seed-0 adapter -- otherwise EXP-38's within-task comparison
        # would rest on whatever candidates happen to overlap. Job tags stay unique per variant.
        rng = np.random.default_rng(int(h6(name.split('__seed')[0]), 16))
        cands = sorted(int(x) for x in rng.choice(range(76), U.N_FLEET_CAND, replace=False))
        specs = [(f"{tag}_p{c}", c) for c in cands if U.get(f"{tag}_p{c}", task) is None]
        for i in range(0, len(specs), U.CHUNK):
            jobs, built = [], []
            for job, c in specs[i:i + U.CHUNK]:
                p = U.build(f"np_{job}", tk, [(c, ref / U.resid_norm(c))])
                built.append(p)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": [task]})
            U.run(jobs, U.PROBE_R, suite, norm)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)
        sc = sorted(((U.get(f"{tag}_p{c}", task), c) for c in cands
                     if U.get(f"{tag}_p{c}", task) is not None), key=lambda z: -z[0])
        if not sc:
            continue
        top = sc[:U.N_PICK]
        if sc[0][0] <= base:
            print(f"[{name[:40]:<40}] cap {cap:.2f} base {base*100:3.0f}% best {sc[0][0]*100:3.0f}% -> ABSTAIN", flush=True)
        else:
            if U.get(f"{tag}_c", task) is None:
                p = U.build(f"np_c_{tag}", tk, [(c, (ref / U.resid_norm(c)) / U.N_PICK) for _, c in top])
                U.run([{"name": f"{tag}_c", "method": "average", "params": {}, "adapters": [],
                        "adapter_paths": [p], "eval_scenes": [task]}], U.COMMIT_R, suite, norm)
                pathlib.Path(p).unlink(missing_ok=True)
            cm = U.get(f"{tag}_c", task)
            print(f"[{name[:40]:<40}] cap {cap:.2f} base {base*100:3.0f}% best {sc[0][0]*100:3.0f}% -> "
                  f"{'--' if cm is None else f'{cm*100:3.0f}%'}  top3 {[c for _, c in top]}", flush=True)
        report(tg, a)
    report(tg, a)


def report(tg, a):
    from scipy import stats
    rows = []
    for name, tk, task, suite in tg:
        tag = f"{a.tag}_{h6(name)}"
        base = U.get(f"{tag}_base", task)
        if base is None:
            continue
        probes = {c: U.get(f"{tag}_p{c}", task) for c in range(76)}
        probes = {c: v for c, v in probes.items() if v is not None}
        if len(probes) < U.N_FLEET_CAND:
            continue
        best = max(probes.values())
        cm = U.get(f"{tag}_c", task)
        abst = best <= base
        final = base if abst else cm
        if final is None:
            continue
        rows.append(dict(name=name, suite=suite, base=base, best=best, commit=cm, abstain=abst,
                         final=final, probes=probes))
    L = [f"=== EXP-39 ({a.tag}): probe method on S16-init new experts, fleet donors ===",
         "baseline@25, 8 random fleet candidates @10 (energy-matched), commit top-3 @25, guard best>base", ""]
    if rows:
        d = np.array([100 * (r["final"] - r["base"]) for r in rows])
        base = np.array([100 * r["base"] for r in rows]); weak = base < 90
        t_ = stats.ttest_1samp(d, 0) if len(d) > 1 else None
        L += [f"  n = {len(rows)} targets (partial)" if len(rows) < len(tg) else f"  n = {len(rows)} targets",
              f"  ALL   : {d.mean():+.2f}pp" + (f"  (SE {stats.sem(d):.2f}, p={t_.pvalue:.4f})" if t_ else ""),
              f"  improved {int((d>0).sum())} / flat {int((d==0).sum())} / harmed {int((d<0).sum())}; "
              f"abstained {sum(r['abstain'] for r in rows)}"]
        for nm, m in (("WEAK  (base<90%)", weak), ("STRONG(base>=90%)", ~weak)):
            if m.sum():
                L.append(f"  {nm} n={int(m.sum()):<3}: {d[m].mean():+.2f}pp  improved {int((d[m]>0).sum())} harmed {int((d[m]<0).sum())}")
        cm = [r for r in rows if not r["abstain"] and r["commit"] is not None]
        if cm:
            L.append(f"  probe overstatement (best probe - commit) on {len(cm)} committed: "
                     f"{np.mean([100*(r['best']-r['commit']) for r in cm]):+.1f}pp  (fleet-internal +5.0, cross-pool ~+19)")
        L += ["", f"  {'target':<46}{'suite':<15}{'base':>6}{'best':>6}{'final':>7}"]
        for r in sorted(rows, key=lambda r: r["base"]):
            L.append(f"  {r['name'][:44]:<46}{r['suite']:<15}{100*r['base']:>5.0f}%{100*r['best']:>5.0f}%"
                     f"{100*r['final']:>6.0f}%{'  abstain' if r['abstain'] else ''}")
        L += ["", "labels: " + json.dumps({r["name"]: {"base": r["base"], "probes": r["probes"]} for r in rows})]
    (REPO / a.report).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
