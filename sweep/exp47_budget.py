#!/usr/bin/env python3
"""EXP-46/47 analysis — probe reliability (fresh vs settled) and the honest probe-budget curve.

Two independent 10-rollout probe runs per (target, candidate): run A = e27/e54, run B = e57 (EXP-46).

(46) reliability: Spearman(A, B) per target. Settled fleet targets vs fresh S16-1.5k variants, three of
     them the same task. Fleet reference from EXP-22.4 same-adapter repeats: 0.41.

(47) budget: for b in {2,3,5,10}, subsample b of run A's 10 outcomes, pick top-3, and value those picks
     by run B. Selection and valuation come from INDEPENDENT rollouts, so bigger budgets get no unfair
     credit -- subsampling one run and scoring on the same run would inflate b=10 by construction.
     Reported against the oracle (best 3 by run B) and a random 3.
"""
import hashlib, json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
h6 = lambda n: hashlib.md5(n.encode()).hexdigest()[:6]
from reinforce_train import scenes as SC76, target_baseline

RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())
lab = json.loads((REPO / "results/analysis/newpool_var50_report.txt").read_text().split("labels: ", 1)[1].strip())
plan = json.load(open(REPO / "sweep/plan_ext.json"))["adapters"]
idx_of = {x["adapter_file"].split("/")[-2]: x for x in plan}

def runB(tag, c, scenes):
    f = REPO / f"results/merge/e57_{tag}_p{c}.jsonl"
    if not f.exists(): return None
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok": sr[d["scene"]] = d["success_rate"]
    ss = [s for s in scenes if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if len(ss) == len(scenes) else None

T = []   # (tag, cohort, base, {c: (A_rate, B_rate)})
for t in (6, 19, 50, 15, 56):
    A = {int(p): s for p, s in RANK[str(t)]}
    d = {c: (a, runB(f"f{t}", c, SC76[t])) for c, a in A.items()}
    if all(v[1] is not None for v in d.values()): T.append((f"f{t}", "settled", target_baseline(t), d))
for name, v in lab.items():
    tag = f"v{h6(name)}"; task = name.split("__", 1)[1].split("__seed")[0]
    A = {int(p): s for p, s in v["probes"].items()}
    d = {c: (a, runB(tag, c, [task])) for c, a in A.items()}
    if d and all(x[1] is not None for x in d.values()): T.append((tag, "fresh", v["base"], d))
if not T:
    sys.exit("no re-probe records yet")

print(f"=== EXP-46: probe test-retest reliability, two independent 10-rollout runs ===")
print(f"{'target':<10}{'cohort':>9}{'base':>6}{'rho(A,B)':>10}{'|A-B| mean':>12}{'best-donor agree':>18}")
rows = []
for tag, co, base, d in T:
    cs = sorted(d); a = np.array([d[c][0] for c in cs]); b = np.array([d[c][1] for c in cs])
    rho = stats.spearmanr(a, b).correlation
    agree = cs[int(np.argmax(a))] == cs[int(np.argmax(b))]
    rows.append((tag, co, base, rho, np.abs(a - b).mean(), agree))
    print(f"{tag:<10}{co:>9}{100*base:>5.0f}%{rho:>+10.2f}{100*np.abs(a-b).mean():>11.0f}pp{('same' if agree else 'differs'):>18}")
for co in ("settled", "fresh"):
    r = [x[3] for x in rows if x[1] == co and np.isfinite(x[3])]
    ag = [x[5] for x in rows if x[1] == co]
    if r: print(f"  {co:<8} mean rho {np.mean(r):+.2f} (n={len(r)})   best-donor agreement {100*np.mean(ag):.0f}%")
rs = [x[3] for x in rows if x[1] == "settled" and np.isfinite(x[3])]; rf = [x[3] for x in rows if x[1] == "fresh" and np.isfinite(x[3])]
if rs and rf:
    print(f"  settled - fresh: {np.mean(rs)-np.mean(rf):+.2f} (Welch p={stats.ttest_ind(rs, rf, equal_var=False).pvalue:.3f})")
    print(f"  reference: same-adapter test-retest on fleet targets = 0.41 (EXP-22.4)")

print(f"\n=== EXP-47: probe budget — select on run A at b rollouts, value on run B (independent) ===")
rng = np.random.default_rng(0); REP = 4000
print(f"{'budget/candidate':<18}{'rollouts/target':>16}{'value of top-3 (run B)':>24}{'vs random':>12}")
res = {}
for b in (2, 3, 5, 10):
    vals = []
    for tag, co, base, d in T:
        cs = sorted(d); a10 = np.array([round(10 * d[c][0]) for c in cs]); bB = np.array([d[c][1] for c in cs])
        for _ in range(REP // len(T)):
            est = np.array([rng.hypergeometric(k, 10 - k, b) / b for k in a10]) if b < 10 else a10 / 10.0
            order = np.argsort(-(est + 1e-9 * rng.random(len(est))))
            vals.append(100 * (bB[order[:3]].mean() - base))
    res[b] = np.mean(vals)
rnd = np.mean([100 * (np.mean([d[c][1] for c in list(d)[:3]]) - base) for tag, co, base, d in T])
orc = np.mean([100 * (np.mean(sorted([d[c][1] for c in d], reverse=True)[:3]) - base) for tag, co, base, d in T])
for b in (2, 3, 5, 10):
    print(f"  {b:>2} rollouts        {8*b:>14}   {res[b]:>+20.2f}pp{res[b]-rnd:>+11.2f}")
print(f"  random 3          {0:>14}   {rnd:>+20.2f}pp")
print(f"  oracle (run B)    {'--':>14}   {orc:>+20.2f}pp")
if res[10] - rnd > 0:
    for b in (2, 3, 5):
        print(f"  b={b}: {100*(res[b]-rnd)/(res[10]-rnd):.0f}% of the 10-rollout gain at {100*b/10:.0f}% of the cost")
