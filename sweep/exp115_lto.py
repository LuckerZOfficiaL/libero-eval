#!/usr/bin/env python3
"""EXP-115 — LEAVE-TEST-OUT shared part (user requirement 10-14): every merged subset computes S16 from the native library
(76 LIBERO-90 fleet adapters) MINUS the subset's members and their duplicate copies; ours_lto = S16_lto + alpha * sum r_i
(residuals w.r.t. that subspace). Baselines are unaffected (no S16) and reuse their EXP-112 / EXP-113 records.

PRE-REGISTERED (10-14, before any rollout):
  val    : alpha in {0.5/k, 1/k, 2/k} on EXP-112's 3 validation subsets per k (k = 2, 4, 8); best per k by mean success
  test   : EXP-112's 12 test subsets per k with the validated alpha  -> comparison vs every baseline (EXP-112 records)
  ctrl   : EXP-113 conditions A, B, D (k=8) with the k=8 alpha       -> vs averaging (EXP-113 records)
  hetero : condition B with the pool = 76 fleet + 30 zero-init externals, minus the subset's members
  10 episodes per member; validation on states 25..34, everything else on states 0..9.
Stages: E115_STAGE val | test | ctrl | hetero | report.  Env: E113_GPUS / E113_ROUND / E113_DEV (via exp113), E115_PART.
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X                                           # library = fleet + zero-init ext + S16-init ext
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
from la_pair_features import load
DUP = {7: 18, 18: 7, 29: 62, 62: 29}
GRID = (0.5, 1.0, 2.0)

class LTOBuilder(E.Builder):
    def __init__(self, hetero=False):
        super().__init__(); self.hetero = hetero; self.fleet = {}
    def ffac(self, t):
        if t not in self.fleet: self.fleet[t] = load(t)
        return self.fleet[t]
    def pool(self, members):
        mem = {E.LIB[i]["key"] for i in members if isinstance(E.LIB[i]["key"], int)}
        ex = mem | {DUP[t] for t in mem if t in DUP}
        P = [self.ffac(t) for t in range(76) if t not in ex]
        if self.hetero: P += [self.fac(i) for i in range(X.NF, X.NF + X.NE) if i not in members]   # zero-init externals
        return P
    def _delta(self, m, p, members, s_):
        if m != "ours_lto": return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        Bs = [f[s_][0].to(E.DEV) for f in F]; As = [f[s_][1].to(E.DEV) for f in F]
        P = self.pool(members)
        return MM.ours_lgo(Bs, As, p["alpha"], [f[s_][0].to(E.DEV) for f in P], [f[s_][1].to(E.DEV) for f in P])

def tag(prefix, a): return f"{prefix}-ours_lto_a{a:.3g}"

def run(items, n, offset, split, hetero=False):
    """items: [(tag, k, j, members, alpha)]"""
    items = [it for it in items if any(E.rec(nm, s) is None for nm, su, scs, ch in E.member_jobs(it[0], it[1], split, it[2], it[3]) for s in scs)]
    print(f"[e115] {split}{' hetero' if hetero else ''}: {len(items)} merged models", flush=True)
    if not items: return
    B = LTOBuilder(hetero); R = int(os.environ.get("E113_ROUND", "3")); rounds = [items[r:r + R] for r in range(0, len(items), R)]
    def build_round(rd):
        js = []
        for tg, k, j, S, a in rd:
            out = E.PRE / f"e115_{tg}_k{k}_{split}{j}"; t1 = time.time(); B.build("ours_lto", {"alpha": a}, None, S, out)
            for name, su, scenes, ch in E.member_jobs(tg, k, split, j, S):
                if all(E.rec(name, s) is not None for s in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e115] built {tg} k={k} {split}{j} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, n, offset, f"e115_{split}{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e115] {split} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def val_alpha(k):
    sc = {}
    for a in GRID:
        v = [np.mean(list(E.member_success(tag("c115", a / k), k, "val", j, S).values()) or [np.nan]) for j, S in enumerate(E.subsets(k, "val"))]
        sc[a] = np.nan if any(np.isnan(v)) else float(np.mean(v))
    ok = {a: v for a, v in sc.items() if not np.isnan(v)}
    if len(ok) < len(GRID): return None, sc
    return max(ok, key=ok.get) / k, sc

def main():
    st = os.environ.get("E115_STAGE", "val"); U._clear_orphaned_builds()
    if st == "val":
        run([(tag("c115", a / k), k, j, S, a / k) for k in (2, 4, 8) for a in GRID for j, S in enumerate(X.ORIG_VAL[k])], 10, 25, "val")
    elif st in ("test", "ctrl", "hetero"):
        al = {}
        for k in (2, 4, 8):
            al[k], sc = val_alpha(k); print(f"[e115] k={k}: validated alpha {al[k]} (grid " + ", ".join(f"{a}/k {v:.3f}" for a, v in sc.items()) + ")", flush=True)
        if st == "test":
            run([(tag("c115", al[k]), k, j, S, al[k]) for k in (2, 4, 8) for j, S in enumerate(X.ORIG_TEST[k])], 10, 0, "test")
        elif st == "ctrl":
            run([(tag(f"c115{c}", al[8]), 8, j, S, al[8]) for c in "ABD" for j, S in enumerate(X.subsets(c))], 10, 0, "test")
        else:
            run([(tag("c115Bh", al[8]), 8, j, S, al[8]) for j, S in enumerate(X.subsets("B"))], 10, 0, "test", hetero=True)
    else:
        report()
    print(f"[e115] {st} DONE", flush=True)

def report():
    from scipy import stats
    import io, contextlib
    sg = X.single()
    def ret(tg, k, j, S, which="all"):
        s = E.member_success(tg, k, "test", j, S)
        v = [s[i] / sg[i] for i in S if i in s and sg.get(i, 0) > 0 and (which == "all" or (which == "fleet") == (i < X.NF))]
        return np.mean(v) if v else np.nan
    al = {k: val_alpha(k)[0] for k in (2, 4, 8)}
    with contextlib.redirect_stdout(io.StringIO()):
        best = E.best_settings(["avg", "ta", "ties", "dare_ties", "cart", "knots_ties", "tsv", "iso_c", "iso_cts", "ct_merge", "ours"])
    L = [f"=== EXP-115 leave-test-out shared part: validated alpha {al} ===", "EXP-112 test subsets: ours_lto vs each method (paired over subsets)"]
    for ks in ((2,), (4,), (8,), (2, 4, 8)):
        L.append(f"  k in {ks}:")
        for b in ["ours", "avg", "ta", "ties", "dare_ties", "cart", "knots_ties", "tsv", "iso_c", "iso_cts", "ct_merge"]:
            d = []
            for k in ks:
                mm, p, c = best[(b, k)]
                for j, S in enumerate(X.ORIG_TEST[k]):
                    d.append(100 * (ret(tag("c115", al[k]), k, j, S) - ret(E.tag(mm, p, c), k, j, S)))
            d = np.array(d); ok = np.isfinite(d)
            if ok.sum() < 2: L.append(f"    vs {b:10s} incomplete"); continue
            L.append(f"    ours_lto - {b:10s} {np.nanmean(d):+6.1f}pp  wins {int((d[ok]>0).sum()):2d}/{ok.sum()}  Wilcoxon p={stats.wilcoxon(d[ok]).pvalue if np.any(d[ok]) else float('nan'):.4f}")
    L.append("EXP-113 controls (k=8): ours_lto vs averaging")
    for c in "ABD":
        d = np.array([100 * (ret(tag(f"c115{c}", al[8]), 8, j, S) - ret(f"c113{c}-{E.tag('avg', {}, None)}", 8, j, S)) for j, S in enumerate(X.subsets(c))]); ok = np.isfinite(d)
        if ok.sum() > 1: L.append(f"  {c}: {np.nanmean(d):+6.1f}pp (wins {int((d[ok]>0).sum())}/{ok.sum()}, Wilcoxon p={stats.wilcoxon(d[ok]).pvalue if np.any(d[ok]) else float('nan'):.4f})")
    d = np.array([100 * (ret(tag("c115Bh", al[8]), 8, j, S) - ret(tag("c115B", al[8]), 8, j, S)) for j, S in enumerate(X.subsets("B"))]); ok = np.isfinite(d)
    if ok.sum() > 1: L.append(f"hetero pool (fleet + external, leave-test-out) minus native pool, condition B: {np.nanmean(d):+6.1f}pp (wins {int((d[ok]>0).sum())}/{ok.sum()}, p={stats.wilcoxon(d[ok]).pvalue if np.any(d[ok]) else float('nan'):.4f})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp115_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
