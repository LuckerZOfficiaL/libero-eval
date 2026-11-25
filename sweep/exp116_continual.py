#!/usr/bin/env python3
"""EXP-116 — training-free CONTINUAL merging with our method (user plan 10-14).

Our merge is sum-based, so adding tasks one at a time gives exactly the model of the set seen so far (order-independent;
it can be maintained with running sums at constant memory). The experiment therefore tracks performance as the set grows.

PRE-REGISTERED (10-14, before any rollout):
  sequences  4 seeded random sequences; each = 16 LIBERO-90 fleet tasks already merged ("old") + 8 zero-init external tasks
             (LIBERO-Goal / Object / Spatial) added one at a time ("new")
  checkpoints after t = 0, 4, 8 additions; the model holds k = 16 + t tasks; evaluated: a fixed 8 of the 16 LIBERO-90 tasks
             (forgetting) + every new task added so far (10 episodes, initial states 0..9)
  methods (user 10-14: add continual-merging and merging-literature baselines; rules fixed, no tuning in a continual setting)
             avg     : W = base + (1/k) sum_i D_i
             ta      : lambda = 1.33/k (EXP-112 validated rule at k=8)
             ties, dare_ties : norm-matched to averaging x 1.33 (EXP-112 k=8 choice)
             cart    : lambda 0.25, gamma 0.08 (EXP-112 k=8 choice)
             maxabs  : per-entry max-magnitude update (training-free MagMax, ECCV 2024), norm-matched x 1.33
             opcm    : orthogonal-projection continual merging (Tang et al. 2025), alpha 0.5 (official default), applied in
                       arrival order (old tasks in a seeded order first, then new tasks), official norm-based rescaling
             ours_lib: shared part S16 from the LIBERO-90 library only (76 fleet adapters; the anchor so far)   (variant 2)
             ours_all: shared part from the LIBERO-90 library + every new task added so far                     (variant 1)
             both: W = base + S + (1/k) * sum of two-sided residuals w.r.t. that S  (alpha = 1/k, EXP-112's choice; no tuning)
             (at t = 0 the two variants coincide -> evaluated once)
  metrics    old-task success / retention (forgetting = drop vs t = 0), new-task success / retention, overall mean
  predictions ours_lib protects LIBERO-90 best; ours_all forgets LIBERO-90 slowly as new tasks shrink the shared part;
             averaging forgets most. Single-task baselines: EXP-112 base records.
Env: E116_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
from la_pair_features import load
NSEQ, NOLD, NNEW, NEVAL, CHECK = 4, 16, 8, 8, (0, 4, 8)
METHODS = [("avg", {}, None), ("ta", None, None), ("ties", E.LINEAR["ties"], 1.33), ("dare_ties", E.LINEAR["dare_ties"], 1.33),
           ("cart", {"lam": 0.25, "gamma": 0.08}, None), ("maxabs", {"lam": 1.0}, 1.33), ("opcm", {"alpha": 0.5}, None),
           ("ours_lib", {}, None), ("ours_all", {}, None)]

SCALE = 4                                                              # 11-06: scaling pseudo-order (EXP-148)
LIBSCALE = {5: 32, 6: X.NF}                                            # 11-07 EXP-150: bigger library (32 / all 74) + order 0's 8 new
TCUT = {7 + 4 * (t // 2 - 1) + s: (s, t) for t in (2, 4, 6) for s in range(NSEQ)}   # 11-08 EXP-151/152: order s, first t new only
CONT = {19: 14, 20: 18}                                                # 11-08 EXP-151: SCALE's 22 new skills arriving per order (8 / 14 / 18 / 22)
# 11-21 EXP-167: the stream on orders 1-3 = order o's library, the same 22 new skills arriving order o's 8 first, then orders
# o+1, o+2, o+3 (mod 4) deduplicated, cut at 14 / 18 / 22 like CONT (arrival 1 = order o itself): 21-23 / 24-26 / 27-29
CONTO = {21 + 3 * (o - 1) + a: (o, n) for o in (1, 2, 3) for a, n in enumerate((14, 18, 22))}
LCONT = {30: 14, 31: 18, 32: 22}                                       # 11-22 EXP-170: the stream on the 32-skill library (5's), SCALE's new as CONT (arrival 1 = 5)
FULL = {33: 5, 34: SCALE}                                              # 11-22 EXP-171: 5th arrival = all 30 externals (SCALE's 22 + the 8 never drawn, seeded) on 5's / SCALE's library
LIBSCALE1, LCONT1 = {35: 32}, {36: 14, 37: 18, 38: 22}                 # 11-22 EXP-171: the 32-skill library on order 1 (+ order 1's 8 new) and its stream (CONTO 21-23's order)
def sequence(s):
    if s == SCALE:                                                     # order 0's library + every order's new skills (22)
        u = []
        for s_ in range(NSEQ): u += [i for i in sequence(s_)[1] if i not in u]
        return sequence(0)[0], u
    if s in CONT: return sequence(SCALE)[0], sequence(SCALE)[1][:CONT[s]]
    if s in LCONT: return sequence(5)[0], sequence(SCALE)[1][:LCONT[s]]
    if s in FULL:
        u = sequence(SCALE)[1]; rest = [i for i in range(X.NF, X.NF + X.NE) if i not in u]
        return sequence(FULL[s])[0], u + [int(x) for x in np.random.default_rng(11710).permutation(rest)]
    if s in LIBSCALE1:                                                 # order 1's 16 first, then seeded extra fleet skills
        old, new = sequence(1); rest = [i for i in range(X.NF) if i not in old]
        return old + [int(x) for x in np.random.default_rng(11711).permutation(rest)][:LIBSCALE1[s] - len(old)], new
    if s in LCONT1: return sequence(35)[0], sequence(23)[1][:LCONT1[s]]
    if s in CONTO:
        o, n = CONTO[s]; u = []
        for s_ in range(o, o + NSEQ): u += [i for i in sequence(s_ % NSEQ)[1] if i not in u]
        return sequence(o)[0], u[:n]
    if s in TCUT:                                                      # 7-10 / 11-14 / 15-18 = orders 0-3 at t = 2 / 4 / 6 (nested)
        old, new = sequence(TCUT[s][0]); return old, new[:TCUT[s][1]]
    if s in LIBSCALE:                                                  # order 0's 16 first, then seeded extra fleet skills (nested)
        old, new = sequence(0); rest = [i for i in range(X.NF) if i not in old]
        return old + [int(x) for x in np.random.default_rng(11650).permutation(rest)][:LIBSCALE[s] - len(old)], new
    rng = np.random.default_rng(11600 + s)
    old = [int(x) for x in rng.choice(range(X.NF), NOLD, replace=False)]                   # seeded order (OPCM order)
    new = [int(x) for x in rng.choice(range(X.NF, X.NF + X.NE), NNEW, replace=False)]      # arrival order
    return old, new

class CBuilder(E.Builder):
    def __init__(self):
        super().__init__(); self.fleet = {}
    def ffac(self, t):
        if t not in self.fleet: self.fleet[t] = load(t)
        return self.fleet[t]
    def scale(self, m, p, c, members):
        if m != "opcm": return super().scale(m, p, c, members)
        key = f"opcm|{','.join(map(str, members))}"                        # official norm-based lambda: mean task norm / ||U||
        if key not in self.norms:
            nu = nt = 0.0
            with torch.no_grad():
                for s_ in self.ck:
                    d = self._delta(m, p, members, s_)
                    if d is None: continue
                    nu += float(d.norm() ** 2)
                    nt += sum(float((self.fac(i)[s_][0] @ self.fac(i)[s_][1]).norm() ** 2) for i in members if s_ in self.fac(i))
            tn = np.mean([np.sqrt(sum(float((self.fac(i)[s_][0] @ self.fac(i)[s_][1]).norm() ** 2) for s_ in self.ck if s_ in self.fac(i))) for i in members])
            self.norms[key] = float(tn / max(np.sqrt(nu), 1e-30))
        return self.norms[key]
    def _delta(self, m, p, members, s_):
        if m == "opcm":
            F = [self.fac(i) for i in members]
            if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
            return MM.opcm_unnorm([f[s_][0].to(E.DEV) for f in F], [f[s_][1].to(E.DEV) for f in F], p["alpha"])
        if m not in ("ours_lib", "ours_all"): return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): return None
        Bs = [f[s_][0].to(E.DEV) for f in F]; As = [f[s_][1].to(E.DEV) for f in F]; k = len(members)
        if m == "ours_lib": return MM.ours(Bs, As, 1.0 / k, U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV))
        P = [self.ffac(t) for t in range(76)] + [self.fac(i) for i in members if i >= X.NF]          # library + new so far
        return MM.ours_lgo(Bs, As, 1.0 / k, [f[s_][0].to(E.DEV) for f in P], [f[s_][1].to(E.DEV) for f in P])

def items():
    """(tag, seq, t, method, params, c, ordered members, evaluated members)"""
    out = []
    for s in range(NSEQ):
        old, new = sequence(s)
        for t in CHECK:
            mem = old + new[:t]; ev = sorted(old[:NEVAL] + new[:t]); k = len(mem)
            for m, p, c in METHODS:
                if t == 0 and m == "ours_all": continue                         # identical to ours_lib at t = 0
                if m == "ta": p = {"lam": 1.33 / k}
                out.append((f"c116s{s}t{t}-{m}", s, t, m, p, c, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[7]) for sc in scs)]
    if os.environ.get("E116_SHARD"):
        keep, n_ = os.environ["E116_SHARD"].split("/"); keep = {int(x) for x in keep.split(",")}
        allit = items(); idx = {it[0]: q for q, it in enumerate(allit)}
        todo = [it for it in todo if idx[it[0]] % int(n_) in keep]
    print(f"[e116] {len(todo)} merged models", flush=True)
    if not todo: return
    B = CBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, p, c, mem, ev in rd:
            out = E.PRE / f"e116_{tg}"; t1 = time.time(); B.build(m, p, c, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e116] built {tg} ({len(mem)} tasks) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e116_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e116] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    sg = X.single(); L = ["=== EXP-116 training-free continual merging: 4 sequences (16 LIBERO-90 + 8 external added one by one) ===",
                          f"{'t':>2} {'method':9s} {'old success':>11s} {'old retention':>13s} {'new success':>11s} {'new retention':>13s} {'all success':>11s}"]
    for t in CHECK:
        for m in [x[0] for x in METHODS]:
            mm = "ours_lib" if (t == 0 and m == "ours_all") else m
            os_, or_, ns, nr, al = [], [], [], [], []
            for s in range(NSEQ):
                old, new = sequence(s); k = len(old) + t; ev = sorted(old[:NEVAL] + new[:t])
                sc = E.member_success(f"c116s{s}t{t}-{mm}", k, "test", s, ev)
                if len(sc) < len(ev): continue
                o = [i for i in ev if i < X.NF]; n = [i for i in ev if i >= X.NF]
                os_.append(np.mean([sc[i] for i in o])); or_.append(np.mean([sc[i] / sg[i] for i in o if sg.get(i, 0) > 0]))
                if n: ns.append(np.mean([sc[i] for i in n])); nr.append(np.mean([sc[i] / sg[i] for i in n if sg.get(i, 0) > 0]))
                al.append(np.mean(list(sc.values())))
            if not os_: L.append(f"{t:>2} {m:9s} incomplete"); continue
            f = lambda v: f"{100*np.mean(v):10.1f}%" if v else "         - "
            L.append(f"{t:>2} {m:9s} {f(os_)} {f(or_):>13s} {f(ns)} {f(nr):>13s} {f(al)}   (n_seq={len(os_)})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp116_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E116_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e116] {st} DONE", flush=True)
