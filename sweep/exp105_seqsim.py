#!/usr/bin/env python3
"""EXP-105 — sequential, certified donor selection, simulated OFFLINE on the EXP-95 ground truth (no new rollouts).

Arms per weak target t: the target alone (no merge; per-episode outcomes of e92_base_f{t}) and the 73 single-donor
merges (per-episode outcomes of e98_t{t}_d{d}), all on the same 25 LIBERO initial states. Per simulation seed the 25
states are split: 15 SELECTION states (an algorithm's rollouts are drawn from them, without replacement, in a
random order per arm) and 10 HELD-OUT states (only used to score the final decision, paired with the target).
Budget: 80 rollouts per target, INCLUDING rollouts of the target itself (needed to decide whether to merge at all).

Methods
  probe        7 random donors x 10 + target x 10; merge the best if its mean beats the target's (deployed-like)
  probe_raw    same, always merge the best donor (no guard)
  sh           successive halving: 16 random donors + target; rounds 2/3/4 rollouts per arm, keep the top half
  ttts         top-two Thompson sampling, Beta(1,1), 16 random donors + target, <= 15 rollouts per arm
  *_cert{d}    certified: merge only if the posterior P(p_donor > p_target | data) >= 1 - d; else keep the target
Scores (mean over 15 targets x seeds): held-out gain vs the target (pp; abstain = 0), merge rate, and harmful merges
(merged donor's held-out success below the target's by >= 2 of 10 states). References: random donor, oracle.
"""
import json, pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from reinforce_train import scenes as SC
from subset_probe import TARGETS as WEAK
import gt_analysis as A_
NSEL, NHO, B, SEEDS = 15, 10, 80, 400

def eps(job, t):
    for ln in open(REPO / f"results/merge/{job}.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok" and r["scene"] == SC[t][0] and r.get("success_step"): return np.array([s >= 0 for s in r["success_step"]], float)

def load(t):
    donors = [d for d in A_.UNIQUE if d != t]
    X = np.stack([eps(f"e92_base_f{t}", t)] + [np.array([s >= 0 for s in A_.rec(t, d)["success_step"]], float) for d in donors])
    return X                                                        # [1 + 73, 25]; row 0 = target alone

class Env:
    def __init__(self, X, rng):
        p = rng.permutation(X.shape[1]); self.sel, self.ho = p[:NSEL], p[NSEL:NSEL + NHO]
        self.X = X; self.order = {a: rng.permutation(self.sel) for a in range(X.shape[0])}; self.n = {}; self.used = 0
    def pull(self, a, k=1):
        i = self.n.get(a, 0); k = min(k, NSEL - i)
        out = self.X[a, self.order[a][i:i + k]]; self.n[a] = i + k; self.used += k; return out
    def gain(self, a):                                              # held-out gain of decision a (0 = keep target)
        return 0.0 if a == 0 else 100 * (self.X[a, self.ho].mean() - self.X[0, self.ho].mean())
    def harmful(self, a):
        return a != 0 and (self.X[0, self.ho].sum() - self.X[a, self.ho].sum()) >= 2

def post_better(s1, n1, s0, n0, rng, m=4000):                       # P(p1 > p0) under Beta(1+s, 1+n-s)
    return float((rng.beta(1 + s1, 1 + n1 - s1, m) > rng.beta(1 + s0, 1 + n0 - s0, m)).mean())

def decide(stats_, rng, guard):
    """stats_: {arm: (successes, n)}; returns arm to deploy under the guard"""
    s0, n0 = stats_[0]
    don = {a: v for a, v in stats_.items() if a != 0 and v[1] > 0}
    best = max(don, key=lambda a: ((don[a][0] + 1) / (don[a][1] + 2), don[a][1]))
    if guard == "none": return best
    if guard == "mean": return best if don[best][0] / don[best][1] > s0 / max(n0, 1) else 0
    return best if post_better(*don[best], s0, n0, rng) >= 1 - guard else 0

def run_probe(env, rng, guard):
    ds = rng.choice(np.arange(1, env.X.shape[0]), 7, replace=False); st = {0: (env.pull(0, 10).sum(), 10)}
    for d in ds: o = env.pull(d, 10); st[d] = (o.sum(), len(o))
    return decide(st, rng, guard)

def run_sh(env, rng, guard):
    arms = list(rng.choice(np.arange(1, env.X.shape[0]), 16, replace=False)); st = {0: [0.0, 0]}
    for a in arms: st[a] = [0.0, 0]
    for r in (2, 3, 4):
        for a in arms + [0]:
            o = env.pull(a, r); st[a][0] += o.sum(); st[a][1] += len(o)
        arms = sorted(arms, key=lambda a: -(st[a][0] + 1) / (st[a][1] + 2))[:max(1, len(arms) // 2)]
    return decide({a: tuple(v) for a, v in st.items() if a == 0 or a in arms}, rng, guard)

def run_ttts(env, rng, guard):
    arms = [0] + list(rng.choice(np.arange(1, env.X.shape[0]), 16, replace=False)); s = {a: 0.0 for a in arms}; n = {a: 0 for a in arms}
    for a in arms[:]:                                               # one initial rollout each
        o = env.pull(a, 1); s[a] += o.sum(); n[a] += len(o)
    A = np.array(arms)
    while env.used < B:
        S_ = np.array([s[a] for a in arms]); N_ = np.array([n[a] for a in arms]); live = N_ < NSEL
        th = np.where(live, rng.beta(1 + S_, 1 + N_ - S_), -1); i1 = int(th.argmax())
        if rng.random() < 0.5:                                      # top-two: resample until a different arm leads
            for _ in range(50):
                th2 = np.where(live, rng.beta(1 + S_, 1 + N_ - S_), -1); i2 = int(th2.argmax())
                if i2 != i1: i1 = i2; break
        a1 = int(A[i1]); o = env.pull(a1, 1); s[a1] += o.sum(); n[a1] += len(o)
    return decide({a: (s[a], n[a]) for a in arms}, rng, guard)

def main():
    rng = np.random.default_rng(105)
    T = list(WEAK); Xs = {t: load(t) for t in T}
    methods = {"probe": (run_probe, "mean"), "probe_raw": (run_probe, "none"),
               "sh": (run_sh, "mean"), "ttts": (run_ttts, "mean")}
    for d_ in (0.2, 0.1, 0.05):
        methods[f"sh_cert{d_}"] = (run_sh, d_); methods[f"ttts_cert{d_}"] = (run_ttts, d_); methods[f"probe_cert{d_}"] = (run_probe, d_)
    res = {m: {"gain": [], "merge": [], "harm": [], "used": []} for m in methods}
    ref = {"random donor": [], "oracle (best of 73 by all 25 states)": []}
    for t in T:
        X = Xs[t]
        for sd in range(SEEDS):
            for m, (fn, guard) in methods.items():
                r_ = np.random.default_rng(hash((t, sd)) % 2**32)            # same split/order for every method
                env = Env(X, r_); a = fn(env, np.random.default_rng(hash((t, sd, m)) % 2**32), guard)
                res[m]["gain"].append(env.gain(a)); res[m]["merge"].append(a != 0); res[m]["harm"].append(env.harmful(a)); res[m]["used"].append(env.used)
            env = Env(X, np.random.default_rng(hash((t, sd)) % 2**32))
            ref["random donor"].append(np.mean([env.gain(a) for a in range(1, X.shape[0])]))
            ref["oracle (best of 73 by all 25 states)"].append(env.gain(1 + int(np.argmax(X[1:].mean(1)))))
    L = [f"=== EXP-105 sequential donor selection, simulated on ground truth: {len(T)} weak targets x {SEEDS} splits ===",
         f"budget {B} rollouts incl. the target's own; selection on {NSEL} states, scored on {NHO} held-out states (paired with the target)",
         f"{'method':18s} {'gain pp':>8s} {'merge %':>8s} {'harmful merges %':>17s} {'rollouts':>9s}"]
    for m, r in res.items():
        L.append(f"{m:18s} {np.mean(r['gain']):+8.2f} {100*np.mean(r['merge']):8.1f} {100*np.mean(r['harm']):17.1f} {np.mean(r['used']):9.1f}")
    for k, v in ref.items(): L.append(f"{k:38s} {np.mean(v):+8.2f}")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp105_seqsim_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
