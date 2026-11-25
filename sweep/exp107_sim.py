#!/usr/bin/env python3
"""EXP-107 design check (offline, ground truth): batch-friendly certified selection rules WITH a headroom gate,
before the live run. Same simulator as EXP-105 (15 selection / 10 held-out states, budget 80 incl. the target).
Gate: 10 target rollouts first; if >= 9/10 succeed -> keep the target (no search).
Rules (batches of 5 rollouts = one orchestrator job):
  sh8   8 random donors x 5 -> top 3 x 5 more -> top 1 x 5 more; target topped up to 15  (10+40+15+5+5 = 75)
  sh12  12 donors x 5 (60) -> top 1 x 5 more (5); target topped up to 15            (10+60+5+5 = 80)
  ttb8  batched top-two Thompson over 8 donors, batches of 5, until the budget
Decision: certified, merge the posterior-best donor only if P(p_d > p_target) >= 1 - delta.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp105_seqsim import load, Env, decide, B, NSEL
from subset_probe import TARGETS as WEAK

def gate(env):
    o = env.pull(0, 10); st = {0: [o.sum(), len(o)]}
    return st, o.sum() >= 9

def sh(env, rng, guard, n0, rounds):
    st, stop = gate(env)
    if stop: return 0
    arms = list(rng.choice(np.arange(1, env.X.shape[0]), n0, replace=False))
    for a in arms: o = env.pull(a, 5); st[a] = [o.sum(), len(o)]
    for keep in rounds:
        arms = sorted(arms, key=lambda a: -(st[a][0] + 1) / (st[a][1] + 2))[:keep]
        for a in arms: o = env.pull(a, 5); st[a][0] += o.sum(); st[a][1] += len(o)
    o = env.pull(0, 5); st[0][0] += o.sum(); st[0][1] += len(o)
    return decide({a: tuple(v) for a, v in st.items() if a == 0 or a in arms}, rng, guard)

def ttb(env, rng, guard, n0):
    st, stop = gate(env)
    if stop: return 0
    arms = [0] + list(rng.choice(np.arange(1, env.X.shape[0]), n0, replace=False))
    for a in arms[1:]: st[a] = [0.0, 0]
    while env.used + 5 <= B:
        S_ = np.array([st[a][0] for a in arms]); N_ = np.array([st[a][1] for a in arms]); live = N_ + 5 <= NSEL
        if not live.any(): break
        th = np.where(live, rng.beta(1 + S_, 1 + N_ - S_), -1); i = int(th.argmax())
        if rng.random() < 0.5:
            for _ in range(50):
                th2 = np.where(live, rng.beta(1 + S_, 1 + N_ - S_), -1); j = int(th2.argmax())
                if j != i: i = j; break
        a = arms[i]; o = env.pull(a, 5); st[a][0] += o.sum(); st[a][1] += len(o)
    return decide({a: tuple(v) for a, v in st.items() if v[1] > 0}, rng, guard)

RULES = {"sh8": lambda e, r, g: sh(e, r, g, 8, [3, 1]), "sh12": lambda e, r, g: sh(e, r, g, 12, [1]),
         "ttb8": lambda e, r, g: ttb(e, r, g, 8), "ttb12": lambda e, r, g: ttb(e, r, g, 12)}

def main():
    T = list(WEAK); Xs = {t: load(t) for t in T}; out = {}
    for name, fn in RULES.items():
        for guard in ("mean", 0.2, 0.1):
            g, m, h, u, gm = [], [], [], [], []
            for t in T:
                for sd in range(300):
                    env = Env(Xs[t], np.random.default_rng(hash((t, sd)) % 2**32))
                    a = fn(env, np.random.default_rng(hash((t, sd, name, str(guard))) % 2**32), guard)
                    g.append(env.gain(a)); m.append(a != 0); h.append(env.harmful(a)); u.append(env.used)
                    if a != 0: gm.append(env.gain(a))
            print(f"{name:6s} guard {str(guard):5s}: gain {np.mean(g):+5.2f}pp | merge {100*np.mean(m):4.1f}% | gain when merged {np.mean(gm):+5.1f}pp | harmful {100*np.mean(h):4.1f}% | rollouts {np.mean(u):4.1f}", flush=True)

if __name__ == "__main__":
    main()
