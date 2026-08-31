#!/usr/bin/env python3
"""E4 — affinity hypernetwork: learn to pick donors, so selection needs no rollouts.

TASK. Given a target LoRA T and k candidate LoRAs, emit k affinity scores; merge the top-n.
Deployment question is "which of these candidates is best for T", NOT "what number will T score".

TWO DESIGN CONSTRAINTS, both forced by measurements rather than taste:

 1. ASYMMETRIC PAIR ENCODER. Affinity is not reciprocal: on matched-headroom reciprocal pairs the
    correlation between aff(A,B) and aff(B,A) is r = -0.005 (t23->t6 = +36pp while t6->t23 = -26pp).
    Any symmetric feature map (cosine, |fT-fp|, fT*fp alone) provably cannot represent that, so the
    encoder concatenates [fT, fp, fT-fp, fT*fp] -- the SIGNED difference is what breaks symmetry.

 2. WITHIN-TARGET LISTWISE LOSS, not absolute regression. Target identity explains LOOCV R^2 = +0.411
    of absolute affinity while donor identity explains -0.075. A regressor therefore minimises its loss
    by learning "how fragile is this target" and ignoring the pair interaction -- the only thing we need.
    A softmax over each target's own candidate list cancels the target offset by construction.

LABELS. 604 measured (target, donor) affinities from e27: affinity = pairwise-probe success - baseline,
each from 10 real rollouts. This is behavioural supervision, amortised. Training with rollouts in the
loop would need all 5,700 ordered pairs = ~28 days of exclusive GPU.

HONEST BASELINES. With ~525 training labels a neural net is not obviously the right tool, so every run
reports: random selection, an oracle, and a ridge ranker on the same features and same loss geometry.
The network earns its place only by beating those.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path("/home/lzhou/libero-eval")
torch.set_num_threads(4)


# ---------------------------------------------------------------- data
def load(labels_file=None):
    z = np.load(REPO / "results/analysis/hypernet_feats.npz")
    F = torch.tensor(z["F"], dtype=torch.float32)
    if labels_file:
        # consolidated labels: every pairwise measurement we own, repeats averaged (see
        # consolidate_labels.py). 'w' = total rollouts behind the label -> used as a quality weight.
        L = json.load(open(labels_file))
        import collections
        by_t = collections.defaultdict(list)
        for k, v in L.items():
            t, p = map(int, k.split("_"))
            by_t[t].append((p, v["aff"], v["w"]))
        groups = []
        for t, rows in by_t.items():
            if len(rows) < 4:
                continue
            o = np.random.default_rng(9000 + t).permutation(len(rows))
            rows = [rows[i] for i in o]
            groups.append((t, [r[0] for r in rows],
                           np.array([r[1] for r in rows], dtype=np.float32),
                           np.array([r[2] for r in rows], dtype=np.float32)))
        plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
        scenes = {a["task_index"]: a["scenes"] for a in plan}
        base = {}
        for ln in open(REPO / "results/baseline_results.jsonl"):
            r = json.loads(ln)
            if r.get("status") == "ok":
                base[r["scene"]] = r["success_rate"]
        def b(t):
            ss = [s for s in scenes[t] if s in base]
            return sum(base[s] for s in ss) / len(ss)
        return F, groups, {t: b(t) for t in range(76)}
    R = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                   .read().split("probe rankings: ")[1].strip())
    plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
    scenes = {a["task_index"]: a["scenes"] for a in plan}
    base = {}
    for ln in open(REPO / "results/baseline_results.jsonl"):
        r = json.loads(ln)
        if r.get("status") == "ok":
            base[r["scene"]] = r["success_rate"]

    def b(t):
        ss = [s for s in scenes[t] if s in base]
        return sum(base[s] for s in ss) / len(ss)

    groups = []                      # one entry per target: (T, candidate ids, affinities in pp)
    for t, v in R.items():
        T = int(t)
        cand = [int(p) for p, _ in v]
        aff = [(s - b(T)) * 100 for _, s in v]
        # CRITICAL: the report stores candidates already sorted by probe score, so list position
        # leaks the label -- a degenerate (constant-score) model would "recover" the oracle purely
        # from argsort tie-breaking. Shuffle deterministically to destroy that channel.
        o = np.random.default_rng(9000 + T).permutation(len(cand))
        groups.append((T, [cand[i] for i in o], np.array(aff, dtype=np.float32)[o],
                       np.full(len(cand), 10.0, dtype=np.float32)))
    return F, groups, {t: b(t) for t in range(76)}


def split(groups, baseline, n_weak=5, n_strong=5, seed=0):
    """Hold out whole TARGETS (their candidate lists are never seen), stratified weak/strong."""
    rng = np.random.default_rng(seed)
    weak = [g[0] for g in groups if baseline[g[0]] < 0.90]
    strong = [g[0] for g in groups if baseline[g[0]] >= 0.90]
    ho = set(rng.choice(weak, n_weak, replace=False).tolist()
             + rng.choice(strong, n_strong, replace=False).tolist())
    return [g for g in groups if g[0] not in ho], [g for g in groups if g[0] in ho], sorted(ho)


# ---------------------------------------------------------------- model
class AffinityNet(nn.Module):
    """Asymmetric pair scorer. Shared per-adapter trunk (T2L's shared-trunk idea), then a head on the
    ordered pair. score(T,p) != score(p,T) by construction."""

    def __init__(self, d_in, d_emb=32, d_hid=64, p_drop=0.3):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, d_emb), nn.GELU(), nn.Dropout(p_drop),
                                   nn.Linear(d_emb, d_emb))
        self.head = nn.Sequential(nn.Linear(4 * d_emb, d_hid), nn.GELU(), nn.Dropout(p_drop),
                                  nn.Linear(d_hid, 1))

    def forward(self, fT, fP):
        a, b = self.trunk(fT), self.trunk(fP)
        return self.head(torch.cat([a, b, a - b, a * b], -1)).squeeze(-1)


def binary_loss(scores, aff, lw):
    """PHASE-2-MOTIVATED OBJECTIVE. Exact ranking is partly unlearnable (label reliability 0.41) and,
    per EXP-22.6, unnecessary: each target has several near-equivalent good donors, and deployment only
    needs to find ONE of them (plus the abstention guard = 'does the best beat baseline'). So train the
    simpler, more reliable question: is this donor GOOD for this target (affinity > 0)?
    Per-pair BCE, weighted by the rollouts behind the label (25-rollout labels count 2.5x)."""
    y = torch.tensor((aff > 0).astype(np.float32))
    w = torch.tensor(lw / 10.0)
    return (w * torch.nn.functional.binary_cross_entropy_with_logits(
        scores, y, reduction="none")).sum() / w.sum()


def listwise_loss(scores, aff, tau=8.0, weight=1.0):
    """Cross-entropy between predicted ranking and the affinity-derived soft target, within one target's
    candidate list. tau is in pp; ~8pp is the scale of a real affinity difference.

    `weight` exists because most targets are SATURATED: 61 of 76 have baseline >=90%, and there every
    candidate probes at ~the same score (measured spread 0.0-0.1 vs 0.5-0.8 on weak targets). Those
    groups carry a near-uniform soft target, so at equal weight they dominate the objective and teach
    the model to emit a CONSTANT score -- which is exactly the collapse we observed (score sd 0.000).
    Weighting each group by its affinity spread lets flat groups contribute ~nothing instead."""
    tgt = torch.softmax(torch.tensor(aff) / tau, 0)
    return -weight * (tgt * torch.log_softmax(scores, 0)).sum()


# ---------------------------------------------------------------- metrics
def evaluate(score_fn, groups, n_pick=3):
    """Deployment-shaped metrics: what affinity do we actually obtain by taking the model's top-n?"""
    sp, top1, picked, best, rand, spread, hit, rhit = [], [], [], [], [], [], [], []
    for T, cand, aff, _w in groups:
        s = np.asarray(score_fn(T, cand), dtype=np.float64)
        order = np.argsort(-s)
        spread.append(s.std())
        if len(set(np.round(s, 9))) > 1:
            from scipy import stats
            sp.append(stats.spearmanr(s, aff).statistic)
        top1.append(aff[order[0]])
        picked.append(aff[order[:n_pick]].mean())
        best.append(np.sort(aff)[::-1][:n_pick].mean())
        rand.append(aff.mean())                       # expected affinity of a random pick
        good = aff > 0
        hit.append(float(good[order[:n_pick]].any()))  # picked at least one genuinely good donor
        # analytic P(random top-n contains a good donor)
        n, g = len(aff), int(good.sum())
        from math import comb
        rhit.append(1.0 - (comb(n - g, n_pick) / comb(n, n_pick) if n - g >= n_pick else 0.0))
    return dict(spearman=float(np.nanmean(sp)) if sp else float("nan"),
                top1=float(np.mean(top1)), topn=float(np.mean(picked)),
                oracle=float(np.mean(best)), random=float(np.mean(rand)),
                spread=float(np.mean(spread)), hit=float(np.mean(hit)), rhit=float(np.mean(rhit)))


def fmt(name, m):
    return (f"  {name:<22} top-3 aff {m['topn']:+6.2f}pp | top-1 {m['top1']:+6.2f} | "
            f"spearman {m['spearman']:+.3f} | hit {m['hit']:.2f} (rand {m['rhit']:.2f}) | "
            f"score sd {m['spread']:.3f}")


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d_emb", type=int, default=32)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--blocks", default="WALS", help="feature blocks to use, subset of WALS")
    ap.add_argument("--n_cand", type=int, default=0,
                    help="subsample this many CANDIDATES per training target (0 = all 8)")
    ap.add_argument("--n_train", type=int, default=0,
                    help="subsample this many TRAINING targets (0 = all); for learning curves")
    ap.add_argument("--spread_w", type=float, default=1.0,
                    help="weight each training group by aff.std()**spread_w (0 = uniform)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--labels", default=None, help="consolidated pair_labels.json (default: e27 only)")
    ap.add_argument("--objective", default="listwise", choices=["listwise", "binary"])
    a = ap.parse_args()

    F, groups, baseline = load(a.labels)
    sl = json.loads(str(np.load(REPO / "results/analysis/hypernet_feats.npz")["slices"]))
    keep = np.concatenate([np.arange(*sl[b]) for b in a.blocks])
    F = F[:, keep]
    print(f"features {tuple(F.shape)} (blocks {a.blocks})   groups {len(groups)}   "
          f"labels {sum(len(g[1]) for g in groups)}")

    rows = []
    for seed in range(a.seeds):
        tr, te, ho = split(groups, baseline, seed=seed)
        if a.n_train and a.n_train < len(tr):
            sel = np.random.default_rng(1000 + seed).choice(len(tr), a.n_train, replace=False)
            tr = [tr[i] for i in sel]
        if a.n_cand:
            rg = np.random.default_rng(2000 + seed)
            sub = []
            for T, c, af, lw in tr:
                ix = rg.choice(len(c), min(a.n_cand, len(c)), replace=False)
                sub.append((T, [c[i] for i in ix], af[ix], lw[ix]))
            tr = sub
        torch.manual_seed(seed)
        net = AffinityNet(F.shape[1], d_emb=a.d_emb, p_drop=a.drop)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        curve = []
        for ep in range(a.epochs):
            net.train()
            perm = np.random.permutation(len(tr))
            tot = 0.0
            for i in perm:
                T, cand, aff, lw = tr[i]
                s = net(F[T].expand(len(cand), -1), F[cand])
                if a.objective == "binary":
                    loss = binary_loss(s, aff, lw)
                else:
                    w = 1.0 if a.spread_w == 0 else float(aff.std()) ** a.spread_w
                    loss = listwise_loss(s, aff, tau=a.tau, weight=w)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip); opt.step()
                tot += float(loss)
            if ep % max(1, a.epochs // 6) == 0 or ep == a.epochs - 1:
                curve.append((ep, tot / len(tr)))
        net.eval()

        def net_score(T, cand):
            with torch.no_grad():
                return net(F[T].expand(len(cand), -1), F[cand]).numpy()

        # ridge ranker on the same pair features, same listwise geometry (closed form, no tuning)
        Xtr, ytr = [], []
        for T, cand, aff, _lw in tr:
            fa = F[T].expand(len(cand), -1); fb = F[cand]
            Xtr.append(torch.cat([fa, fb, fa - fb, fa * fb], -1).numpy())
            ytr.append(aff - aff.mean())              # centre within target = same offset removal
        X = np.vstack(Xtr); y = np.concatenate(ytr)
        w = np.linalg.solve(X.T @ X + 10.0 * np.eye(X.shape[1]), X.T @ y)

        def ridge_score(T, cand):
            fa = F[T].expand(len(cand), -1); fb = F[cand]
            return torch.cat([fa, fb, fa - fb, fa * fb], -1).numpy() @ w

        mn, mr = evaluate(net_score, te), evaluate(ridge_score, te)
        mtr = evaluate(net_score, tr)
        rows.append((mn, mr))
        print(f"[seed {seed}] holdout {ho}")
        print(f"  loss {' -> '.join(f'{l:.3f}@{e}' for e, l in curve)}   (chance = {np.log(8):.3f})")
        print(fmt("hypernet TRAIN", mtr) + f"   [train oracle {mtr['oracle']:+.2f}]")
        print(fmt("hypernet", mn)); print(fmt("ridge", mr))
        print(f"  {'random':<22} top-3 aff {mn['random']:+6.2f}pp   "
              f"{'oracle':<8} top-3 aff {mn['oracle']:+6.2f}pp")

    def agg(i, k):
        v = [r[i][k] for r in rows]
        return np.nanmean(v), np.nanstd(v) / np.sqrt(len(v))
    L = [f"=== E4 AFFINITY HYPERNETWORK — {a.tag} ===",
         f"blocks={a.blocks} d_emb={a.d_emb} drop={a.drop} tau={a.tau} lr={a.lr} wd={a.wd} "
         f"epochs={a.epochs} seeds={a.seeds}",
         f"held out 10 targets (5 weak + 5 strong) per seed; {sum(len(g[1]) for g in groups)} labels total", "",
         f"{'selector':<14}{'top-3 affinity':>16}{'top-1':>12}{'spearman':>12}"]
    for nm, i in (("hypernet", 0), ("ridge", 1)):
        m, e = agg(i, "topn"); m1, _ = agg(i, "top1"); s, _ = agg(i, "spearman")
        L.append(f"{nm:<14}{m:>+10.2f} ±{e:<4.2f}{m1:>+12.2f}{s:>12.3f}")
    m, e = agg(0, "random"); L.append(f"{'random':<14}{m:>+10.2f} ±{e:<4.2f}")
    m, e = agg(0, "oracle"); L.append(f"{'oracle':<14}{m:>+10.2f} ±{e:<4.2f}")
    out = "\n".join(L)
    (REPO / f"results/analysis/hypernet_{a.tag}.txt").write_text(out + "\n")
    print("\n" + out)


if __name__ == "__main__":
    main()
