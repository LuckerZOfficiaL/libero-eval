#!/usr/bin/env python3
"""EXP-91 stage 2 — do linear-algebra pair scores rank donors the way the behavioural probe does?

Labels: every fleet-target x fleet-donor single-donor probe we own (success of S16 + r_t + c r_d):
e26/e27 probes (10 rollouts), e44_p (EXP-32), e89/e89d2-4 (EXP-80 draws), e75 (EXP-69/74 draws), e93 (25 states).
Repeats of a pair are pooled (rollout-weighted). Only within-target comparisons are used (baseline cancels).
Scores: every feature x band (all / vision / LLM-early / LLM-late / expert / head) from la_pair_features.npz,
both signs. Protocol: split targets in half (seeded); pick the best score on half A by mean within-target
Spearman; report it on half B (pre-registered), and the reverse; plus top-3 pick value vs random.
Reference: two independent probe runs agree at rho = 0.41 (EXP-22.4) -- the ceiling for any predictor.
"""
import glob, json, pathlib, re, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
N = 76; DUP = {7: 18, 18: 7, 29: 62, 62: 29}
BANDS = ["all", "vision", "llm_early", "llm_late", "expert", "head"]

def labels():
    pat = [(r"e2[67]_probe_t(\d+)_p(\d+)$", 1, 2), (r"e44_p_f(\d+)_(\d+)$", 1, 2), (r"e89(?:d\d)?_p_f(\d+)_(\d+)$", 1, 2),
           (r"e75_t(\d+)_d\d+_p(\d+)$", 1, 2), (r"e93_t(\d+)_p(\d+)$", 1, 2)]
    acc = {}
    for f in glob.glob(str(REPO / "results/merge/*.jsonl")):
        nm = pathlib.Path(f).stem
        for rx, it, ip in pat:
            m = re.match(rx, nm)
            if not m: continue
            t, d = int(m.group(it)), int(m.group(ip))
            if t >= N or d >= N or d == t or DUP.get(t) == d: break
            for ln in open(f):
                r = json.loads(ln)
                if r.get("status") == "ok":
                    n = r.get("n_trials", 10); s, w = acc.get((t, d), (0.0, 0))
                    acc[(t, d)] = (s + r["success_rate"] * n, w + n); break
            break
    return {k: s / w for k, (s, w) in acc.items()}

def main():
    z = np.load(REPO / "results/analysis/la_pair_features.npz")
    lab = labels()
    by_t = {}
    for (t, d), v in lab.items(): by_t.setdefault(t, {})[d] = v
    T = sorted(t for t, v in by_t.items() if len(v) >= 5 and t not in (18, 62))
    print(f"labels: {len(lab)} (target, donor) pairs over {len(by_t)} targets; {len(T)} targets with >=5 donors "
          f"(median {int(np.median([len(by_t[t]) for t in T]))} donors each)")
    # features [6 bands incl. 'all', N, N]
    def agg(k): x = z[k]; return np.concatenate([x.sum(0, keepdims=True), x], 0)
    ip, aip, rn2, dn2 = agg("ip"), agg("absip"), agg("rn2"), agg("dn2")
    feats = {}
    for b in range(6):
        rt, rd = rn2[b][:, None], rn2[b][None, :]
        feats[f"cos_r|{BANDS[b]}"] = ip[b] / np.sqrt(rt * rd)
        feats[f"sign_agree|{BANDS[b]}"] = ip[b] / np.maximum(aip[b], 1e-30)
        feats[f"cos_D|{BANDS[b]}"] = agg("ipD")[b] / np.sqrt(dn2[b][:, None] * dn2[b][None, :])
        nm = np.maximum(agg("OL")[b].diagonal()[:, None], 1e-30)      # normalise overlaps by self-overlap (module count)
        feats[f"overlap_out|{BANDS[b]}"] = agg("OL")[b] / nm
        feats[f"overlap_in|{BANDS[b]}"] = agg("OR")[b] / nm
        feats[f"donor_in_target_2s|{BANDS[b]}"] = agg("DinT2")[b] / rd
        feats[f"donor_in_target_out|{BANDS[b]}"] = agg("DinTL")[b] / rd
        feats[f"donor_in_target_in|{BANDS[b]}"] = agg("DinTR")[b] / rd
        feats[f"target_in_donor_2s|{BANDS[b]}"] = agg("DinT2")[b].T / rt
        feats[f"donor_resid_norm|{BANDS[b]}"] = np.broadcast_to(np.sqrt(rd), (N, N))
    try:
        C = np.load(REPO / "results/analysis/linalg_structure.npz")["coreCG"]; dc = np.sqrt(np.diag(C))
        feats["centred_core_cos|all"] = C / np.outer(dc, dc)
    except Exception:
        pass
    run_protocol(feats, by_t, T, "EXP-91: linear-algebra donor scores vs behavioural probe labels", "exp91_la_pair_eval.txt")

def run_protocol(feats, by_t, T, title, outname):
    def within(F, ts, sign=1):
        rhos = []
        for t in ts:
            ds = sorted(by_t[t]); y = np.array([by_t[t][d] for d in ds]); x = sign * F[t, ds]
            if y.std() == 0 or np.std(x) == 0: continue
            rhos.append(stats.spearmanr(x, y).statistic)
        return np.array(rhos)
    def pick_value(F, ts, sign=1, k=3):
        g, o = [], []
        for t in ts:
            ds = sorted(by_t[t]); y = np.array([by_t[t][d] for d in ds]); x = sign * F[t, ds]
            top = np.argsort(-x)[:k]; g.append(100 * (y[top].mean() - y.mean())); o.append(100 * (np.sort(y)[::-1][:k].mean() - y.mean()))
        return np.mean(g), np.mean(o)
    L = []; P = lambda *a: L.append(" ".join(str(x) for x in a))
    P(f"=== {title} ({len(T)} targets, within-target) ===")
    P("reference: probe vs an independent probe run, rho = 0.41 (the ceiling for any predictor)")
    res = []
    for nm, F in feats.items():
        r = within(F, T); res.append((nm, r.mean(), r.std(ddof=1) / np.sqrt(len(r)), len(r)))
    res.sort(key=lambda z: -abs(z[1]))
    P(f"\nall {len(res)} scores on all targets, sorted by |mean rho| (NOT multiple-comparison corrected):")
    for nm, m, se, n in res[:15]:
        P(f"  {nm:<34} rho {m:+.3f} (SE {se:.3f}, n={n})")
    P(f"  ... median |rho| over all scores {np.median([abs(r[1]) for r in res]):.3f}")
    rng = np.random.default_rng(91); perm = rng.permutation(T); A, B = sorted(perm[:len(T)//2]), sorted(perm[len(T)//2:])
    P("\npre-registered split (choose score AND sign on one half, test on the other):")
    for nmA, (tr, te) in (("A->B", (A, B)), ("B->A", (B, A))):
        best = max(((nm, s, within(F, tr, s).mean()) for nm, F in feats.items() for s in (1, -1)), key=lambda z: z[2])
        r = within(feats[best[0]], te, best[1]); gv, ov = pick_value(feats[best[0]], te, best[1])
        P(f"  {nmA}: chose {best[0]} sign {best[1]:+d} (train rho {best[2]:+.3f}) -> TEST rho {r.mean():+.3f} (SE {r.std(ddof=1)/np.sqrt(len(r)):.3f}, "
          f"p={stats.ttest_1samp(r,0).pvalue:.2f}); top-3 pick value {gv:+.2f}pp over random (oracle {ov:+.2f}pp)")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/{outname}").write_text(txt + "\n")

if __name__ == "__main__":
    main()
