#!/usr/bin/env python3
"""EXP-85 stage 2 -- behavioural-fingerprint PRIOR for donor selection.

Describe every fleet task by WHERE and HOW its own adapter fails on its 25 fixed states (e92_base_f{t}:
per-episode PROGRESS and SUCCESS_STEP from main_dense.py) and feed that as an extra feature block 'P' to
the W+A/48 affinity model that is our standing reference (+1.14pp fleet-holdout, EXP-42). Same labels
(pair_labels_ext.json), same folds (hypernet.split), same seeds -> paired seed-for-seed:
    WA   = reference    WAP = reference + fingerprint    P = fingerprint alone
Targets >= 76 (external / variants) have no per-episode baseline: mean-filled with a 'missing' flag.
"""
import argparse, json, pathlib, re, subprocess, sys, numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from scipy import stats
from reinforce_train import scenes as SC

SRC = REPO / "results/analysis/hypernet_featsexta48.npz"
OUT = REPO / "results/analysis/hypernet_feats_fp.npz"

def fingerprint(t):
    """One vector per fleet task from its e92 record(s); None if absent."""
    f = REPO / f"results/merge/e92_base_f{t}.jsonl"
    if not f.exists(): return None
    recs = [json.loads(l) for l in open(f) if l.strip()]
    recs = [r for r in recs if r.get("status") == "ok" and isinstance(r.get("progress"), list)]
    if not recs: return None
    P = np.concatenate([r["progress"] for r in recs]).astype(float)
    S = np.concatenate([r["success_step"] for r in recs]).astype(float)
    ok = S >= 0; fail = ~ok
    pf = P[fail] if fail.any() else np.array([1.0])          # progress reached by failed episodes
    ss = S[ok] / 300.0 if ok.any() else np.array([1.0])       # normalised time-to-success
    v = [ok.mean(), fail.mean(),
         pf.mean(), np.percentile(pf, 25), np.percentile(pf, 50), np.percentile(pf, 75),
         (pf < 0.5).mean(), (pf > 0.8).mean(),                 # never got going vs nearly finished
         ss.mean(), ss.std(), ss.min(), ss.max(),
         P.mean(), P.std(), P.min()]
    # distribution shape: sorted progress and sorted (fail -> 1.0) time-to-success, resampled to 25 points
    sp = np.sort(P); st = np.sort(np.where(ok, S / 300.0, 1.0))
    q = np.linspace(0, 1, 25)
    v += list(np.quantile(sp, q)) + list(np.quantile(st, q))
    return np.array(v, dtype=np.float32)

def build():
    z = np.load(SRC, allow_pickle=True)
    F = z["F"]; tasks = [int(x) for x in z["tasks"]]; sl = json.loads(str(z["slices"]))
    rows = {t: fingerprint(t) for t in range(76)}
    have = [t for t in rows if rows[t] is not None]
    d = len(rows[have[0]]); mean = np.mean([rows[t] for t in have], 0)
    Pm = np.zeros((len(tasks), d + 1), np.float32)
    for i, t in enumerate(tasks):
        if t < 76 and rows.get(t) is not None: Pm[i, :d] = rows[t]
        else: Pm[i, :d] = mean; Pm[i, d] = 1.0                  # missing flag
    mu, sd = Pm[:, :d].mean(0), Pm[:, :d].std(0) + 1e-6
    Pm[:, :d] = (Pm[:, :d] - mu) / sd
    sl["P"] = [F.shape[1], F.shape[1] + d + 1]
    np.savez(OUT, F=np.concatenate([F, Pm], 1).astype(np.float32), tasks=np.array(tasks), slices=json.dumps(sl))
    print(f"[e85] fingerprint block: {len(have)}/76 fleet tasks measured, {d}+1 dims; F -> {F.shape[1] + d + 1}", flush=True)
    return len(have)

def run_variant(blocks, mode, seeds, epochs, tag, lr):
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/hypernet.py"), "--blocks", blocks, "--holdout", mode,
           "--seeds", str(seeds), "--epochs", str(epochs), "--lr", str(lr), "--labels", "results/analysis/pair_labels_ext.json",
           "--feats", "results/analysis/hypernet_feats_fp.npz", "--baselines", "results/analysis/baselines_ext.json",
           "--tag", f"e85_{tag}_{blocks}_{mode}"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout
    # per-seed: hypernet top-3 aff, ridge top-3 aff, random
    hy = [float(x) for x in re.findall(r"^  hypernet\s+top-3 aff\s+([+-]\d+\.\d+)pp", out, re.M)]
    rd = [float(x) for x in re.findall(r"^  ridge\s+top-3 aff\s+([+-]\d+\.\d+)pp", out, re.M)]
    rn = [float(x) for x in re.findall(r"^  random\s+top-3 aff\s+([+-]\d+\.\d+)pp", out, re.M)]
    if not (len(hy) == len(rd) == len(rn) == seeds):
        print(out[-3000:]); raise SystemExit(f"parse failed for {blocks}/{mode}: {len(hy)} {len(rd)} {len(rn)}")
    return np.array(hy) - np.array(rn), np.array(rd) - np.array(rn)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=36); ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--tag", default="full")
    # hypernet.py's defaults (lr 3e-3, 300 ep) collapse the net to constant scores (score sd 0, spearman nan);
    # the +1.14 reference (EXP-42 / EXP-82 feat arm) used lr 1e-3, 150 epochs.
    ap.add_argument("--lr", type=float, default=1e-3); a = ap.parse_args()
    n = build()
    res = {}
    for mode in ("fleet", "new"):
        for blocks in ("WA", "WAP", "P", "WALSP"):
            res[(blocks, mode)] = run_variant(blocks, mode, a.seeds, a.epochs, a.tag, a.lr)
            h, r = res[(blocks, mode)]
            print(f"  {mode:5s} {blocks:6s} hypernet {h.mean():+.2f} ±{h.std()/np.sqrt(len(h)):.2f}   ridge {r.mean():+.2f}", flush=True)
    L = [f"=== EXP-85 fingerprint prior ({n}/76 fleet tasks measured; {a.seeds} seeds x {a.epochs} epochs; top-3 affinity gain over random, pp) ==="]
    for mode in ("fleet", "new"):
        ref = res[("WA", mode)][0]
        for blocks in ("WA", "WAP", "P", "WALSP"):
            h, r = res[(blocks, mode)]; d = h - ref
            p = stats.ttest_rel(h, ref).pvalue if blocks != "WA" else float("nan")
            L.append(f"{mode:5s} {blocks:6s} hypernet {h.mean():+.2f} ±{h.std()/np.sqrt(len(h)):.2f} | ridge {r.mean():+.2f} | paired vs WA {d.mean():+.2f} p={p:.3f}")
    out = "\n".join(L); print("\n" + out)
    (REPO / f"results/analysis/exp85_prior_{a.tag}.txt").write_text(out + "\n")

if __name__ == "__main__":
    main()
