#!/usr/bin/env python3
"""EXP-23.3 — DEPLOY the hypernetwork: real rollouts on its picks, zero selection cost.

The v2 network transfers in label space (~+1pp over random, p<0.01, on held-out targets). The claim
that matters is deployment-shaped: for a NEW target, rank ALL eligible pool donors (not just the 8 that
happen to have labels), merge the top-3, and measure at 25 rollouts. Selection cost: ZERO rollouts.

Protocol per weak target T (the 15 with baseline < 90%):
  * train an ensemble of 5 nets (binary objective, WLS features) on ALL consolidated labels EXCLUDING
    every pair whose target is T — leave-one-target-out, so T is genuinely unseen;
  * average the ensemble's scores over all 75 candidates (minus T, minus its md5 twin);
  * commit W = S_16 + 1.0*r_T + (0.5/3)*sum(top-3 residuals) at 25 rollouts  (identical operator and
    budget to the e27 probe commits, which are the paired comparison).

Read: hypernet picks vs (a) baseline, (b) the selection-free random-5 screen (e23), and (c) the
behavioural probe's top-3 (e27) — the free selector against the 80-rollout one.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "sweep"))
sys.path.insert(0, str(REPO / "scripts"))
import hypernet as H
import safetensors.torch as st
import apply_lora as al

K = 16
ALPHA = 0.5
N_PICK = 3
ROLLOUTS = 25
ENSEMBLE = 5
TWIN = {7: 18, 18: 7, 29: 62, 62: 29}
OUTD = REPO / "results/analysis/decomp_adapters"
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
USIG = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]

# WLS feature view (activation block excluded: no signal — see EXP-23.2 side finding)
fz = np.load(REPO / "results/analysis/hypernet_feats.npz")
sl = json.loads(str(fz["slices"]))
keep = np.concatenate([np.arange(*sl[b]) for b in "WLS"])
F = torch.tensor(fz["F"][:, keep], dtype=torch.float32)
_, groups, baseline = H.load(str(REPO / "results/analysis/pair_labels.json"))
WEAK = sorted([t for t in range(76) if baseline[t] < 0.90], key=lambda t: baseline[t])


def picks_for(T):
    """Leave-one-target-out ensemble; returns the top-N_PICK donors among ALL eligible candidates."""
    tr = [g for g in groups if g[0] != T]
    elig = [c for c in range(76) if c != T and TWIN.get(T) != c]
    scores = np.zeros(len(elig))
    for e in range(ENSEMBLE):
        torch.manual_seed(1000 * T + e)
        np.random.seed(1000 * T + e)
        net = H.AffinityNet(F.shape[1], d_emb=32, p_drop=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        for ep in range(150):
            net.train()
            for i in np.random.permutation(len(tr)):
                t_, cand, aff, lw = tr[i]
                s = net(F[t_].expand(len(cand), -1), F[cand])
                loss = H.binary_loss(s, aff, lw)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        with torch.no_grad():
            scores += net(F[T].expand(len(elig), -1), F[elig]).numpy()
    order = np.argsort(-scores)
    return [elig[i] for i in order[:N_PICK]], {elig[i]: float(scores[i] / ENSEMBLE) for i in order}


_cache = {}
def load_fold(t):
    if t in _cache:
        return _cache[t]
    if len(_cache) > 16:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(afile[t])).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return _cache.setdefault(t, out)


def build(name, target, partners):
    tensors = {}
    folds = {t: load_fold(t) for t in [target] + partners}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        for t, c in [(target, 1.0)] + [(p, ALPHA / len(partners)) for p in partners]:
            b, a = folds[t][stem]
            M = U.T @ b
            N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def rate(job, t):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return None
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok":
            sr[d["scene"]] = d["success_rate"]
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss) / len(ss) if len(ss) == len(scenes[t]) else None


def main():
    picks = {}
    pf = REPO / "results/analysis/hypernet_picks.json"
    if pf.exists():
        picks = {int(k): v for k, v in json.load(open(pf)).items()}
    for T in WEAK:
        if T in picks:
            continue
        top, ranked = picks_for(T)
        picks[T] = top
        print(f"[pick] t{T}: {top}", flush=True)
        json.dump({str(k): v for k, v in picks.items()}, open(pf, "w"), indent=1)
    print("[stage] all picks computed; committing", flush=True)
    while not subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                             capture_output=True).returncode != 0:
        time.sleep(120)
    jobs, built = [], []
    for T in WEAK:
        if rate(f"e33_hn_t{T}", T) is not None:
            continue
        path = build(f"hn_t{T}", T, picks[T])
        built.append(path)
        jobs.append({"name": f"e33_hn_t{T}", "method": "average", "params": {}, "adapters": [],
                     "adapter_paths": [path], "eval_scenes": scenes[T]})
    if jobs:
        planp = REPO / "sweep/_hn_current.json"
        json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
    for p_ in built:
        pathlib.Path(p_).unlink(missing_ok=True)
    report(picks)


def report(picks):
    from scipy import stats
    R = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                   .read().split("probe rankings: ")[1].strip())
    L = ["=== EXP-23.3: HYPERNET-SELECTED merging, real rollouts, ZERO selection cost ===",
         "leave-one-target-out ensemble; top-3 of ALL 75 candidates; commit @25 (same op/budget as e27)", "",
         f"{'target':<8}{'base':>6}{'hypernet picks':>18}{'rand5':>8}{'probe3':>8}{'HN3':>8}"
         f"{'HN-base':>9}{'HN-probe':>10}"]
    dh, dp, dr = [], [], []
    for T in WEAK:
        b = baseline[T]
        hn = rate(f"e33_hn_t{T}", T)
        pr = rate(f"e27_commit_t{T}_top3", T)
        r5 = rate(f"e23_t{T}", T)
        row = f"t{T:<7}{b*100:>5.0f}%{str(picks.get(T)):>18}"
        row += f"{'' if r5 is None else f'{r5*100:.0f}%':>8}"
        row += f"{'' if pr is None else f'{pr*100:.0f}%':>8}"
        row += f"{'' if hn is None else f'{hn*100:.0f}%':>8}"
        if hn is not None:
            dh.append((hn - b) * 100)
            row += f"{(hn-b)*100:>+9.0f}"
            if pr is not None:
                dp.append((hn - pr) * 100)
                row += f"{(hn-pr)*100:>+10.0f}"
        if r5 is not None and hn is not None:
            dr.append((hn - r5) * 100)
        L.append(row)
    if dh:
        L += ["", f"  HYPERNET vs baseline : {np.mean(dh):+.2f}pp  (SE {stats.sem(dh):.2f}, n={len(dh)})",
              f"  HYPERNET vs random-5 : {np.mean(dr):+.2f}pp" if dr else "",
              f"  HYPERNET vs probe    : {np.mean(dp):+.2f}pp  (probe costs 80 rollouts/scene; hypernet costs 0)"]
    open(REPO / "results/analysis/hypernet_deploy_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
