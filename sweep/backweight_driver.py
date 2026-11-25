#!/usr/bin/env python3
"""EXP-35 — validate BACK-WEIGHTED dosing at fleet scale (all 76 targets).

The operator has always used ONE coefficient for all 458 modules:
    W = S_16 + r_target + c * r_donor,     c = 0.5 everywhere.
EXP-33 (5 targets, energy-matched fixed patterns) suggested that WHERE the donor's residual lands
matters more than how much of it arrives:
    back  (llm-late / expert / action)  +21.0pp      uniform +11.0pp      vision-only +1.0pp
Every pattern injected the same Frobenius norm, so that ordering is about placement, not dose.

This is the deployment-grade test of that, on ALL 76 fleet targets, as a strictly PAIRED comparison:
  * same targets
  * same donors -- each target's e27 top-3, already selected and already committed under uniform dosing
  * same commit budget (25 rollouts)
  * same total injected energy, rescaled per target so only the DISTRIBUTION across bands differs
The uniform arm therefore costs nothing: it is the existing e27 commit. Only the back arm is run.

If back-weighting holds up here it is a free upgrade to the operator -- no learning, no extra rollouts
at deployment, just a better default. That would be the first change to the operator since alpha=0.5
was verified, and unlike the hypernetwork it requires nothing we do not already have.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
import apply_lora as al
import hostcfg
from reinforce_train import group_of, fold, stems, USIG, VS, scenes, target_baseline

K = 16
ALPHA = 0.5
N_PICK = 3
COMMIT_R = 25
BACK = np.array([0.5, 0.5, 2.0, 2.0, 2.0])      # relative weights; rescaled to match uniform energy
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META = hostcfg.apply(META)
CHUNK = max(6, 2 * META["n_workers"])
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
TARGETS = [t for t in range(76) if str(t) in RANK]



def _clear_orphaned_builds():
    """Adapters in decomp_adapters/ are build->eval->delete transients. If no orchestrator is running,
    any that exist were abandoned by a killed run -- and 30+ of them is a blown disk quota (happened
    twice). Clear them at startup rather than trusting every kill path to clean up."""
    import subprocess as _sp
    if _sp.run(["pgrep", "-f", "merge_orchestrat[o]r.py"], capture_output=True).returncode != 0:
        for _f in (REPO / "results/analysis/decomp_adapters").glob("*.safetensors"):
            _f.unlink(missing_ok=True)


# NOTE: not called at import time. A chunked driver builds its adapters BEFORE its orchestrator
# starts, so an import from an analysis script in that window deleted two live probe files
# (EXP-32, 09-04 21:48). Drivers call it from main() only.

def band_energy(donor):
    f = fold(donor)
    e = np.zeros(5)
    for stem in stems:
        if stem not in f:
            continue
        b, a = f[stem]
        D = b @ a
        Un = USIG[stem] / torch.linalg.norm(USIG[stem], dim=0).clamp(min=1e-12)
        r = D - Un @ (Un.T @ D @ VS[stem]) @ VS[stem].T
        e[group_of(stem)] += float((r ** 2).sum())
    return e


def matched_coefs(donors):
    """Per-donor 5-vectors: the BACK shape, rescaled so total injected energy equals uniform ALPHA/3."""
    out = {}
    uni = ALPHA / N_PICK
    for d in donors:
        E = band_energy(d)
        target = (uni ** 2) * E.sum()
        got = float((BACK ** 2 * E).sum())
        out[d] = BACK * float(np.sqrt(target / got)) if got > 0 else BACK * 0.0
    return out


def build(name, target, coefs):
    ft = fold(target)
    folds = {d: fold(d) for d in coefs}
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        g = group_of(stem)
        Bb, Ab = [Usig], [V.T]
        if stem in ft:
            b, a = ft[stem]
            M = U.T @ b; N = a @ V
            Bb += [b, -(U @ M)]; Ab += [a, N @ V.T]
        for d, cv in coefs.items():
            c = float(cv[g])
            if stem not in folds[d] or c == 0.0:
                continue
            b, a = folds[d][stem]
            M = U.T @ b; N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def get(job, t):
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
    _clear_orphaned_builds()
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(90)
    todo = [t for t in TARGETS if get(f"e46_back_t{t}", t) is None]
    print(f"[back] {len(todo)} targets to run (uniform arm reuses the e27 commits)", flush=True)
    for i in range(0, len(todo), CHUNK):
        jobs, built = [], []
        for t in todo[i:i + CHUNK]:
            picks = [p for p, _ in RANK[str(t)][:N_PICK]]
            path = build(f"bw_t{t}", t, matched_coefs(picks))
            built.append(path)
            jobs.append({"name": f"e46_back_t{t}", "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [path], "eval_scenes": scenes[t]})
        planp = REPO / "sweep/_bw_current.json"
        json.dump(dict(META, n_trials=COMMIT_R, jobs=jobs), open(planp, "w"), indent=1)
        print(f"[run] chunk {i//CHUNK+1}/{(len(todo)+CHUNK-1)//CHUNK}", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
        report(final=False)
    report(final=True)


def report(final):
    from scipy import stats
    rows = []
    for t in TARGETS:
        u = get(f"e27_commit_t{t}_top3", t)
        b = get(f"e46_back_t{t}", t)
        if u is None or b is None:
            continue
        base = target_baseline(t)
        rows.append((t, base * 100, (u - base) * 100, (b - base) * 100))
    L = ["=== EXP-35: BACK-WEIGHTED vs UNIFORM dosing, all fleet targets ===",
         "paired: same targets, same e27 top-3 donors, same 25-rollout commit, same injected energy;",
         "only the DISTRIBUTION across layer bands differs "
         "(vision/llm-early low, llm-late/expert/action high)", ""]
    if rows:
        d = np.array([r[3] - r[2] for r in rows])
        bu = np.array([r[2] for r in rows]); bb = np.array([r[3] for r in rows])
        base = np.array([r[1] for r in rows]); weak = base < 90
        t_ = stats.ttest_1samp(d, 0)
        L += [f"  n = {len(rows)} targets" + ("" if final else "  (partial)"),
              f"  uniform dosing      : {bu.mean():+.2f}pp vs baseline",
              f"  BACK-WEIGHTED dosing: {bb.mean():+.2f}pp vs baseline",
              f"  paired difference   : {d.mean():+.2f}pp  (SE {stats.sem(d):.2f}, t={t_.statistic:+.2f}, "
              f"p={t_.pvalue:.4f})",
              f"  back better on {int((d>0).sum())}/{len(d)} targets"]
        for nm, m in (("WEAK  (base<90%)", weak), ("STRONG(base>=90%)", ~weak)):
            if m.sum():
                L.append(f"  {nm} n={int(m.sum()):<3}: uniform {bu[m].mean():+.2f}  "
                         f"back {bb[m].mean():+.2f}  diff {d[m].mean():+.2f}pp")
        L += ["", "  (EXP-33 pilot, 4 targets: uniform +11.0, back +21.0, diff +10.0pp p=0.269)"]
    open(REPO / "results/analysis/backweight_report.txt", "w").write("\n".join(L) + "\n")
    if final:
        print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
