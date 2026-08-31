#!/usr/bin/env python3
"""PHASE 0 — how RELIABLE is a 10-rollout probe ranking? (label-noise floor)

NOTE ON WHAT THIS MEASURES. The intent was to ask whether affinity is task-level or weight-level, using
the 50 LoRA checkpoints per task in dnd_libero90_dataset. Those checkpoints turn out to be functionally
IDENTICAL -- steps 9500-9990 sit in the converged tail, max elementwise difference 1.4e-5, cosine
1.0000. So they cannot answer the weight-sensitivity question.

What they DO give is something more fundamental: four near-identical adapters probed independently is a
TEST-RETEST experiment. It measures how much of a donor ranking is reproducible signal and how much is
rollout noise. This matters more than the original question, because a 10-rollout probe on a single
scene has a binomial SD of ~16pp at p=0.5, while the affinity differences we are asking the network to
rank are 10-30pp. If rankings do not reproduce across repeats of the same adapter, then our 604 training
labels are largely noise -- which would explain the hypernetwork failure at a deeper level than "not
enough tasks", and would also mean part of the "oracle gap" we are chasing is unachievable by anyone.
Consistent with this, the measured winner's curse (+5.0pp) already implies substantial per-probe noise.

ORIGINAL FRAMING (not answerable with this dataset):

This is a go/no-go for the entire learned-selector programme, and it is cheap.

The hypernetwork's job is to predict, for an unseen target, which donors help it. That is only a
well-posed learning problem if affinity is a stable property of the TASK. If instead donor rankings
swing between two checkpoints of the SAME task taken 10 SGD steps apart, then affinity is a property
of the exact weights -- effectively noise from the model's point of view -- and NO amount of training
data could let a network predict it. That would close the direction definitively, rather than leaving
it at "needs ~204 tasks".

`/media/pinas/mehdi/dnd_libero90_dataset` gives us 50 LoRA checkpoints per task (steps 9500-9990),
all 458 stems matching our decomposition exactly. We take 4 spread across that range for each of 3
weak tasks (weak = has headroom, so affinity is measurable at all), and probe the SAME 8 donors that
e27 used for that task.

Per (task, checkpoint) we measure:
  * the checkpoint's own SOLO baseline (checkpoints differ in quality; affinity must be measured
    against the right reference)
  * 8 donor probes with the standard operator W = S_16 + 1.0*r_ckpt + 0.5*r_donor

READ: Spearman correlation between donor rankings across checkpoint pairs of the same task.
  ~ +1  -> affinity is task-level; the learning problem is well-posed and data volume is the issue.
  ~  0  -> affinity is weight-level; the hypernetwork direction is dead regardless of data.
"""
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "scripts"))
import safetensors.torch as st
import apply_lora as al

DND = pathlib.Path("/media/pinas/mehdi/dnd_libero90_dataset")
OUTD = REPO / "results/analysis/decomp_adapters"
K = 16
ALPHA = 0.5
ROLLOUTS = 10
TASKS = [6, 19, 50]                      # weak => affinity is measurable
CKPTS = [9500, 9660, 9820, 9990]         # spread across the available range
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
# tmpfs build cache is 32G and each merged policy is 6.8G -> 4 workers overflow it. See EXPERIMENTS.md.
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]
RANK = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                  .read().split("probe rankings: ")[1].strip())
DONORS = {t: [p for p, _ in RANK[str(t)]] for t in TASKS}
_cache = {}


def fold_from(path):
    key = str(path)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 12:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(path))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    _cache[key] = out
    return out


def ckpt_path(t, step):
    return DND / f"libero90_task{t}" / f"{step}.safetensors"


def build(name, tgt_fold, donor_fold):
    """W = S_16 + 1.0*r_target + ALPHA*r_donor  (donor_fold None => solo baseline of the target)."""
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        terms = [(tgt_fold, 1.0)] + ([(donor_fold, ALPHA)] if donor_fold is not None else [])
        for fold, c in terms:
            b, a = fold[stem]
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


def gpus_free():
    return subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                          capture_output=True).returncode != 0


def run_batch(jobs):
    if not jobs:
        return
    planp = REPO / "sweep/_ckpt_current.json"
    json.dump(dict(META, n_trials=ROLLOUTS, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    print("[stage] waiting for GPUs", flush=True)
    while not gpus_free():
        time.sleep(120)
    for t in TASKS:
        for step in CKPTS:
            cp = ckpt_path(t, step)
            if not cp.exists():
                print(f"[skip] missing {cp}", flush=True)
                continue
            tf = fold_from(cp)
            # unit of work: 1 solo baseline + 8 donor probes for this checkpoint
            units = [("solo", None)] + [(f"p{p}", p) for p in DONORS[t]]
            todo = [(tag, p) for tag, p in units
                    if rate(f"e30_t{t}_s{step}_{tag}", t) is None]
            for i in range(0, len(todo), 3):
                jobs, built = [], []
                for tag, p in todo[i:i + 3]:
                    df = fold_from(afile[p]) if p is not None else None
                    path = build(f"ck_t{t}_s{step}_{tag}", tf, df)
                    built.append(path)
                    jobs.append({"name": f"e30_t{t}_s{step}_{tag}", "method": "average", "params": {},
                                 "adapters": [], "adapter_paths": [path], "eval_scenes": scenes[t]})
                print(f"[run] t{t} step{step}: {[j['name'] for j in jobs]}", flush=True)
                run_batch(jobs)
                for p_ in built:
                    pathlib.Path(p_).unlink(missing_ok=True)
            print(f"[done] t{t} step{step}", flush=True)
        report()          # incremental, so partial results are usable
    report()


def report():
    from scipy import stats
    L = ["=== PHASE 0: is affinity TASK-level or WEIGHT-level? ===",
         f"{len(TASKS)} weak tasks x {len(CKPTS)} LoRA checkpoints (steps {CKPTS[0]}-{CKPTS[-1]}) "
         f"x 8 donors, {ROLLOUTS} rollouts", "",
         "affinity(ckpt, donor) = probe success - that CHECKPOINT's own solo baseline", ""]
    allr = []
    for t in TASKS:
        aff, base, have = {}, {}, []
        for step in CKPTS:
            b = rate(f"e30_t{t}_s{step}_solo", t)
            if b is None:
                continue
            row = []
            for p in DONORS[t]:
                v = rate(f"e30_t{t}_s{step}_p{p}", t)
                row.append(None if v is None else (v - b) * 100)
            if all(x is not None for x in row):
                aff[step] = np.array(row, dtype=float)
                base[step] = b * 100
                have.append(step)
        if len(have) < 2:
            L.append(f"t{t}: {len(have)} complete checkpoint(s) — need >=2")
            continue
        L.append(f"t{t}  (donors {DONORS[t]})")
        for step in have:
            L.append(f"   step {step}: solo {base[step]:5.1f}%   affinities "
                     + " ".join(f"{x:+5.0f}" for x in aff[step]))
        rs = []
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                r = stats.spearmanr(aff[have[i]], aff[have[j]]).statistic
                if not np.isnan(r):
                    rs.append(r)
        if rs:
            allr += rs
            L.append(f"   pairwise rank correlation across checkpoints: mean rho = {np.mean(rs):+.3f}  "
                     f"(n={len(rs)} pairs, range {min(rs):+.2f}..{max(rs):+.2f})")
        L.append(f"   solo baseline spread across checkpoints: "
                 f"{max(base.values())-min(base.values()):.1f}pp")
        L.append("")
    if allr:
        L += [f"OVERALL mean rank correlation across checkpoints of the same task: {np.mean(allr):+.3f} "
              f"(n={len(allr)})", "",
              "  rho ~ +1  => probe rankings are reproducible; labels are sound, data volume is the blocker.",
              "  rho ~  0  => rankings do not reproduce; the labels are largely noise and the",
              "               achievable oracle is far below the measured one."]
    open(REPO / "results/analysis/ckpt_stability_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
