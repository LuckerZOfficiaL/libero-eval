#!/usr/bin/env python3
"""EXP-30 — train LoRA2Coef with REINFORCE, reward = rollout performance delta.

WHY NOT BACKPROP. The plan asked to "compute the gradient on the rollouts and backpropagate". No
gradient exists: a rollout is a discrete success/failure produced by a non-differentiable physics
simulator, so d(success)/d(coefficient) is undefined. What IS true is that the merge is LINEAR in the
coefficients, W(c) = D_target + sum_g c_g * r_donor[g], so a score-function estimator gives us an
unbiased gradient for the price of extra rollouts:

    c ~ N(mu_theta, sigma)          mu_theta = network output (sigmoid, starts at 0.5)
    R = success(merged) - success(target)                      <- the performance delta
    grad_theta J = E[ (R - b) * grad_theta log p(c | theta) ]   <- b = running baseline

VARIANCE CONTROL, because rollouts are the budget:
  * ANTITHETIC PAIRS. Every perturbation eps is evaluated as mu+eps AND mu-eps. The two share a
    baseline and their difference cancels the reward's common component, roughly halving the samples
    needed for the same gradient precision.
  * RUNNING BASELINE b, per target: subtracting a target's own mean reward removes the large
    target-to-target offset that dominates raw reward (target identity explained R^2=0.41 of affinity
    in the earlier study, donor identity ~0).
  * WARM START at c=0.5. The head is zero-initialised so the network begins at the dose already
    verified optimal (+9.78pp on weak targets). Random initial coefficients would make every early
    merge bad, every reward equally poor, and leave REINFORCE with no contrast to learn from.

REWARD, not 1/(difference). The plan's 1/(performance difference) is singular as the difference -> 0
and inverts sign when the merge is worse. The delta itself is bounded, signed, and directly the thing
we want to maximise.

BUDGET. Each REINFORCE step costs POP antithetic pairs x 2 x ROLLOUTS rollouts, plus one baseline
evaluation per target (cached). Defaults: POP=4, ROLLOUTS=10 -> 80 rollouts per step per target.
"""
import argparse
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
from lora2coef import LoRA2Coef

K = 16
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
META = hostcfg.apply(META)
base_sr = {}
for ln in open(REPO / "results/baseline_results.jsonl"):
    r = json.loads(ln)
    if r.get("status") == "ok":
        base_sr[r["scene"]] = r["success_rate"]

# layer groups for v2: coarse bands, not 458 modules. REINFORCE sees ONE scalar reward per rollout
# batch, so a 458-dim action space is hopelessly under-determined at any affordable budget.

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

def group_of(stem):
    if "vision_tower" in stem or "siglip" in stem:
        return 0
    if "gemma_expert" in stem:
        return 3
    if "action" in stem:
        return 4
    if "language_model.layers." in stem:
        n = int(stem.split("language_model.layers.")[1].split(".")[0])
        return 1 if n < 9 else 2
    return 2


GROUPS = 5
_cache = {}


def fold(t):
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


def build(name, target, donor, coefs):
    """W = S_16 + r_target + c_g * r_donor, with c_g applied per layer group."""
    ft, fd = fold(target), fold(donor)
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        c = float(coefs[group_of(stem)] if len(coefs) > 1 else coefs[0])
        Bb, Ab = [Usig], [V.T]
        for f, k in ((ft, 1.0), (fd, c)):
            if stem not in f:
                continue
            b, a = f[stem]
            M = U.T @ b
            N = a @ V
            Bb += [k * b, -k * (U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def read(job, t):
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


# Each merged adapter is ~0.9GB on disk, and a REINFORCE step needs batch*pop*2 of them (64 at the
# defaults). Building them all up front is ~59GB and blew shiva's 45GB quota. Build in CHUNKS instead:
# disk stays bounded at CHUNK*0.9GB while the queue stays deep enough to keep every worker busy.
CHUNK = max(8, 2 * META["n_workers"])


def evaluate(specs, rollouts):
    """specs: [(job, target, donor, coefs)] -> {job: success_rate}, evaluated in disk-bounded chunks."""
    todo = [s for s in specs if read(s[0], s[1]) is None]
    for i in range(0, len(todo), CHUNK):
        jobs, built = [], []
        for job, t, d, c in todo[i:i + CHUNK]:
            p = build(job, t, d, c)
            built.append(p)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": scenes[t]})
        planp = REPO / "sweep/_rl_current.json"
        json.dump(dict(META, n_trials=rollouts, jobs=jobs), open(planp, "w"), indent=1)
        print(f"  [eval] chunk {i//CHUNK + 1}/{(len(todo)+CHUNK-1)//CHUNK} "
              f"({len(jobs)} merges, ~{len(jobs)*0.9:.0f}GB)", flush=True)
        subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                        str(planp)], cwd=REPO)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)
    return {job: read(job, t) for job, t, _, _ in specs}


def target_baseline(t):
    ss = [s for s in scenes[t] if s in base_sr]
    return sum(base_sr[s] for s in ss) / len(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=1, help="1 = scalar alpha (v1); 5 = layer bands (v2)")
    ap.add_argument("--batch", type=int, default=8,
                    help="training targets SAMPLED per step (from the 66-target train pool)")
    ap.add_argument("--holdout_seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--pop", type=int, default=4, help="antithetic PAIRS per step")
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--prior", type=float, default=0.05,
                    help="weak pull toward c=0.5, the dose already verified optimal; keeps the policy "
                         "from wandering to a boundary before it has learned to discriminate")
    ap.add_argument("--tag", default="v1")
    a = ap.parse_args()

    E27 = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                     .read().split("probe rankings: ")[1].strip())
    tok = torch.load(REPO / "results/analysis/lora_tokens_raw.pt")
    X, tasks = tok["X"], list(tok["tasks"])
    idx = {t: i for i, t in enumerate(tasks)}
    # HELD-OUT PROTOCOL, matching the earlier hypernetwork study: 10 targets never trained on
    # (5 weak, 5 strong, stratified), the other 66 form the training pool. Each step SAMPLES a
    # minibatch from that pool -- using all 66 every step would cost 5,280 rollouts per step.
    hrng = np.random.default_rng(a.holdout_seed)
    weak = [t for t in range(76) if target_baseline(t) < 0.90]
    strong = [t for t in range(76) if target_baseline(t) >= 0.90]
    holdout = sorted(hrng.choice(weak, 5, replace=False).tolist()
                     + hrng.choice(strong, 5, replace=False).tolist())
    train_pool = [t for t in range(76) if t not in holdout]
    json.dump({"holdout": holdout, "train_pool": train_pool},
              open(REPO / f"results/analysis/rl_{a.tag}_split.json", "w"), indent=1)
    rng = np.random.default_rng(7)

    net = LoRA2Coef(n_mod=X.shape[1], in_h=X.shape[2], in_w=X.shape[3], n_groups=a.groups)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    baseline = {t: None for t in train_pool}   # per-target running baseline, whole pool
    hist = []
    print(f"[rl] groups={a.groups} | train pool {len(train_pool)} targets, batch {a.batch}/step | "
          f"holdout {holdout}", flush=True)
    print(f"[rl] steps={a.steps} pop={a.pop} rollouts={a.rollouts} -> "
          f"{a.steps*a.batch*a.pop*2*a.rollouts} training rollouts", flush=True)

    for step in range(a.steps):
        specs, meta = [], []
        targets = rng.choice(train_pool, min(a.batch, len(train_pool)), replace=False).tolist()
        for t in targets:
            # STRATIFIED donor sampling. Drawing donors uniformly at random is what broke the first
            # attempt: a random donor is worth ~0 (the selection-free screen measured -0.18pp), so the
            # reward landscape has a CONSTANT optimum -- "use less donor" -- and the network duly
            # collapsed to c=0.385 for every pair (spread across 30 donors: 0.0005). To learn a
            # donor-DEPENDENT dose it must see pairs where a HIGH coefficient is rewarded, so half the
            # batch now draws from the target's known-good donors (its e27 top-3) and half stays random.
            if str(t) in E27 and rng.random() < 0.5:
                d = int(rng.choice([p for p, _ in E27[str(t)][:3]]))
            else:
                d = int(rng.choice([c for c in tasks if c != t]))
            mu = net(X[idx[t]].unsqueeze(0), X[idx[d]].unsqueeze(0))[0]
            for k in range(a.pop):
                eps = torch.randn(a.groups) * a.sigma
                for sign, tagp in ((1, "p"), (-1, "m")):
                    c = (mu.detach() + sign * eps).clamp(0.02, 1.5)
                    job = f"e42_{a.tag}_s{step}_t{t}_d{d}_k{k}{tagp}"
                    specs.append((job, t, d, c.tolist()))
                    meta.append((t, d, mu, eps, sign, job))
        res = evaluate(specs, a.rollouts)
        loss = torch.zeros(())
        rews = []
        # collect advantages first so they can be standardised: raw rewards are dominated by which
        # targets happened to be drawn, and an unnormalised advantage made the first run take huge
        # steps (0.498 -> 0.312 -> 0.000 in two updates).
        raw = []
        for (t, d, mu, eps, sign, job) in meta:
            sr = res.get(job)
            if sr is not None:
                raw.append((sr - target_baseline(t)) * 100.0)
        astd = float(np.std(raw)) if len(raw) > 1 else 1.0
        astd = max(astd, 1e-3)
        for (t, d, mu, eps, sign, job) in meta:
            sr = res.get(job)
            if sr is None:
                continue
            R = (sr - target_baseline(t)) * 100.0        # performance delta in pp
            b = baseline[t]
            baseline[t] = R if b is None else 0.7 * b + 0.3 * R
            adv = (R - (b if b is not None else R)) / astd        # standardised
            rews.append(R)
            # score-function gradient of a Gaussian policy: (c - mu)/sigma^2 = sign*eps/sigma^2
            logp = -((sign * eps) ** 2).sum() / (2 * a.sigma ** 2) + (mu * (sign * eps)).sum() / a.sigma ** 2
            loss = loss - adv * logp
        if rews:
            loss = loss / len(rews)
            # weak prior toward the verified dose, applied to the CURRENT policy output
            if a.prior > 0:
                mus = torch.stack([net(X[idx[t]].unsqueeze(0), X[idx[d]].unsqueeze(0))[0]
                                   for t, d, *_ in {(m[0], m[1]): m for m in meta}.values()])
                loss = loss + a.prior * ((mus - 0.5) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                # probe the coefficient on HELD-OUT targets, so the trace reflects generalisation
                probe = [float(net(X[idx[t]].unsqueeze(0),
                                   X[idx[int(rng.choice([c for c in train_pool if c != t]))]].unsqueeze(0)).mean())
                         for t in holdout]
            with torch.no_grad():
                t0 = holdout[0]
                spread_c = [float(net(X[idx[t0]].unsqueeze(0), X[idx[d]].unsqueeze(0)))
                            for d in train_pool[:30]]
            spread = float(np.max(spread_c) - np.min(spread_c))
            hist.append({"step": step, "mean_reward": float(np.mean(rews)),
                         "coef_mean": float(np.mean(probe)), "coef_spread": spread})
            print(f"[rl] step {step}: mean reward {np.mean(rews):+.1f}pp  "
                  f"coef -> {np.mean(probe):.3f}  spread {spread:.4f}  (n={len(rews)})", flush=True)
            json.dump(hist, open(REPO / f"results/analysis/rl_{a.tag}_history.json", "w"), indent=1)
            torch.save(net.state_dict(), REPO / f"results/analysis/rl_{a.tag}_net.pt")
    report(a, hist)


def report(a, hist):
    L = [f"=== EXP-30: REINFORCE training of LoRA2Coef ({a.tag}, groups={a.groups}) ===",
         f"reward = rollout success(merged) - success(target); antithetic pairs, running baseline;",
         f"warm start at c=0.5 (the verified optimum), so the network learns DEVIATIONS from it", "",
         f"{'step':<6}{'mean reward (pp)':>18}{'mean coefficient':>18}"]
    for h in hist:
        L.append(f"{h['step']:<6}{h['mean_reward']:>+18.1f}{h['coef_mean']:>18.3f}")
    if len(hist) >= 2:
        d = hist[-1]["mean_reward"] - hist[0]["mean_reward"]
        L += ["", f"  reward change over training: {d:+.1f}pp",
              f"  coefficient moved {hist[0]['coef_mean']:.3f} -> {hist[-1]['coef_mean']:.3f} "
              f"(0.5 = the fixed-alpha baseline this must beat)"]
    open(REPO / f"results/analysis/rl_{a.tag}_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
