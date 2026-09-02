#!/usr/bin/env python3
"""EXP-32 — THE UNIFIED SCREEN: one pool of 84 LoRAs, every eligible one a target in turn.

This subsumes the separate external-LoRA experiments into a single table measured under one budget:

    slice of this run                     reproduces
    fleet target,  fleet donors           the all-76 screen        (+2.22pp, p=0.001)
    fleet target,  external donors        EXP-26                   (-1.6pp)
    external target, fleet donors         EXP-27 / EXP-29          (+16.0pp, upper bound)
    any target,    combined candidates    EXP-28                   (+8.0pp, n=2)

POOL. 84 adapters: 76 fleet + 8 external suite-experts. All 84 can DONATE (a donor needs no evaluation
machinery). Only 81 can be TARGETS: spatial_expert, object_expert and 10_expert ship no action
normalisation, and evaluating an adapter against the wrong norm stats reads ~0% on nearly every task --
a broken measurement that would masquerade as a weak adapter. They stay in as donors.

TARGETS = (adapter, task) pairs, 126 of them: 76 fleet adapters with one task each, plus 5 external
experts across their 10-task suites. An expert is not a single target; its weak tasks are where the
method has anything to do.

CANDIDATES per target: 8 fleet + 8 external, stratified rather than sampled uniformly from the 83.
Uniform sampling would put ~0.8 external adapters in a candidate list and the mixed-pool question would
go untested; stratifying guarantees both populations are represented and lets the probe choose.

TWO OPERATOR FORMS, chosen by a measured property rather than by target provenance:
    capture@16 >= 0.6   ->  W = S_16 + r_target + c*r_donor     (fleet: capture ~0.80)
    capture@16 <  0.6   ->  W = D_target + c*r_donor            (external: capture ~0.11)
The second keeps the incoming adapter intact. This is not a change of method: swapping our shared
component into a foreign adapter was measured at 0% on every task tried, because S_16 simply does not
describe it. Using capture to pick the form makes the rule mechanical and testable on any new adapter.
Fleet-target/fleet-donor probes already exist under this form (e27, 604 of them) and are reused.
"""
import glob
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

K = 16
PROBE_R = 10
COMMIT_R = 25
BASE_R = 25
N_FLEET_CAND = 8
N_PICK = 3
CAPTURE_GATE = 0.6
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
FLEET_NORM = META["norm_stats"]
EXTPATH = {p.split("/")[-3]: p for p in
           glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/adapter_model.safetensors")}
EXTNORM = {}
for _p in glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/assets/*/*/norm_stats.json"):
    EXTNORM[_p.split("/")[-6]] = _p
# Only goal_expert has an UNAMBIGUOUS suite. The other four were trained on unions -- their norm-stats
# paths read libero_10_goal, libero_object_10, libero_10_goal_object, libero_goal_object -- so which
# suite to score them on is a guess, and guessing wrong is exactly the failure that made an expert read
# 0% earlier. They remain DONORS (donors need no evaluation); promoting them to targets requires
# knowing which suite each was actually trained for.
EXT_SUITE = {"pi05_libero_goal_expert": "libero_goal"}
CHUNK = max(6, 2 * META["n_workers"])
E27 = json.loads(open(REPO / "results/analysis/probe_selector76_report.txt")
                 .read().split("probe rankings: ")[1].strip())
_cache = {}
_cap = {}



def _clear_orphaned_builds():
    """Adapters in decomp_adapters/ are build->eval->delete transients. If no orchestrator is running,
    any that exist were abandoned by a killed run -- and 30+ of them is a blown disk quota (happened
    twice). Clear them at startup rather than trusting every kill path to clean up."""
    import subprocess as _sp
    if _sp.run(["pgrep", "-f", "merge_orchestrat[o]r.py"], capture_output=True).returncode != 0:
        for _f in (REPO / "results/analysis/decomp_adapters").glob("*.safetensors"):
            _f.unlink(missing_ok=True)


_clear_orphaned_builds()

def fold(key):
    """key: int -> fleet adapter; str -> external expert name."""
    if key in _cache:
        return _cache[key]
    if len(_cache) > 14:
        _cache.clear()
    path = afile[key] if isinstance(key, int) else EXTPATH[key]
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
    return _cache.setdefault(key, out)


def capture(key):
    if key in _cap:
        return _cap[key]
    f = fold(key)
    num = den = 0.0
    for s in stems:
        if s not in f:
            continue
        b, a = f[s]
        D = b @ a
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        P = Un @ (Un.T @ D @ VS[s]) @ VS[s].T
        num += float((P ** 2).sum()); den += float((D ** 2).sum())
    return _cap.setdefault(key, num / max(den, 1e-12))


def resid_norm(key):
    f = fold(key)
    tot = 0.0
    for s in stems:
        if s not in f:
            continue
        b, a = f[s]
        D = b @ a
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        tot += float(((D - Un @ (Un.T @ D @ VS[s]) @ VS[s].T) ** 2).sum())
    return tot ** 0.5


def build(name, tgt, donors):
    """donors: [(key, coeff)]. Operator form chosen by the target's capture@16."""
    keep_intact = capture(tgt) < CAPTURE_GATE
    tf = fold(tgt)
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        if keep_intact:
            Bb, Ab = [], []
            if stem in tf:
                b, a = tf[stem]
                Bb += [b]; Ab += [a]
        else:
            Bb, Ab = [Usig], [V.T]
            if stem in tf:
                b, a = tf[stem]
                M = U.T @ b; N = a @ V
                Bb += [b, -(U @ M)]; Ab += [a, N @ V.T]
        for dk, c in donors:
            fd = fold(dk)
            if stem not in fd:
                continue
            b, a = fd[stem]
            M = U.T @ b; N = a @ V
            Bb += [c * b, -c * (U @ M)]
            Ab += [a, N @ V.T]
        if not Bb:
            continue
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = REPO / f"results/analysis/decomp_adapters/{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def get(job, task):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return None
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok" and d["scene"] == task:
            return d["success_rate"]
    return None


def run(jobs, n, suite, norm):
    if not jobs:
        return
    for j in jobs:
        j["suite"] = suite
        j["norm_stats"] = norm
    planp = REPO / "sweep/_unified_current.json"
    json.dump(dict(META, n_trials=n, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def targets():
    """[(tag, target_key, task, suite, norm)] -- 126 (adapter, task) pairs."""
    out = []
    for t in range(76):
        out.append((f"f{t}", t, scenes[t][0], "libero_90", FLEET_NORM))
    # task names per external suite, taken from the LIBERO benchmark itself
    from exttgt_full_driver import TASKS as GOAL_TASKS
    suite_tasks = {"libero_goal": GOAL_TASKS}
    for nm in sorted(EXTNORM):
        su = EXT_SUITE.get(nm)
        if su is None:
            continue
        tasks = suite_tasks.get(su)
        if tasks is None:                       # discover from the benchmark on demand
            continue
        for tk in tasks:
            out.append((f"x{nm[11:]}_{tasks.index(tk)}", nm, tk, su, EXTNORM[nm]))
    return out


def main():
    rng = np.random.default_rng(2026)
    ref = 0.5 * float(np.mean([resid_norm(t) for t in range(0, 76, 8)]))
    ext_names = sorted(EXTPATH)
    tgts = targets()
    print(f"[unified] {len(tgts)} targets | donor pool {76 + len(ext_names)} "
          f"({len(ext_names)} external) | candidates 8 fleet + {len(ext_names)} external", flush=True)
    for tag, tk, task, suite, norm in tgts:
        cap = capture(tk)
        form = "keep-intact" if cap < CAPTURE_GATE else "S16"
        # baseline
        if get(f"e44_base_{tag}", task) is None:
            p = build(f"u_base_{tag}", tk, [])
            run([{"name": f"e44_base_{tag}", "method": "average", "params": {}, "adapters": [],
                  "adapter_paths": [p], "eval_scenes": [task]}], BASE_R, suite, norm)
            pathlib.Path(p).unlink(missing_ok=True)
        base = get(f"e44_base_{tag}", task)
        if base is None:
            print(f"[skip] {tag} {task[:30]}: no baseline", flush=True)
            continue
        # candidates: 8 fleet (excluding self) + all external (excluding self)
        # For fleet targets reuse the SAME 8 candidates e27 probed, so those 604 probes count and we
        # only pay for the external half of the candidate list.
        if isinstance(tk, int) and str(tk) in E27:
            fleet_c = [p for p, _ in E27[str(tk)][:N_FLEET_CAND]]
        else:
            fleet_c = [int(x) for x in rng.choice([c for c in range(76) if c != tk], N_FLEET_CAND,
                                                  replace=False)]
        cands = [(c, "int") for c in fleet_c] + [(nm, "ext") for nm in ext_names if nm != tk]
        specs = [(f"e44_p_{tag}_{c if isinstance(c,int) else c[11:]}", c) for c, _ in cands]
        specs = [(j, c) for (j, c) in specs if get(j, task) is None]
        for i in range(0, len(specs), CHUNK):
            jobs, built = [], []
            for job, c in specs[i:i + CHUNK]:
                p = build(f"u_{job}", tk, [(c, ref / resid_norm(c))])
                built.append(p)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": [task]})
            run(jobs, PROBE_R, suite, norm)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)
        sc = sorted(((get(f"e44_p_{tag}_{c if isinstance(c,int) else c[11:]}", task), c, kind)
                     for c, kind in cands
                     if get(f"e44_p_{tag}_{c if isinstance(c,int) else c[11:]}", task) is not None),
                    key=lambda z: -z[0])
        if not sc:
            continue
        top = sc[:N_PICK]
        nx = sum(1 for _, _, k in top if k == "ext")
        if sc[0][0] <= base:
            print(f"[{tag}] {task[:26]:<28} cap {cap:.2f} {form:<12} base {base*100:3.0f}% "
                  f"best {sc[0][0]*100:3.0f}% -> ABSTAIN", flush=True)
            continue
        if get(f"e44_c_{tag}", task) is None:
            p = build(f"u_c_{tag}", tk, [(c, (ref / resid_norm(c)) / N_PICK) for _, c, _ in top])
            run([{"name": f"e44_c_{tag}", "method": "average", "params": {}, "adapters": [],
                  "adapter_paths": [p], "eval_scenes": [task]}], COMMIT_R, suite, norm)
            pathlib.Path(p).unlink(missing_ok=True)
        cm = get(f"e44_c_{tag}", task)
        print(f"[{tag}] {task[:26]:<28} cap {cap:.2f} {form:<12} base {base*100:3.0f}% -> "
              f"{'--' if cm is None else f'{cm*100:3.0f}%'}  ext-in-top3 {nx}/3", flush=True)
        report(tgts)
    report(tgts)


def report(tgts):
    from scipy import stats
    rows = []
    for tag, tk, task, suite, norm in tgts:
        b = get(f"e44_base_{tag}", task)
        if b is None:
            continue
        c = get(f"e44_c_{tag}", task)
        eff = c if c is not None else b                 # abstain => unchanged
        kind = "fleet" if isinstance(tk, int) else "external"
        rows.append((tag, kind, b, eff, (eff - b) * 100, c is not None))
    L = ["=== EXP-32: UNIFIED 84-LoRA POOL — every eligible adapter a target in turn ===",
         "84 donors (76 fleet + 8 external); 81 target-capable; candidates = 8 fleet + 8 external",
         "operator form chosen by capture@16 (>=0.6 S16 form, <0.6 keep-intact)", ""]
    if rows:
        d = np.array([r[4] for r in rows]); b = np.array([r[2] for r in rows])
        kind = np.array([r[1] for r in rows]); weak = b < 0.90
        L.append(f"  ALL {len(d)} targets : {d.mean():+.2f}pp (SE {stats.sem(d):.2f})  "
                 f"committed {int(sum(r[5] for r in rows))}  abstained {int(sum(not r[5] for r in rows))}")
        for nm, m in (("fleet targets", kind == "fleet"), ("external targets", kind == "external"),
                      ("WEAK  (base<90%)", weak), ("STRONG(base>=90%)", ~weak)):
            if m.sum():
                L.append(f"  {nm:<20} n={int(m.sum()):<4} {d[m].mean():+.2f}pp  "
                         f"improved {int((d[m]>0).sum())}  flat {int((d[m]==0).sum())}  "
                         f"harmed {int((d[m]<0).sum())}")
        L.append(f"\n  pool-wide success: {b.mean()*100:.1f}% -> {np.array([r[3] for r in rows]).mean()*100:.1f}%")
    open(REPO / "results/analysis/unified_pool_report.txt", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
