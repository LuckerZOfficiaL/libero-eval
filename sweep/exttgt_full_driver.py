#!/usr/bin/env python3
"""EXP-29 — apply the method to an external expert's WHOLE suite, not just its weakest tasks.

EXP-27 established the direction (+16.0pp) but on a hand-picked subset: the 3 weakest tasks of one
expert. By this project's own reporting rule that is a smoke test, and its weakness is N, not rollout
count. So this runs every task in the suite and reports the deployment-grade split -- all / weak /
strong, with the count harmed -- exactly the shape that turned the internal method's 5-target pilot
(+4.0pp, which did not survive) into the honest all-76 screen.

Precision is fixed in passing: baselines are re-measured at the COMMIT budget so the before/after
comparison is like-for-like. EXP-27 compared base@10 (SD ~13pp) against commit@25 (SD ~9pp), which is
not a fair difference.

Per task:
    baseline @25
    8 pool donors probed @10  (the standard probe, its budget already shown optimal)
    top-3 committed @25, under the ABSTENTION GUARD: commit only if the best probe beats the baseline.
      That guard took the internal method from +2.22pp to +2.77pp and cut harmed targets 3 -> 1, and
      it is the mechanism that should protect the already-strong tasks here.

Donors are energy-matched to the internal dose, and the incoming adapter is kept INTACT
(W = D_external + c*r_donor) because our S_16 captures only ~0.11 of a foreign adapter.
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
BASE_R = 25
PROBE_R = 10
COMMIT_R = 25
N_CAND = 8
N_PICK = 3
EXPERT = "pi05_libero_goal_expert"
SUITE = "libero_goal"
TASKS = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove",
         "put_the_wine_bottle_on_top_of_the_cabinet", "open_the_top_drawer_and_put_the_bowl_inside",
         "put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack",
         "turn_on_the_stove", "put_the_cream_cheese_in_the_bowl",
         "push_the_plate_to_the_front_of_the_stove", "put_the_bowl_on_top_of_the_cabinet"]
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
USIG = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META = hostcfg.apply(META)
EXTPATH = {p.split("/")[-3]: p for p in
           glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/adapter_model.safetensors")}
EXTNORM = {}
for _p in glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/assets/*/*/norm_stats.json"):
    EXTNORM[_p.split("/")[-6]] = _p
rng = np.random.default_rng(123)
CANDS = sorted(int(x) for x in rng.choice(76, N_CAND, replace=False))
CHUNK = max(6, 2 * META["n_workers"])          # bound disk: each merged policy is ~0.9GB
_cache = {}


def fold(p):
    key = str(p)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 14:
        _cache.clear()
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(p))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return _cache.setdefault(key, out)


def resid_norm(f):
    tot = 0.0
    for s in stems:
        if s not in f:
            continue
        b, a = f[s]
        D = b @ a
        Un = USIG[s] / torch.linalg.norm(USIG[s], dim=0).clamp(min=1e-12)
        tot += float(((D - Un @ (Un.T @ D @ VS[s]) @ VS[s].T) ** 2).sum())
    return tot ** 0.5


def build(name, tgt_fold, donors):
    """W = D_external + sum(c * r_donor): the foreign adapter is kept intact."""
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [], []
        if stem in tgt_fold:
            b, a = tgt_fold[stem]
            Bb += [b]; Ab += [a]
        for f, c in donors:
            if stem not in f:
                continue
            b, a = f[stem]
            M = U.T @ b
            N = a @ V
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


def run(jobs, n):
    if not jobs:
        return
    for j in jobs:
        j["suite"] = SUITE
        j["norm_stats"] = EXTNORM[EXPERT]
    planp = REPO / "sweep/_extfull_current.json"
    json.dump(dict(META, n_trials=n, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    tf = fold(EXTPATH[EXPERT])
    ref = 0.5 * float(np.mean([resid_norm(fold(afile[t])) for t in range(0, 76, 8)]))
    dcoef = {c: ref / resid_norm(fold(afile[c])) for c in CANDS}
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(90)

    # ---- stage 1: baselines at the COMMIT budget, all tasks ----
    todo = [t for t in TASKS if get("e43_base", t) is None]
    if todo:
        p = build("xf_base", tf, [])
        print(f"[base] {len(todo)} tasks @{BASE_R}", flush=True)
        run([{"name": "e43_base", "method": "average", "params": {}, "adapters": [],
              "adapter_paths": [p], "eval_scenes": todo}], BASE_R)
        pathlib.Path(p).unlink(missing_ok=True)
    base = {t: get("e43_base", t) for t in TASKS}
    print("[base] " + "  ".join(f"{t[:22]}={None if base[t] is None else f'{base[t]*100:.0f}%'}"
                                for t in TASKS), flush=True)

    # ---- stage 2: probe every task, chunked to bound disk ----
    specs = [(f"e43_probe_{TASKS.index(t)}_p{c}", t, c) for t in TASKS for c in CANDS]
    specs = [s for s in specs if get(s[0], s[1]) is None]
    for i in range(0, len(specs), CHUNK):
        jobs, built = [], []
        for job, t, c in specs[i:i + CHUNK]:
            p = build(f"xf_{job}", tf, [(fold(afile[c]), dcoef[c])])
            built.append(p)
            jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                         "adapter_paths": [p], "eval_scenes": [t]})
        print(f"[probe] chunk {i//CHUNK+1}/{(len(specs)+CHUNK-1)//CHUNK}", flush=True)
        run(jobs, PROBE_R)
        for p in built:
            pathlib.Path(p).unlink(missing_ok=True)

    # ---- stage 3: commit top-3 under the abstention guard ----
    jobs, built, picks = [], [], {}
    for t in TASKS:
        i = TASKS.index(t)
        sc = sorted(((get(f"e43_probe_{i}_p{c}", t), c) for c in CANDS
                     if get(f"e43_probe_{i}_p{c}", t) is not None), reverse=True)
        if not sc or base[t] is None:
            continue
        best, top = sc[0][0], [c for _, c in sc[:N_PICK]]
        picks[t] = (top, best)
        if best <= base[t]:
            print(f"[guard] {t[:34]}: best probe {best*100:.0f}% <= base {base[t]*100:.0f}% -> ABSTAIN",
                  flush=True)
            continue
        if get(f"e43_commit_{i}", t) is not None:
            continue
        p = build(f"xf_c{i}", tf, [(fold(afile[c]), dcoef[c] / N_PICK) for c in top])
        built.append(p)
        jobs.append({"name": f"e43_commit_{i}", "method": "average", "params": {}, "adapters": [],
                     "adapter_paths": [p], "eval_scenes": [t]})
    for i in range(0, len(jobs), CHUNK):
        run(jobs[i:i + CHUNK], COMMIT_R)
    for p in built:
        pathlib.Path(p).unlink(missing_ok=True)
    json.dump({t: {"top3": v[0], "best_probe": v[1]} for t, v in picks.items()},
              open(REPO / "results/analysis/exttgt_full_picks.json", "w"), indent=1)
    report(picks)


def report(picks):
    from scipy import stats
    L = [f"=== EXP-29: the method applied to the WHOLE {SUITE} suite of {EXPERT} ===",
         "baseline @25, 8 pool donors probed @10, top-3 committed @25 under the abstention guard",
         "(EXP-27 measured only the 3 weakest tasks, with a 10-rollout baseline)", "",
         f"{'task':<44}{'base':>7}{'best probe':>12}{'commit':>9}{'delta':>8}{'action':>10}"]
    rows = []
    for t in TASKS:
        i = TASKS.index(t)
        b = get("e43_base", t)
        c = get(f"e43_commit_{i}", t)
        bp = picks.get(t, (None, None))[1]
        act = "abstain" if (t in picks and c is None) else ("commit" if c is not None else "-")
        row = f"{t[:42]:<44}"
        row += f"{'' if b is None else f'{b*100:.0f}%':>7}"
        row += f"{'' if bp is None else f'{bp*100:.0f}%':>12}"
        row += f"{'' if c is None else f'{c*100:.0f}%':>9}"
        if b is not None:
            eff = (c if c is not None else b)          # abstain => the policy is unchanged
            rows.append((t, b, eff, (eff - b) * 100))
            row += f"{(eff-b)*100:>+8.0f}"
        row += f"{act:>10}"
        L.append(row)
    if rows:
        d = np.array([r[3] for r in rows])
        b = np.array([r[1] for r in rows])
        weak = b < 0.90
        L += ["", f"  ALL {len(d)} tasks : {d.mean():+.2f}pp  (SE {stats.sem(d):.2f})"]
        for nm, m in (("WEAK  (base<90%)", weak), ("STRONG(base>=90%)", ~weak)):
            if m.sum():
                L.append(f"  {nm} n={int(m.sum()):<3}: {d[m].mean():+.2f}pp  "
                         f"improved {int((d[m]>0).sum())}  flat {int((d[m]==0).sum())}  "
                         f"harmed {int((d[m]<0).sum())}")
        L.append(f"  suite success: {b.mean()*100:.1f}% -> {np.array([r[2] for r in rows]).mean()*100:.1f}%")
        L += ["", "  (EXP-27, 3 weakest tasks only, base@10: +16.0pp)"]
    open(REPO / "results/analysis/exttgt_full_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
