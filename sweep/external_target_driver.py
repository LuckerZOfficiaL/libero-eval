#!/usr/bin/env python3
"""EXP-27 — THE REAL DEPLOYMENT TEST: an EXTERNAL LoRA is the TARGET, our pool supplies the donors.

Every result so far improves a LoRA that came from our own fleet. The question that matters for anyone
adopting this is the reverse: **someone hands you a LoRA trained elsewhere — can your pool improve it?**

OPERATOR CHANGE, forced by measurement. For an internal target, S_16 + r_T reconstructs the target
(capture@16 = 0.84-0.87), which is why "swap the shared component" was free. For an EXTERNAL target
capture is only 0.19-0.39, so S_16 + r_ext would REPLACE most of the incoming adapter's shared structure
with ours — a large, unjustified edit. The honest deployment form keeps the incoming LoRA intact:

    W = D_external  +  alpha * r_donor          (r_donor = donor residual w.r.t. our soup basis)

i.e. add a whisper of a pool partner's task-specific part to the adapter as given. alpha is energy-
matched to the dose internal targets receive (mean ||0.5 * r_internal||), so this is the same treatment,
not a bigger one.

We also run ONE diagnostic arm per task, `swap`: W = S_16 + r_ext, which asks directly whether our
fleet's shared component is compatible with a foreign adapter at all. If swap collapses, that measures
the pool-consistency boundary; if it survives, the shared component is more universal than capture@16
suggests.

STAGE 1  baseline: the external expert alone on its own suite (10 tasks @10 rollouts) -> find headroom.
STAGE 2  on the WEAKEST tasks: probe 8 pool donors @10, commit top-3 @25, plus the swap diagnostic.
"""
import glob
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

OUTD = REPO / "results/analysis/decomp_adapters"
K = 16
PROBE_R = 10
COMMIT_R = 25
N_CAND = 8
N_WEAK = 3                 # tasks per expert taken into stage 2
EXPERTS = {                # external target -> its suite
    "pi05_libero_goal_expert": "libero_goal",
    "pi05_libero_spatial_expert": "libero_spatial",
}
TASKS = {
    "libero_goal": ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove",
                    "put_the_wine_bottle_on_top_of_the_cabinet",
                    "open_the_top_drawer_and_put_the_bowl_inside",
                    "put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack",
                    "turn_on_the_stove", "put_the_cream_cheese_in_the_bowl",
                    "push_the_plate_to_the_front_of_the_stove", "put_the_bowl_on_top_of_the_cabinet"],
    "libero_spatial": ["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
                       "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"],
}
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
z = np.load(REPO / "results/analysis/loo_soup_factors_t1.npz")
stems = [str(s) for s in z["stems"]]
USIG = {s: torch.from_numpy(z[f"U_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
VS = {s: torch.from_numpy(z[f"V_{i}"]).float()[:, :K] for i, s in enumerate(stems)}
TMPL = json.load(open(REPO / "sweep/e18e_plan.json"))
META = {k: TMPL[k] for k in ("base_model", "videos_dir", "logdir", "save_videos", "results_dir",
                             "n_workers", "gpu_per_worker", "port_per_worker", "norm_stats")}
META["n_workers"] = 3
META["gpu_per_worker"] = [1, 2, 1]
EXTPATH = {p.split("/")[-3]: p for p in
           glob.glob("/media/pinas/Chris/pi05_libero_lora_experts/*/*/adapter_model.safetensors")}
rng = np.random.default_rng(123)
CANDS = sorted(int(x) for x in rng.choice(76, N_CAND, replace=False))
_cache = {}


def fold(p):
    key = str(p)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 20:
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


def build_keep(name, tgt_fold, donors):
    """W = D_target + sum(c * r_donor): the incoming adapter is kept INTACT (full delta, no projection);
    only the donors are reduced to residuals."""
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [], []
        if stem in tgt_fold:
            b, a = tgt_fold[stem]
            Bb += [b]; Ab += [a]                       # full external delta, untouched
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
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def build_swap(name, tgt_fold):
    """Diagnostic: W = S_16 + r_target — replace the foreign shared component with OURS."""
    tensors = {}
    for stem in stems:
        Usig = USIG[stem]
        U = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        V = VS[stem]
        Bb, Ab = [Usig], [V.T]
        if stem in tgt_fold:
            b, a = tgt_fold[stem]
            M = U.T @ b
            N = a @ V
            Bb += [b, -(U @ M)]
            Ab += [a, N @ V.T]
        tensors[f"{stem}.lora_A.weight"] = torch.cat(Ab, 0).contiguous()
        tensors[f"{stem}.lora_B.weight"] = torch.cat(Bb, 1).contiguous()
    p_ = OUTD / f"{name}.safetensors"
    st.save_file(tensors, str(p_))
    return str(p_)


def rate(job, task):
    f = REPO / f"results/merge/{job}.jsonl"
    if not f.exists():
        return None
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok" and d["scene"] == task:
            return d["success_rate"]
    return None


def run_plan(jobs, n, suite):
    if not jobs:
        return
    for j in jobs:
        j["suite"] = suite
    planp = REPO / "sweep/_extt_current.json"
    json.dump(dict(META, n_trials=n, jobs=jobs), open(planp, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"),
                    str(planp)], cwd=REPO)


def main():
    ref = 0.5 * float(np.mean([resid_norm(fold(afile[t])) for t in range(0, 76, 8)]))
    dcoef = {c: ref / resid_norm(fold(afile[c])) for c in CANDS}
    print(f"[energy] reference dose {ref:.2f}; donor coeffs "
          f"{ {c: round(v,3) for c,v in dcoef.items()} }", flush=True)
    while subprocess.run(["pgrep", "-f", "merge_orchestrat[o]r.py"],
                         capture_output=True).returncode == 0:
        time.sleep(120)
    for expert, suite in EXPERTS.items():
        tf = fold(EXTPATH[expert])
        # ---- stage 1: the expert alone, on its own suite ----
        todo = [t for t in TASKS[suite] if rate(f"e37_base_{expert}", t) is None]
        if todo:
            path = build_keep(f"et_base_{expert}", tf, [])
            print(f"[base] {expert}: {len(todo)} tasks", flush=True)
            run_plan([{"name": f"e37_base_{expert}", "method": "average", "params": {},
                       "adapters": [], "adapter_paths": [path], "eval_scenes": todo}], PROBE_R, suite)
            pathlib.Path(path).unlink(missing_ok=True)
        scored = sorted((rate(f"e37_base_{expert}", t), t) for t in TASKS[suite]
                        if rate(f"e37_base_{expert}", t) is not None)
        weak = [t for _, t in scored[:N_WEAK]]
        print(f"[weak] {expert}: {[(t[:38], f'{s*100:.0f}%') for s, t in scored[:N_WEAK]]}", flush=True)
        # ---- stage 2: probe our pool as donors, plus the swap diagnostic ----
        jobs, built = [], []
        for c in CANDS:
            job = f"e37_probe_{expert}_p{c}"
            miss = [t for t in weak if rate(job, t) is None]
            if miss:
                path = build_keep(f"et_p{c}_{expert}", tf, [(fold(afile[c]), dcoef[c])])
                built.append(path)
                jobs.append({"name": job, "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [path], "eval_scenes": miss})
        if any(rate(f"e37_swap_{expert}", t) is None for t in weak):
            path = build_swap(f"et_swap_{expert}", tf)
            built.append(path)
            jobs.append({"name": f"e37_swap_{expert}", "method": "average", "params": {},
                         "adapters": [], "adapter_paths": [path], "eval_scenes": weak})
        if jobs:
            print(f"[probe] {expert}: {len(jobs)} arms x {len(weak)} tasks", flush=True)
            run_plan(jobs, PROBE_R, suite)
        for p_ in built:
            pathlib.Path(p_).unlink(missing_ok=True)
        # ---- commit top-3 donors per weak task ----
        for t in weak:
            sc = sorted(((rate(f"e37_probe_{expert}_p{c}", t), c) for c in CANDS
                         if rate(f"e37_probe_{expert}_p{c}", t) is not None), reverse=True)
            if not sc:
                continue
            top = [c for _, c in sc[:3]]
            b = rate(f"e37_base_{expert}", t)
            print(f"[select] {expert} / {t[:34]}: base {b*100:.0f}%  best probe {sc[0][0]*100:.0f}%  "
                  f"top3 {top}", flush=True)
            jn = f"e37_commit_{expert}_{TASKS[suite].index(t)}"
            if rate(jn, t) is None:
                path = build_keep(f"et_c_{expert}_{TASKS[suite].index(t)}", tf,
                                  [(fold(afile[c]), dcoef[c] / 3) for c in top])
                run_plan([{"name": jn, "method": "average", "params": {}, "adapters": [],
                           "adapter_paths": [path], "eval_scenes": [t]}], COMMIT_R, suite)
                pathlib.Path(path).unlink(missing_ok=True)
        print(f"[done] {expert}", flush=True)
        report()
    report()


def report():
    L = ["=== EXP-27: EXTERNAL LoRA AS THE TARGET — can our pool improve a foreign adapter? ===",
         "operator W = D_external + alpha*r_donor (incoming adapter kept INTACT; donors energy-matched)",
         "'swap' = S_16 + r_external, the diagnostic for whether OUR shared component fits a foreign LoRA", ""]
    for expert, suite in EXPERTS.items():
        sc = sorted((rate(f"e37_base_{expert}", t), t) for t in TASKS[suite]
                    if rate(f"e37_base_{expert}", t) is not None)
        if not sc:
            continue
        L.append(f"{expert}  ({suite}, {len(sc)} tasks, mean {100*np.mean([s for s,_ in sc]):.0f}%)")
        L.append(f"  {'task':<40}{'base':>7}{'swap':>7}{'best probe':>12}{'commit':>9}{'delta':>7}")
        d = []
        for s, t in sc[:N_WEAK]:
            sw = rate(f"e37_swap_{expert}", t)
            pr = [rate(f"e37_probe_{expert}_p{c}", t) for c in CANDS]
            pr = [x for x in pr if x is not None]
            cm = rate(f"e37_commit_{expert}_{TASKS[suite].index(t)}", t)
            row = f"  {t[:38]:<40}{s*100:>6.0f}%"
            row += f"{'' if sw is None else f'{sw*100:.0f}%':>7}"
            row += f"{'' if not pr else f'{max(pr)*100:.0f}%':>12}"
            row += f"{'' if cm is None else f'{cm*100:.0f}%':>9}"
            if cm is not None:
                d.append((cm - s) * 100)
                row += f"{(cm-s)*100:>+7.0f}"
            L.append(row)
        if d:
            L.append(f"  -> mean improvement on its weakest {len(d)} tasks: {np.mean(d):+.1f}pp")
        L.append("")
    open(REPO / "results/analysis/external_target_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
