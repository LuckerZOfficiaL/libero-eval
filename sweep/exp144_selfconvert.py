#!/usr/bin/env python3
"""EXP-144 — data-free conversion of from-scratch skills into mergeable form through their OWN rollouts (10-26 pilot).

EXP-138..143: a from-scratch (foreign-A) external is dead on arrival in any training-free merge — it needs near-full weight
on its own inputs, its update fires on every task, the library's shared part S breaks many externals even at full weight,
and no token pathway localises the skill. Write-protected (WP) adapters trained on the skill's DEMOS merge well (EXP-124/134:
new skills 52.6% at t=8 vs 17.3% from scratch), but in the from-scratch setting the skill's data is not available.
VLA-specific route: a robot skill can generate its own training data — run the received LoRA in closed loop, keep the
successful episodes (the task's success signal), and train a WP adapter (S16 init, A frozen at V^T, no writes into the
shared outputs; `train_lora.py --init s16wp --rollouts`) on them. No original demonstrations are used.
Pilot: order 0 (EXP-116), its 8 externals.
  collect  each external's single-task model (W0 + its LoRA), 50 episodes on initial states 10..49 (test states 0..9 never
           used), dumps every 2 steps + every executed action -> results/exp144_dumps/
  train    WP adapter per external on its successful episodes only, 1,500 steps, batch 8 (EXP-123 recipe)
           -> results/analysis/new_loras_s16wp_self/
  base     converted adapters' single-task success, 2 x 10 episodes, states 0..9
  run      the EXP-134 wp merge (S16 + 0.25 x sum of new-skill residuals + 1/k x library residuals) at t = 8 with the
           converted adapters (c144s0t8-ours_w0.25), all 24 skills, 10 eps, states 0..9
References (same skills, order, states): scratch ours c134zs0t8-ours; WP-on-demos c134ws0t8-ours_w0.25.
PRE-REGISTERED (10-26, before any rollout): new-skill retention measured against the ORIGINAL from-scratch skill's single-task
success. Conversion works if the converted merge keeps >= 30% of the externals (scratch ours 5.7% on order 0) with library
>= scratch ours - 5pp; matches WP-on-demos if within 10pp of it. Also reported: converted adapters' single-task success
(the price of training on own rollouts).
10-26 RESULT order 0 (w 0.25): externals 44.5% (scratch ours 5.7, WP on demos 61.0), library 70.4% (76.1 / 78.5).
Extension: E144_SEQS (orders; their externals are converted, order 0's registered first), E144_WS (new-skill weights).
Env: E144_STAGE collect|train|calib|base|run|report; E144_SHARD k/n (collect; train is a shared claim queue); E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp128_retain as RT                                            # RBuilder (priv); registers the WP-on-demos adapters
import exp124_wpcontinual as W
import exp116_continual as C
import exp112_bench as E
import exp113_controls as X
import unified_pool_driver as U
# 10-26 (after the order-0 pilot): E144_SEQS = orders whose externals are converted / merged, E144_WS = new-skill weights
SEQS = [int(x) for x in os.environ.get("E144_SEQS", "0").split(",")]
WS = [float(x) for x in os.environ.get("E144_WS", "0.25").split(",")]
T, NCOL, OFF, POOL = 8, 50, 10, 40
# 10-26: rollout dumps live on the NAS archive (shiva's 50 GB home quota); E144_DUMPS overrides
DUMPS = pathlib.Path(os.environ.get("E144_DUMPS", "/media/pinas/lzhou/libero-eval-archive/exp144_dumps"))
ADIR = REPO / "results/analysis/new_loras_s16wp_self"
NEWS = []                                                            # union of the orders' externals; order 0's first keeps
for s_ in sorted(set(SEQS) | {0}): NEWS += [i for i in C.sequence(s_)[1] if i not in NEWS]   # its LIB indices (record chunks)
CV = {}                                                              # original external index -> converted LIB index
for i in NEWS:
    e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / name / "adapter_model.safetensors"
    U.EXTPATH["np16self:" + name] = str(path)
    CV[i] = len(E.LIB); E.LIB.append(dict(e, key="np16self:" + name, tid="v" + e["tid"][1:], path=str(path)))
def tag(s_, w): return f"c144s{s_}t{len(C.sequence(s_)[1])}-ours_w{w:g}"   # t = 8 (22 at C.SCALE)

def orch(jobs, tag, **extra):
    p = E.plan(jobs, extra.pop("n", 10), extra.pop("offset", 0), tag)
    if extra: d = json.load(open(p)); d.update(extra); json.dump(d, open(p, "w"), indent=1)
    subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(p)], cwd=REPO)

def collect():
    J = [{"name": f"e144_{E.LIB[i]['tid']}_self", "method": "average", "params": {}, "adapters": [], "adapter_paths": [E.LIB[i]["path"]],
          "eval_scenes": [E.LIB[i]["scene"]], "suite": E.LIB[i]["suite"], "norm_stats": U.FLEET_NORM}
         for i in NEWS if E.rec(f"e144_{E.LIB[i]['tid']}_self", E.LIB[i]["scene"]) is None]
    if os.environ.get("E144_SHARD"):                                 # 10-26: shard on the skill's position in NEWS (stable across launches)
        k_, n_ = (int(x) for x in os.environ["E144_SHARD"].split("/"))
        J = [j for j in J if [f"e144_{E.LIB[i]['tid']}_self" for i in NEWS].index(j["name"]) % n_ == k_]
    for j in J:                                                      # a failed earlier attempt's partial episodes -> set aside
        d = DUMPS / j["name"]
        if d.exists(): d.rename(d.with_name(f"{d.name}_partial_{time.strftime('%m%d%H%M')}"))
    print(f"[e144] collect: {len(J)} skills x {NCOL} episodes (states {OFF}..{OFF + POOL - 1})", flush=True)
    if J: orch(J, "e144col", n=NCOL, offset=OFF, state_pool=POOL, dump_obs_dir=str(DUMPS), dump_stride=2)

def pattern(i): return str(DUMPS / f"e144_{E.LIB[i]['tid']}_self" / "*" / "ep*_success.npz")

def train():
    # 10-26: shared queue across GPUs — claim the next collected, untrained skill with a lock file; wait for pending collections
    # until the collector writes ADIR/.collect_done
    while True:
        left = [i for i in NEWS if not pathlib.Path(E.LIB[CV[i]]["path"]).exists() and not (ADIR / f".claim_{E.LIB[i]['tid']}").exists()]
        ready = [i for i in left if E.rec(f"e144_{E.LIB[i]['tid']}_self", E.LIB[i]["scene"])]
        if not ready:
            if not left or (ADIR / ".collect_done").exists(): break
            time.sleep(60); continue
        i = ready[0]; ADIR.mkdir(parents=True, exist_ok=True)
        try: os.close(os.open(ADIR / f".claim_{E.LIB[i]['tid']}", os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        e = E.LIB[i]; out = pathlib.Path(E.LIB[CV[i]]["path"]).parent; t1 = time.time()
        log = REPO / f"results/merge_logs/e144_{e['tid']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--suite", e["suite"] + "_no_noops",
                            "--task", e["scene"], "--steps", "1500", "--batch", "8", "--device", "cuda:0", "--init", "s16wp",
                            "--rollouts", pattern(i), "--out", str(out)], cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        print(f"[e144] trained {e['tid']} -> exit {r.returncode} in {(time.time()-t1)/60:.0f} min", flush=True)

def base():
    J = [{"name": f"e144_base_{E.LIB[CV[i]]['tid']}_r{r}", "method": "average", "params": {}, "adapters": [],
          "adapter_paths": [E.LIB[CV[i]]["path"]], "eval_scenes": [E.LIB[i]["scene"]], "suite": E.LIB[i]["suite"], "norm_stats": U.FLEET_NORM}
         for r in (1, 2) for i in NEWS if pathlib.Path(E.LIB[CV[i]]["path"]).exists() and E.rec(f"e144_base_{E.LIB[CV[i]]['tid']}_r{r}", E.LIB[i]["scene"]) is None]
    print(f"[e144] base: {len(J)} single-task jobs", flush=True)
    if J: orch(J, "e144base")

def calib():
    # 10-26: shiva GPU 1 off the bus blocks EGL machine-wide -> evals render with OSMesa (CPU); check it against the EGL numbers
    # with the original from-scratch skills' single-task runs (test states 0..9)
    J = [{"name": f"e144_osm_{E.LIB[i]['tid']}", "method": "average", "params": {}, "adapters": [], "adapter_paths": [E.LIB[i]["path"]],
          "eval_scenes": [E.LIB[i]["scene"]], "suite": E.LIB[i]["suite"], "norm_stats": U.FLEET_NORM}
         for i in NEWS if E.rec(f"e144_osm_{E.LIB[i]['tid']}", E.LIB[i]["scene"]) is None]
    print(f"[e144] calib: {len(J)} single-task jobs (OSMesa)", flush=True)
    if J: orch(J, "e144osm")

def run():
    for s_ in SEQS:
        OLD, NEW = C.sequence(s_); newc = [CV[i] for i in NEW]; mem = OLD + newc; ev = sorted(mem)
        for w in WS:
            TAG = tag(s_, w)
            js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(mem), "test", s_, ev) if any(E.rec(nm, sc) is None for sc in scs)]
            print(f"[e144] run {TAG}: {len(js)} jobs", flush=True)
            if not js: continue
            out = E.PRE / f"e144_{TAG}"; t1 = time.time()
            RT.RBuilder().build("priv", {"w": w, "P": newc}, None, mem, out)
            print(f"[e144] built {TAG} in {time.time()-t1:.0f}s", flush=True)
            orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                   "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e144run")
            shutil.rmtree(out, ignore_errors=True)

def report():
    for s_ in SEQS: report_seq(s_)

def report_seq(SEQ):
    OLD, NEW = C.sequence(SEQ)
    sg = X.single(); L = [f"=== EXP-144 order {SEQ}, t={T}: from-scratch externals converted through their own rollouts ==="]
    def single_conv(i):
        rs = [E.rec(f"e144_base_{E.LIB[CV[i]]['tid']}_r{r}", E.LIB[i]["scene"]) for r in (1, 2)]; rs = [x["success_rate"] for x in rs if x]
        return np.mean(rs) if rs else None
    def col(i):
        n = s = 0
        for f in (DUMPS / f"e144_{E.LIB[i]['tid']}_self").glob("*/ep*.npz"): n += 1; s += f.name.endswith("_success.npz")
        return s, n
    ref = {"scratch ours (1/k)": (f"c134zs{SEQ}t{T}-ours", NEW), "WP on demos (0.25)": (f"c134ws{SEQ}t{T}-ours_w0.25", W.sequence(SEQ)[1]),
           **{f"converted ({w:g})": (tag(SEQ, w), [CV[i] for i in NEW]) for w in WS}}
    res = {}
    for nm, (tg, new) in ref.items():
        sc = E.member_success(tg, len(OLD) + len(new), "test", SEQ, sorted(OLD + new))
        if not sc: continue
        res[nm] = ({i: sc[i] / sg[i] for i in OLD if i in sc and sg.get(i)}, {o: sc[c] / sg[o] for o, c in zip(NEW, new) if c in sc and sg.get(o)})
    L.append("skill      own(scratch)  rollouts ok/n  converted single  |  retention vs own: " + " / ".join(res))
    osm = {i: E.rec(f"e144_osm_{E.LIB[i]['tid']}", E.LIB[i]["scene"]) for i in NEW}; osm = {i: r["success_rate"] for i, r in osm.items() if r}
    if osm: L.append("OSMesa check (original skills, single-task): " + "  ".join(f"{E.LIB[i]['tid']} {100*sg.get(i, 0):.0f}->{100*v:.0f}" for i, v in osm.items())
                     + f"  | mean EGL {100*np.mean([sg.get(i, 0) for i in osm]):.1f}% vs OSMesa {100*np.mean(list(osm.values())):.1f}%")
    for i in NEW:
        s, n = col(i); sc_ = single_conv(i)
        L.append(f"{E.LIB[i]['tid']:9s}  {100*sg.get(i, 0):5.0f}%      {s:3d}/{n:<3d}       " + ("    —   " if sc_ is None else f"{100*sc_:6.0f}%  ")
                 + "  |  " + " / ".join("  —  " if i not in res[k][1] else f"{100*res[k][1][i]:5.0f}%" for k in res))
    for k in res:
        lib, nw = res[k]
        L.append(f"{k:20s} library {100*np.mean(list(lib.values())) if lib else float('nan'):5.1f}% (n={len(lib)})  externals "
                 f"{100*np.mean(list(nw.values())) if nw else float('nan'):5.1f}% (n={len(nw)}, vs the original skill's own success)")
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp144_report_s{SEQ}_w{'-'.join(f'{w:g}' for w in WS)}.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E144_STAGE", "report"); U._clear_orphaned_builds()
    {"collect": collect, "train": train, "calib": calib, "base": base, "run": run, "report": report}[st_]()
    print(f"[e144] {st_} DONE", flush=True)
