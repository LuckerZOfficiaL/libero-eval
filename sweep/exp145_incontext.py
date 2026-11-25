#!/usr/bin/env python3
"""EXP-145 — in-context conversion: train each converted external INSIDE the merge it will live in (10-26 pilot, order 0).

EXP-144: from-scratch externals converted to write-protected (WP) adapters on their OWN successful rollouts keep 44.5% of
their success in the t=8 merge at new-skill weight 0.25 (single-task 81%), library 70.4% (scratch ours 76.1); w 0.125 / 0.5
-> externals 24.8 / 16.9%, library 79.0 / 30.0. The converted adapter is trained ALONE (S16 + B_perp V^T) but acts at weight
w next to 16 library and 7 other new residuals. The receiver knows that merge, so it can train the adapter for it.
  ctxr  per external i: the merge of the OTHER 23 members (S16 + 0.25 x the other converted residuals + 1/24 x the library
        residuals, as PBuilder builds it) folded into the frozen base; i's private part B_perp (A = V^T frozen) acts at x0.25,
        starts from its EXP-144 converted adapter, 750 steps batch 8 on its own successful rollouts (EXP-144 dumps), half of
        the steps on library replay: the library's own demos (5 per library skill of the order, its own prompt) through the
        same merged model (the receiver owns the library data; no external data is used).
  ctx   the same without replay (ablation: is the library cost the price of in-context training?).
        One parallel (Jacobi) round: every context holds the EXP-144 converted residuals of the other new skills.
  run   the EXP-144 priv merge (S16 + 0.25 x new residuals + 1/24 x library residuals) with the in-context adapters, t=8,
        all 24 skills, 10 eps, test states 0..9.
PRE-REGISTERED (10-26 19:40, before any training): retention vs the ORIGINAL from-scratch skill's single-task success, as
EXP-144. In-context training helps if externals >= EXP-144 converted (44.5%) + 10pp with library >= 70.4 - 3 (= 67.4%).
ctxr also removes the library cost if library >= scratch ours (76.1) - 3 (= 73.1%) with externals >= 44.5%.
10-27 RESULT: NO-GO as run — ctxr library 0.0 / externals 0.0 (the 8 updates share one library correction, cosine 0.67, summed
~8x), ctx 38.6 / 31.1 (each private part grows 1.7x). Follow-up (interp, no training): FedAvg-style aggregation of the parallel
round, adapter = converted + lam x (ctxr - converted) (both are S16 + B_perp V^T, so this is a linear blend of B), lam 1/8 = mean.
Env: E145_SEQ (order, default 0); E145_STAGE prep|train|interp|run|report; E145_VARS (default ctxr,ctx); E145_LAMS + E145_BLEND (blends of ctxr, or ctx
for the no-replay ablation, 11-04); E113_GPUS / E113_DEV.
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
SEQ = int(os.environ.get("E145_SEQ", "0"))                              # 11-05: orders 1-3 (generality)
os.environ.setdefault("E144_SEQS", "0" if SEQ == 0 else f"0,{SEQ}")
import exp144_selfconvert as S                                        # registers the converted adapters (S.CV), order 0 first
import exp128_retain as RT
import exp124_wpcontinual as W
import exp116_continual as C
import exp112_bench as E
import exp113_controls as X
import unified_pool_driver as U
T, WN, STEPS = len(C.sequence(SEQ)[1]), 0.25, 750                      # T = 8 (22 at the scaling pseudo-order, 11-06)
VARS = os.environ.get("E145_VARS", "ctxr,ctx").split(",")
LAMS = [float(x) for x in os.environ.get("E145_LAMS", "0.125,0.25").split(",") if x]   # 10-27: blends conv + lam (ctxr - conv)
BLEND = os.environ.get("E145_BLEND", "ctxr")                           # 11-04: blend base (ctx = the no-replay ablation)
VARS += [f"{BLEND}_l{l:g}" for l in LAMS if f"{BLEND}_l{l:g}" not in VARS]
# 11-05: both FedAvg updates at once, conv + a (ctx - conv) + b (ctxr - conv): ctx's mean lifts externals (EXP-145c), ctxr's
# mean repairs the library (EXP-145b). E145_MIX "a:b,..." -> variants mix_q{a}_r{b}.
MIX = [tuple(float(x) for x in ab.split(":")) for ab in os.environ.get("E145_MIX", "").split(",") if ab]
VARS += [f"mix_q{a:g}_r{b:g}" for a, b in MIX]
ADIR = REPO / ("results/analysis/new_loras_ctx" if SEQ == 0 else f"results/analysis/new_loras_ctx_s{SEQ}")   # per order: contexts differ
OLD, NEW = C.sequence(SEQ); NEW = NEW[:T]; K_ = len(OLD) + len(NEW)
CX = {v: {} for v in VARS}                                            # variant -> original external index -> in-context LIB index
for v in VARS:
    for i in NEW:
        e = E.LIB[i]; name = str(e["key"]).split(":", 1)[1]; path = ADIR / v / name / "adapter_model.safetensors"
        U.EXTPATH[f"np16{v}:" + name] = str(path)
        CX[v][i] = len(E.LIB); E.LIB.append(dict(e, key=f"np16{v}:" + name, tid={"ctxr": "r", "ctx": "q"}.get(v, "g" if v.startswith("ctx_") else "h" if v.startswith("mix_") else "f") + e["tid"][1:], path=str(path)))
def tag(v): return f"c145s{SEQ}t{T}-{v}_w{WN:g}"

def prep():
    """context spec per external (the other 23 members with their merge weights, init = its converted adapter) + replay file"""
    ADIR.mkdir(parents=True, exist_ok=True)
    for i in NEW:
        mem = [(E.LIB[j]["path"], 1.0 / K_) for j in OLD] + [(E.LIB[S.CV[l]]["path"], WN) for l in NEW if l != i]
        assert all(pathlib.Path(p).exists() for p, _ in mem) and pathlib.Path(E.LIB[S.CV[i]]["path"]).exists()
        json.dump({"members": mem, "w": WN, "init_adapter": E.LIB[S.CV[i]]["path"]}, open(ADIR / f"ctx_{E.LIB[i]['tid']}.json", "w"), indent=1)
    rp = ADIR / f"replay_s{SEQ}.txt"; rp.write_text("".join(f"libero_90_no_noops {E.LIB[j]['scene']}\n" for j in OLD))
    for j in OLD: assert (pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5/libero_90_no_noops") / f"{E.LIB[j]['scene']}_demo.hdf5").exists(), j
    print(f"[e145] prep: {len(NEW)} context specs ({K_ - 1} members each), replay {len(OLD)} library tasks", flush=True)

def train():
    # shared claim queue across GPUs (as EXP-144): variants in VARS order, externals in arrival order
    while True:
        left = [(v, i) for v in VARS if v in ("ctxr", "ctx") for i in NEW if not pathlib.Path(E.LIB[CX[v][i]]["path"]).exists()
                and not (ADIR / v / f".claim_{E.LIB[i]['tid']}").exists()]
        if not left: break
        v, i = left[0]; (ADIR / v).mkdir(parents=True, exist_ok=True)
        try: os.close(os.open(ADIR / v / f".claim_{E.LIB[i]['tid']}", os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        e = E.LIB[i]; out = pathlib.Path(E.LIB[CX[v][i]]["path"]).parent; t1 = time.time()
        log = REPO / f"results/merge_logs/e145_{v}_{e['tid']}{'' if SEQ == 0 else f'_s{SEQ}'}.log"
        cmd = [str(REPO / ".venv/bin/python"), str(REPO / "sweep/train_lora.py"), "--suite", e["suite"] + "_no_noops",
               "--task", e["scene"], "--steps", str(STEPS), "--batch", "8", "--device", "cuda:0", "--init", "ctx",
               "--context", str(ADIR / f"ctx_{e['tid']}.json"), "--rollouts", S.pattern(i), "--out", str(out)]
        if v == "ctxr": cmd += ["--replay_tasks", str(ADIR / f"replay_s{SEQ}.txt"), "--replay_frac", "0.5", "--replay_demos", "5"]
        r = subprocess.run(cmd, cwd=REPO, stdout=open(log, "w"), stderr=subprocess.STDOUT)
        print(f"[e145] trained {v} {e['tid']} -> exit {r.returncode} in {(time.time()-t1)/60:.0f} min", flush=True)

def interp():
    import safetensors.torch as st
    for l in LAMS:
        v = f"{BLEND}_l{l:g}"
        for i in NEW:
            dst = pathlib.Path(E.LIB[CX[v][i]]["path"])
            if dst.exists(): continue
            a, b = (pathlib.Path(E.LIB[x]["path"]) for x in (S.CV[i], CX[BLEND][i]))
            A, B = st.load_file(str(a)), st.load_file(str(b)); assert set(A) == set(B)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(a.parent / "adapter_config.json", dst.parent / "adapter_config.json")
            st.save_file({k: (A[k].float() + l * (B[k].float() - A[k].float())).to(A[k].dtype) for k in A}, str(dst))
        print(f"[e145] interp {v}: {len(NEW)} adapters", flush=True)
    for a_, b_ in MIX:
        v = f"mix_q{a_:g}_r{b_:g}"
        for i in NEW:
            dst = pathlib.Path(E.LIB[CX[v][i]]["path"])
            if dst.exists(): continue
            c, q, r = (pathlib.Path(E.LIB[x]["path"]) for x in (S.CV[i], CX["ctx"][i], CX["ctxr"][i]))
            C_, Q, R = (st.load_file(str(x)) for x in (c, q, r)); assert set(C_) == set(Q) == set(R)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(c.parent / "adapter_config.json", dst.parent / "adapter_config.json")
            st.save_file({k: (C_[k].float() + a_ * (Q[k].float() - C_[k].float()) + b_ * (R[k].float() - C_[k].float())).to(C_[k].dtype)
                          for k in C_}, str(dst))
        print(f"[e145] interp {v}: {len(NEW)} adapters", flush=True)

def run():
    for v in VARS:
        newc = [CX[v][i] for i in NEW]; mem = OLD + newc; ev = sorted(mem); TAG = tag(v)
        if not all(pathlib.Path(E.LIB[c]["path"]).exists() for c in newc): print(f"[e145] run {TAG}: adapters missing", flush=True); continue
        js = [(nm, su, scs) for nm, su, scs, ch in E.member_jobs(TAG, len(mem), "test", SEQ, ev) if any(E.rec(nm, sc) is None for sc in scs)]
        print(f"[e145] run {TAG}: {len(js)} jobs", flush=True)
        if not js: continue
        out = E.PRE / f"e145_{TAG}"; t1 = time.time()
        RT.RBuilder().build("priv", {"w": WN, "P": newc}, None, mem, out)
        print(f"[e145] built {TAG} in {time.time()-t1:.0f}s", flush=True)
        S.orch([{"name": nm, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                 "eval_scenes": scs, "suite": su, "norm_stats": U.FLEET_NORM} for nm, su, scs in js], "e145run")
        shutil.rmtree(out, ignore_errors=True)

def report():
    from scipy import stats
    sg = X.single(); L = [f"=== EXP-145 order {SEQ}, t={T}: converted externals trained inside the merge (w {WN:g}) ==="]
    ref = {"scratch ours (1/k)": (f"c134zs{SEQ}t{T}-ours", NEW), "WP on demos (0.25)": (f"c134ws{SEQ}t{T}-ours_w0.25", W.sequence(SEQ)[1][:T]),
           "converted (0.25)": (S.tag(SEQ, WN), [S.CV[i] for i in NEW]), **{f"in-context {v}": (tag(v), [CX[v][i] for i in NEW]) for v in VARS}}
    res = {}
    for nm, (tg, new) in ref.items():
        sc = E.member_success(tg, len(OLD) + len(new), "test", SEQ, sorted(OLD + new))
        if not sc: continue
        res[nm] = ({i: sc[i] / sg[i] for i in OLD if i in sc and sg.get(i)}, {o: sc[c] / sg[o] for o, c in zip(NEW, new) if c in sc and sg.get(o)})
    L.append("skill      own(scratch)  |  retention vs own: " + " / ".join(res))
    for i in NEW:
        L.append(f"{E.LIB[i]['tid']:9s}  {100*sg.get(i, 0):5.0f}%  |  " + " / ".join("  —  " if i not in res[k][1] else f"{100*res[k][1][i]:5.0f}%" for k in res))
    base = res.get("converted (0.25)")
    for k in res:
        lib, nw = res[k]; s = ""
        if base and k.startswith("in-context"):
            for g, a, b in (("ext", nw, base[1]), ("lib", lib, base[0])):
                ks = sorted(set(a) & set(b)); d = [a[q] - b[q] for q in ks]
                p = stats.wilcoxon(d).pvalue if any(abs(x) > 1e-9 for x in d) else 1.0
                s += f"  {g} vs converted {100*np.mean(d):+.1f}pp ({sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}, p={p:.3f})"
        L.append(f"{k:22s} library {100*np.mean(list(lib.values())) if lib else float('nan'):5.1f}% (n={len(lib)})  externals "
                 f"{100*np.mean(list(nw.values())) if nw else float('nan'):5.1f}% (n={len(nw)})" + s)
    txt = "\n".join(L); print(txt); (REPO / f"results/analysis/exp145_report_s{SEQ}.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E145_STAGE", "report"); U._clear_orphaned_builds()
    {"prep": prep, "train": train, "interp": interp, "run": run, "report": report}[st_]()
