#!/usr/bin/env python3
"""EXP-146 — CUSD baseline (Wang, Zou, Li, NeurIPS 2026, arXiv 2611.04108; user-flagged 11-04) in the scratch continual setting.

CUSD: per layer a shared input basis from the K task updates (HO-GSVD-derived), each direction scored by how unevenly the
tasks' energy spreads over it; low-dispersion directions = common, high = unique (KDE valley); merge = F(common parts) +
beta * sum of unique parts. CUSD+TA at beta = lambda is plain TA, so with F = averaging (lambda = 1/k) CUSD is averaging
with the task-dominated directions damped (beta < 1/k). Same family as ours (ours with an in-set shared part is also
averaging); ours differs by an EXTERNAL library anchor at full strength, two-sided, fixed across steps. Not spectral (keeps
the averaging rule), so unlike TSV / Iso-C / Iso-CTS it should not collapse at k = 24 -> the closest threat to the claim
that the library anchor is what protects the library.
Setting = EXP-134 scratch at t = 8: 16 library skills + 8 from-scratch externals (EXP-116 orders 0..3), all 24 evaluated,
10 eps; F = averaging (the strongest published rule on our VLA merges), beta = b / k.
  diag   (CPU ok) per module: directions in the joint row space, unique count, unique share of energy, and per skill the
         share of ITS energy CUSD calls unique (library vs externals) -> what beta damps, before any rollout
  val    order 0, validation states 25..34: scoring rel (paper: generative models) x b {0, 0.25, 0.5, 2}, abs x b 0.5
         (11-04, before any rollout, after diag: rel calls ~63% of each external's energy unique and ~1-2% of the library's;
         abs calls ~70% of every library skill's energy unique -> abs b 0 dropped; b 2 doubles the externals' private parts)
  pick   highest mean success over the 24 skills on val -> results/analysis/exp146_pick.json
  run    the picked setting, test states 0..9, orders E146_SEQS (default 0..3)
  report retention vs single-task (as EXP-134): library / externals / overall, vs ours c134zs{s}t8-ours and averaging
         c134zs{s}t8-avg, Wilcoxon over skills pooled across orders
PRE-REGISTERED (11-04, before any rollout): selection on val by mean success, never on test. CUSD threatens the claim if its
library retention is within 3pp of ours (pooled over 4 orders) at externals >= ours - 3pp. Expected (diag will tell): the
unique directions are mostly the externals', so small beta damps the externals; library between averaging and ours.
Env: E146_STAGE diag|val|pick|run|report; E146_SEQS; E146_CFG "mode,b" overrides the pick; E113_GPUS / E113_DEV / E113_ROUND.
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X                                           # sets E112_DEV / E112_GPUS from E113_*
import exp116_continual as C
import exp124_wpcontinual as W
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
SEQS = [int(x) for x in os.environ.get("E146_SEQS", "0,1,2,3").split(",")]
T = 8
VAL = [("rel", 0.0), ("rel", 0.25), ("rel", 0.5), ("rel", 2.0), ("abs", 0.5)]   # 11-04 after diag: abs b 0 -> rel b 2
PICK = REPO / "results/analysis/exp146_pick.json"
def tag(s, mode, b): return f"c146s{s}t{T}-cusd_{mode}_b{b:g}"
def members(s):
    old, new = C.sequence(s); return old, new[:T], old + new[:T]
def params(mode, b, k): return {"beta": b / k, "mode": mode, "F": "avg"}

class _Fac(E.Builder):
    """adapter factors only (no base model): for diag"""
    def __init__(self):
        self.AB = {}; self.AB[0] = f0 = E.load(0); self.ck = {s_: None for s_ in U.stems if s_ in f0}

def diag():
    B = _Fac(); dev = E.DEV
    for s in SEQS:
        old, new, mem = members(s); k = len(mem); t1 = time.time(); rows = []
        F = [B.fac(i) for i in mem]
        for mode in ("rel", "abs"):
            for s_ in B.ck:
                if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F): continue
                st_ = []; MM.cusd([f[s_][0].to(dev).float() for f in F], [f[s_][1].to(dev).float() for f in F], 0.0, mode=mode, stats=st_)
                rows.append(dict(st_[0], stem=s_, mode=mode))
        out = {"seq": s, "members": mem, "n_old": len(old), "rows": rows}
        json.dump(out, open(REPO / f"results/analysis/exp146_diag_s{s}.json", "w"))
        L = [f"=== EXP-146 diag order {s}: CUSD partition of the t={T} merge ({k} skills: {len(old)} library + {T} externals) ==="]
        for mode in ("rel", "abs"):
            R = [r for r in rows if r["mode"] == mode]; tu = np.array([r["task_unique"] for r in R])     # modules x skills
            e = np.array([r["task_energy"] for r in R]); w = e / e.sum(0, keepdims=True).clip(1e-30)        # energy-weighted
            lib, ext = (w * tu)[:, :len(old)].sum(0), (w * tu)[:, len(old):].sum(0)
            L.append(f"{mode}: modules {len(R)}, with a valley {sum(r['n_unique'] > 0 for r in R)}; directions/module {np.mean([r['R'] for r in R]):.0f}, "
                     f"unique/module {np.mean([r['n_unique'] for r in R]):.1f}; unique share of total energy {np.mean([r['unique_energy'] for r in R]):.3f}")
            L.append(f"   share of each skill's own energy called unique: library {lib.mean():.3f} (min {lib.min():.3f}, max {lib.max():.3f}) | "
                     f"externals {ext.mean():.3f} (min {ext.min():.3f}, max {ext.max():.3f})")
        L.append(f"({time.time()-t1:.0f}s)"); txt = "\n".join(L); print(txt, flush=True)
        (REPO / f"results/analysis/exp146_diag_s{s}.txt").write_text(txt + "\n")

def evaluate(todo, split, offset):
    """todo: [(tag, seq, params, ordered members, evaluated members)]; rounds as EXP-134 (next round built during eval)"""
    if not todo: return
    B = E.Builder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, p, mem, ev in rd:
            out = E.PRE / f"e146_{tg}_{split}"; t1 = time.time(); B.build("cusd", p, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), split, s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e146] built {tg} {split} ({len(mem)} skills) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, offset, f"e146{split}_{r}"))], cwd=REPO)
        try: nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        except Exception:                                                # never orphan a running eval (EXP-134 lesson)
            proc.wait()
            for d in {j["prebuilt_dir"] for j in cur} | {str(E.PRE / f"e146_{it[0]}_{split}") for it in rounds[r + 1]}: shutil.rmtree(d, ignore_errors=True)
            raise
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e146] {split} round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def missing(tg, k, split, s, ev): return any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(tg, k, split, s, ev) for sc in scs)

def val():
    old, new, mem = members(0); ev = sorted(mem)
    todo = [(tag(0, m, b), 0, params(m, b, len(mem)), mem, ev) for m, b in VAL if missing(tag(0, m, b), len(mem), "val", 0, ev)]
    print(f"[e146] val: {len(todo)} merged models", flush=True); evaluate(todo, "val", 25)

def val_scores():
    old, new, mem = members(0); ev = sorted(mem); out = {}
    for m, b in VAL:
        sc = E.member_success(tag(0, m, b), len(mem), "val", 0, ev)
        if len(sc) == len(ev): out[f"{m},{b:g}"] = (float(np.mean(list(sc.values()))), float(np.mean([sc[i] for i in old])), float(np.mean([sc[i] for i in new])))
    return out

def pick():
    sc = val_scores(); assert len(sc) == len(VAL), f"val incomplete: {sorted(sc)}"
    best = max(sc, key=lambda c: sc[c][0]); json.dump({"pick": best, "val": sc}, open(PICK, "w"), indent=1)
    print(f"[e146] pick {best} (val mean success {sc[best][0]:.3f}); " + ", ".join(f"{c} {v[0]:.3f}" for c, v in sc.items()), flush=True)

def cfg():
    c = os.environ.get("E146_CFG") or json.load(open(PICK))["pick"]; m, b = c.split(","); return m, float(b)

def run():
    m, b = cfg(); todo = []
    for s in SEQS:
        old, new, mem = members(s); ev = sorted(mem)
        if missing(tag(s, m, b), len(mem), "test", s, ev): todo.append((tag(s, m, b), s, params(m, b, len(mem)), mem, ev))
    print(f"[e146] run {m},{b:g}: {len(todo)} merged models", flush=True); evaluate(todo, "test", 0)

def report():
    from scipy import stats
    m, b = cfg(); sg = W.single(); L = [f"=== EXP-146 CUSD (+averaging, {m} scoring, beta = {b:g}/k) vs ours / averaging, scratch, t={T}, retention vs single-task ==="]
    sc_ = val_scores()
    if sc_: L.append("val (order 0, states 25..34, mean success all / library / externals): " + "; ".join(f"{c} {v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}" for c, v in sc_.items()))
    arms = {"averaging": lambda s: f"c134zs{s}t{T}-avg", "ours": lambda s: f"c134zs{s}t{T}-ours", "CUSD": lambda s: tag(s, m, b)}
    R = {a: {} for a in arms}
    for s in SEQS:
        old, new, mem = members(s); ev = sorted(mem)
        for a, f in arms.items():
            sc = E.member_success(f(s), len(mem), "test", s, ev)
            if len(sc) < len(ev): continue
            R[a].update({("lib" if i in old else "ext", s, i): sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0})
    L.append(f"{'arm':10s} " + " ".join(f"{'order ' + str(s):>15s}" for s in SEQS) + f" {'pooled':>15s}   (library % / externals %)")
    for a in arms:
        cells = []
        for s in SEQS + [None]:
            v = {k_: x for k_, x in R[a].items() if s is None or k_[1] == s}
            lib = [x for k_, x in v.items() if k_[0] == "lib"]; ext = [x for k_, x in v.items() if k_[0] == "ext"]
            cells.append(f"{100*np.mean(lib):5.1f} / {100*np.mean(ext):5.1f}" if lib and ext else "—")
        L.append(f"{a:10s} " + " ".join(f"{c:>15s}" for c in cells))
    for ref in ("ours", "averaging"):
        for g in ("lib", "ext"):
            ks = sorted(k_ for k_ in set(R["CUSD"]) & set(R[ref]) if k_[0] == g); d = [R["CUSD"][k_] - R[ref][k_] for k_ in ks]
            if not d: continue
            p = stats.wilcoxon(d).pvalue if any(abs(x) > 1e-9 for x in d) else 1.0
            L.append(f"CUSD - {ref:9s} {g}: {100*np.mean(d):+5.1f}pp (n={len(d)}, {sum(x > 1e-9 for x in d)} up / {sum(x < -1e-9 for x in d)} down, p={p:.3f})")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp146_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E146_STAGE", "report"); U._clear_orphaned_builds()
    {"diag": diag, "val": val, "pick": pick, "run": run, "report": report}[st]()
