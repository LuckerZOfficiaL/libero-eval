#!/usr/bin/env python3
"""EXP-140 — soft-gated external skills in the merge (10-25, after EXP-138: soft gating passed the offline gate).

EXP-138: an external (from-scratch) update fires on every task's inputs as strongly as on its own; the soft-gated read
factor R' = R (G_e + lam I)(G_e + G_others + lam I)^-1 (fit on input activations of all 24 skills in the order's merge)
keeps 57% of its own output energy (error 0.14) while cutting what reaches library / other-external inputs to 9% / 13%.
If the gating holds in closed loop, the externals can be merged at a much higher weight without breaking the rest.
Arms (order s in 0..2, t = 8, members = 16 library + 8 external, EXP-116 sequences), library always as in ours
(S + (1/24) sum of library residuals):
  soft_k      externals soft-gated at 1/24 (same weight as ours: the gating effect alone)
  soft_w0.25  externals soft-gated at 0.25
  soft_w1     externals soft-gated at 1
  plain_w0.25 externals ungated (their residual, as in ours) at 0.25 (control: weight without gating)
Reference: ours = EXP-134 c134zs{s}t8-ours (same orders / skills / states). Evaluated on all 24 skills, 10 eps, states 0..9.
PRE-REGISTERED (10-25, before any rollout): (1) soft_k: library >= ours - 2pp, externals within 5pp of ours; (2) soft_w0.25:
externals >= ours + 15pp with library >= ours - 5pp; (3) plain_w0.25: library <= ours - 15pp; (4) soft_w1 exploratory.
Method GO if soft_w0.25 or soft_w1 has externals >= ours + 15pp with library >= ours - 5pp (paired over order x skill).
12 models, 288 evaluations. Env: E140_STAGE run|report; E140_SHARD k/n; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
import safetensors.torch as st
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X
import exp116_continual as C
import exp112_bench as E
import unified_pool_driver as U
MM = E.MM
SEQS, T = (0, 1, 2), 8
ARMS = [("soft_k", "soft", None), ("soft_w0.25", "soft", 0.25), ("soft_w1", "soft", 1.0), ("plain_w0.25", "plain", 0.25)]

def items():
    out = []
    for nm, kind, w in ARMS:                                           # arm-major: each GPU shard gets every arm
        for s in SEQS:
            old, new = C.sequence(s); mem = old + new
            out.append((f"c140s{s}t{T}-{nm}", s, kind, w, old, new, sorted(mem)))
    return out

class GBuilder(E.Builder):
    def __init__(self):
        super().__init__(); self.SV = {}
    def sv(self, s):
        if s not in self.SV:
            self.SV[s] = {}
            for f in sorted((REPO / f"results/analysis/exp138_save_s{s}").glob("pass*.pt")): self.SV[s].update(torch.load(f))
            print(f"[e140] EXP-138 saves order {s}: {len(self.SV[s])} layers", flush=True)
        return self.SV[s]
    def delta(self, s, kind, w, old, new, s_):
        Usig, Vs = U.USIG[s_].to(E.DEV), U.VS[s_].to(E.DEV); k = len(old) + len(new)
        Fo = [self.fac(i) for i in old if s_ in self.fac(i)]
        D = MM.ours([f[s_][0].to(E.DEV) for f in Fo], [f[s_][1].to(E.DEV) for f in Fo], 1.0 / k, Usig, Vs) if Fo else Usig @ Vs.T
        Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); sv = self.sv(s).get(s_)
        for e in new:
            f = self.fac(e).get(s_)
            if f is None: continue
            b, a = f[0].to(E.DEV).float(), f[1].to(E.DEV).float()
            if kind == "plain" or sv is None or e not in sv["same"]: De = MM._resid_sum([b], [a], Un, Vs)[1]   # as ours
            else:
                Rs = sv["R_soft1"][e].to(E.DEV).float(); De = b @ Rs if sv["same"][e] else Rs.T @ a
            D = D + (1.0 / k if w is None else w) * De
        return D
    def build_arm(self, it, out):
        tg, s, kind, w, old, new, ev = it; sd = dict(self.base_sd)
        with torch.no_grad():
            for s_, ck in self.ck.items():
                D = self.delta(s, kind, w, old, new, s_); W = self.base_sd[ck]; D = D.float(); D = D if D.shape == W.shape else D.T
                sd[ck] = (W.to(E.DEV).float() + D).to(W.dtype).cpu()
        shutil.rmtree(out, ignore_errors=True); (out / "assets/physical-intelligence/libero").mkdir(parents=True)
        st.save_file(sd, str(out / "model.safetensors"))
        shutil.copy2(E.BASE_DIR / "config.json", out / "config.json")
        shutil.copy2(REPO / "results/analysis/norm_assets/norm_stats.json", out / "assets/physical-intelligence/libero/norm_stats.json")
        torch.cuda.empty_cache()

def run():
    allit = items()
    todo = [it for it in allit if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[6]) for sc in scs)]
    if os.environ.get("E140_SHARD"):
        k_, n_ = (int(x) for x in os.environ["E140_SHARD"].split("/")); todo = [it for it in todo if allit.index(it) % n_ == k_]
    print(f"[e140] {len(todo)} merged models, {sum(len(x[6]) for x in todo)} evaluations", flush=True)
    if not todo: return
    B = GBuilder(); R = int(os.environ.get("E113_ROUND", "1")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for it in rd:
            out = E.PRE / f"e140_{it[0]}"; t1 = time.time(); B.build_arm(it, out)
            for name, su, scenes, ch in E.member_jobs(it[0], len(it[6]), "test", it[1], it[6]):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e140] built {it[0]} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e140{os.environ.get('E140_SHARD', '').replace('/', 'of')}_{r}"))], cwd=REPO)
        try: nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        except Exception:                                                # never orphan a running eval nor leave its models
            proc.wait()
            for d in {j["prebuilt_dir"] for j in cur} | {str(E.PRE / f"e140_{it[0]}") for it in rounds[r + 1]}: shutil.rmtree(d, ignore_errors=True)
            raise
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e140] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def ret(tg, s, sg):
    old, new = C.sequence(s); ev = sorted(old + new); sc = E.member_success(tg, len(ev), "test", s, ev)
    if len(sc) < len(ev): return None
    return {("old" if i in old else "new", (s, i)): sc[i] / sg[i] for i in ev if sg.get(i, 0) > 0}

def report():
    from scipy import stats
    sg = X.single(); L = ["=== EXP-140 soft-gated externals in the merge (t=8, orders 0-2): retention library / external ==="]
    ref = {s: ret(f"c134zs{s}t8-ours", s, sg) for s in SEQS}
    m = lambda d, g: 100 * np.mean([v for (g_, _), v in d.items() if g_ == g])
    if all(ref.values()):
        r0 = {k: v for d in ref.values() for k, v in d.items()}; L.append(f"ours (EXP-134)  {m(r0, 'old'):5.1f} / {m(r0, 'new'):5.1f}")
    for nm, kind, w in ARMS:
        rr = [ret(f"c140s{s}t{T}-{nm}", s, sg) for s in SEQS]
        if any(r is None for r in rr) or not all(ref.values()): L.append(f"{nm:14s}  incomplete"); continue
        d = {k: v for r in rr for k, v in r.items()}; line = f"{nm:14s}  {m(d, 'old'):5.1f} / {m(d, 'new'):5.1f}   vs ours:"
        for g in ("old", "new"):
            ks = [k for k in d if k[0] == g and k in r0]; x = np.array([100 * (d[k] - r0[k]) for k in ks])
            line += f"  {'library' if g == 'old' else 'external'} {x.mean():+5.1f}pp (n={len(x)}, p={stats.wilcoxon(x).pvalue if np.any(x) else float('nan'):.2g})"
        L.append(line)
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp140_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st_ = os.environ.get("E140_STAGE", "run"); U._clear_orphaned_builds()
    run() if st_ == "run" else report()
    print(f"[e140] {st_} DONE", flush=True)
