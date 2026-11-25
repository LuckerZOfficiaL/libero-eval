#!/usr/bin/env python3
"""EXP-119 — our shared part + OPCM on the task-specific parts (follow-up to EXP-118, 10-16).

EXP-116/118: our merge (LIBERO-90 shared part S16 at full strength + residuals at 1/k) solves forgetting but new tasks stay
~20%; OPCM is the only method that learns new (anchored) tasks (57.5% after 4, 39.1% after 8) but keeps LIBERO-90 at 71%.
OPCM keeps only each new task's directions outside the merged model's dominant subspace and rescales the merge to one task's
size (factor ~0.3 vs 1/k). Hypothesis: stability comes from the shared part, plasticity from OPCM's treatment of the
task-specific parts -> combine them.

PRE-REGISTERED (10-16, before any rollout):
  arms       ours_opcm: W = base + S16 + lam * OPCM(r_1..r_k)   (OPCM alpha 0.5 = official default, arrival order, applied to
                        the two-sided residuals w.r.t. S16; lam = mean_i ||r_i|| / ||OPCM(r)||, the official norm rule)
             ours_norm: W = base + S16 + lam * sum_i r_i, lam = mean_i ||r_i|| / ||sum_i r_i||   (control: OPCM's scaling
                        without its projection -> separates "projection" from "larger weight")
  conditions new tasks from scratch (EXP-116 sequences) and anchored (EXP-118 sequences); t = 4, 8; same evaluated tasks,
             10 episodes, states 0..9. No tuning (alpha and the scale rule are OPCM's defaults).
  readout    old / new retention vs ours_lib and opcm (EXP-116 / EXP-118 records); primary = ours_opcm vs ours_lib and vs opcm at
             t = 8, old and new, paired over task instances (Wilcoxon), per condition
  predictions anchored: ours_opcm keeps LIBERO-90 >= 80% (shared part) and lifts new tasks well above ours_lib (toward OPCM's
             39%); if ours_norm matches it, the gain is the larger weight, not the projection. Scratch: smaller gains
             (EXP-117: foreign adapters interfere when weighted up).
Env: E119_STAGE run|report; E113_GPUS / E113_DEV / E113_ROUND (via exp113).
"""
import os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp116_continual as C
import exp118_anchored as A
import exp113_controls as X
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
ARMS, TS = ("ours_opcm", "ours_norm"), (4, 8)
COND = {"z": ("c116", C.sequence), "a": ("c118", A.sequence)}          # scratch / anchored: reference prefix, sequences

class HBuilder(C.CBuilder):
    def _active(self, members, s_):
        return any(s_ in self.fac(i) and float(self.fac(i)[s_][0].abs().max()) > 0 for i in members)
    def _rfac(self, members, s_):
        """factors of the two-sided residual r_i = D_i - Un Un^T D_i V V^T  (Br @ Ar == r_i, rank <= 32)"""
        Us = U.USIG[s_].to(E.DEV).float(); Un = Us / torch.linalg.norm(Us, dim=0).clamp(min=1e-12); V = U.VS[s_].to(E.DEV).float()
        out = []
        for i in members:
            B, A_ = (x.to(E.DEV).float() for x in self.fac(i)[s_])
            out.append((torch.cat([B, -Un @ (Un.T @ B)], 1), torch.cat([A_, (A_ @ V) @ V.T], 0)))
        return out
    def _rmerge(self, m, members, s_):
        F = self._rfac(members, s_); Bs, As = [f[0] for f in F], [f[1] for f in F]
        return (MM.opcm_unnorm(Bs, As, 0.5) if m == "ours_opcm" else MM.ta(Bs, As, 1.0)), F
    def lam(self, m, members):
        key = f"{m}|{','.join(map(str, members))}"
        if key not in self.norms:
            nr, nt = 0.0, np.zeros(len(members))
            with torch.no_grad():
                for s_ in self.ck:
                    if not self._active(members, s_): continue
                    R, F = self._rmerge(m, members, s_); nr += float(R.norm() ** 2)
                    for q, (b, a) in enumerate(F): nt[q] += float(((b.T @ b) * (a @ a.T).T).sum())
            self.norms[key] = float(np.mean(np.sqrt(nt)) / max(np.sqrt(nr), 1e-30))
        return self.norms[key]
    def _delta(self, m, p, members, s_):
        if m not in ARMS: return super()._delta(m, p, members, s_)
        if not self._active(members, s_): return None
        R, _ = self._rmerge(m, members, s_)
        return U.USIG[s_].to(E.DEV).float() @ U.VS[s_].to(E.DEV).float().T + p["lam"] * R

def items():
    out = []
    for cd, (_, seqf) in COND.items():
        for s in range(C.NSEQ):
            old, new = seqf(s)
            for t in TS:
                mem = old + new[:t]; ev = sorted(old[:C.NEVAL] + new[:t])
                for m in ARMS: out.append((f"c119{cd}s{s}t{t}-{m}", s, t, m, mem, ev))
    return out

def run():
    todo = [it for it in items() if any(E.rec(nm, sc) is None for nm, su, scs, ch in E.member_jobs(it[0], len(it[4]), "test", it[1], it[5]) for sc in scs)]
    print(f"[e119] {len(todo)} merged models", flush=True)
    if not todo: return
    B = HBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [todo[r:r + R] for r in range(0, len(todo), R)]
    def build_round(rd):
        js = []
        for tg, s, t, m, mem, ev in rd:
            out = E.PRE / f"e119_{tg}"; t1 = time.time(); lam = B.lam(m, mem); B.build(m, {"lam": lam}, None, mem, out)
            for name, su, scenes, ch in E.member_jobs(tg, len(mem), "test", s, ev):
                if all(E.rec(name, sc) is not None for sc in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e119] built {tg} ({len(mem)} tasks, lam {lam:.3f}) in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0])
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e119_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e119] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    def pool(pref, seqf, t, m, which):
        o = {}
        for s in range(C.NSEQ):
            a = A.inst(pref, seqf, s, t, m, which)
            if a is None: return None
            o.update(a)
        return o
    L = ["=== EXP-119 our shared part + OPCM on the residuals (ours_opcm) / + norm-rescaled residual sum (ours_norm) ===",
         f"{'cond':8s} {'t':>2} {'method':10s} {'old ret':>8s} {'new ret':>8s}"]
    R = {}
    for cd, (pref, seqf) in COND.items():
        for t in TS:
            for m, pf in (("ours_lib", pref), ("opcm", pref), ("ours_opcm", f"c119{cd}"), ("ours_norm", f"c119{cd}")):
                R[(cd, t, m)] = (pool(pf, seqf, t, m, "old"), pool(pf, seqf, t, m, "new"))
                f = lambda d: f"{100*np.mean(list(d.values())):7.1f}%" if d else " incompl"
                L.append(f"{'scratch' if cd == 'z' else 'anchored':8s} {t:>2} {m:10s} {f(R[(cd, t, m)][0])} {f(R[(cd, t, m)][1])}")
    L.append("primary (t = 8), paired over task instances")
    for cd in COND:
        for a_, b_ in (("ours_opcm", "ours_lib"), ("ours_opcm", "opcm"), ("ours_opcm", "ours_norm")):
            for j, nm in ((0, "old"), (1, "new")):
                x, y = R[(cd, 8, a_)][j], R[(cd, 8, b_)][j]
                if not x or not y: continue
                ks = [q for q in x if q in y]; d = np.array([100 * (x[q] - y[q]) for q in ks]); p = stats.wilcoxon(d).pvalue if np.any(d) else float("nan")
                L.append(f"  {'scratch' if cd == 'z' else 'anchored':8s} {nm}: {a_} - {b_:9s} {d.mean():+6.1f}pp  better {int((d>0).sum())} / worse {int((d<0).sum())}  p={p:.4f}")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp119_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E119_STAGE", "run"); U._clear_orphaned_builds()
    run() if st == "run" else report()
    print(f"[e119] {st} DONE", flush=True)
