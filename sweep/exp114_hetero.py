#!/usr/bin/env python3
"""EXP-114 — the shared part S16 estimated from the heterogeneous pool (76 fleet + 30 zero-init external = 106 adapters)
instead of the native library (76 fleet), and its effect on our merge.

PRE-REGISTERED (10-14, before any rollout):
  stage subspace: per adapted module, top-16 SVD of the pool MEAN of the deltas (core space), for (a) the 76 fleet
                  adapters -- must reproduce the stored S16 (unified_pool_driver USIG/VS) -- and (b) the 106-adapter pool;
                  report norm ratio ||S16_h|| / ||S16_f|| and subspace overlap; save results/analysis/exp114_s16_hetero.npz
  stage run:      EXP-113 condition B subsets (12, k=8, 4 fleet + 4 zero-init external); arm ours_h = S16_h + (1/8) * sum
                  of two-sided residuals w.r.t. S16_h; 10 eps per member on states 0..9
  stage report:   paired with EXP-113 B (ours with the fleet S16; avg). Prediction: avg < ours_h < ours.
Env: E114_STAGE subspace|run|report, E114_DEV, E113_* for the run (GPUs, rounds).
"""
import json, os, pathlib, shutil, subprocess, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import exp113_controls as X                                          # extends the library (fleet, external, S16-init)
import exp112_bench as E
import merge_methods as MM
import unified_pool_driver as U
from la_pair_features import load
OUT = REPO / "results/analysis/exp114_s16_hetero.npz"
DEV = os.environ.get("E114_DEV", "cuda:0")

def subspace():
    t0 = time.time()
    fleet = [load(t) for t in range(76)]                              # the 76 adapters the stored S16 was built from
    ext = [U.fold(e["key"]) for e in E.EXT]                           # 30 zero-init externals
    stems = [s for s in U.stems if s in U.USIG and all(s in f for f in fleet + ext)]
    res, save = {"ratio": [], "overlapU": [], "overlapV": [], "repro": []}, {}
    def top16(F, s_):
        B = torch.cat([f[s_][0] for f in F], 1).to(DEV).double() / len(F); A = torch.cat([f[s_][1] for f in F], 0).to(DEV).double()
        Qb, Rb = torch.linalg.qr(B); Qa, Ra = torch.linalg.qr(A.T)
        u, s, vh = torch.linalg.svd(Rb @ Ra.T, full_matrices=False)
        return (Qb @ u[:, :16]) * s[:16], Qa @ vh[:16].T
    for n, s_ in enumerate(stems):
        if float(fleet[0][s_][0].abs().max()) == 0 and all(float(f[s_][0].abs().max()) == 0 for f in fleet): continue
        Uf, Vf = top16(fleet, s_); Uh, Vh = top16(fleet + ext, s_)
        Sf = (Uf @ Vf.T); Sh = (Uh @ Vh.T); So = (U.USIG[s_].to(DEV).double() @ U.VS[s_].to(DEV).double().T)
        res["repro"].append(float((Sf - So).norm() / So.norm()))
        res["ratio"].append(float(Sh.norm() / Sf.norm()))
        un = lambda M: M / torch.linalg.norm(M, dim=0).clamp(min=1e-30)
        res["overlapU"].append(float((un(Uf).T @ un(Uh)).pow(2).sum() / 16)); res["overlapV"].append(float((Vf.T @ Vh).pow(2).sum() / 16))
        save[f"U|{s_}"] = Uh.float().cpu().numpy(); save[f"V|{s_}"] = Vh.float().cpu().numpy()
        if n % 50 == 0: print(f"[e114] {n+1}/{len(stems)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(OUT, **save)
    q = lambda v: f"median {np.median(v):.4f}, range {np.min(v):.4f}-{np.max(v):.4f}"
    L = [f"=== EXP-114 subspace: S16 from 106 adapters (76 fleet + 30 zero-init external) vs from the 76 fleet adapters; {len(res['ratio'])} modules ===",
         f"reproduction of the stored fleet S16 by this code (relative error): {q(res['repro'])}",
         f"norm ratio ||S16_hetero|| / ||S16_fleet||: {q(res['ratio'])}   (76/106 = {76/106:.3f})",
         f"subspace overlap (mean squared cosine of the 16 directions), output side: {q(res['overlapU'])}; input side: {q(res['overlapV'])}"]
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp114_subspace.txt").write_text(txt + "\n")

class HBuilder(E.Builder):
    def __init__(self):
        super().__init__(); z = np.load(OUT); self.H = {k_: torch.from_numpy(z[k_]) for k_ in z.files}
    def _delta(self, m, p, members, s_):
        if m != "ours_h": return super()._delta(m, p, members, s_)
        F = [self.fac(i) for i in members]
        if not any(s_ in f and float(f[s_][0].abs().max()) > 0 for f in F) or f"U|{s_}" not in self.H: return None
        Bs = [f[s_][0].to(E.DEV) for f in F]; As = [f[s_][1].to(E.DEV) for f in F]
        return MM.ours(Bs, As, p["alpha"], self.H[f"U|{s_}"].to(E.DEV), self.H[f"V|{s_}"].to(E.DEV))

ARM = ("ours_h", {"alpha": 1 / 8}, None)
def tg(): return f"c114B-{E.tag(*ARM)}"

def run():
    S_list = X.subsets("B")
    items = [(j, S) for j, S in enumerate(S_list) if any(E.rec(nm, s) is None for nm, su, scs, ch in E.member_jobs(tg(), 8, "test", j, S) for s in scs)]
    print(f"[e114] {len(items)} merged models (condition B, ours_h)", flush=True)
    B = HBuilder(); R = int(os.environ.get("E113_ROUND", "3")); rounds = [items[r:r + R] for r in range(0, len(items), R)]
    def build_round(rd):
        js = []
        for j, S in rd:
            out = E.PRE / f"e114_{j}"; t1 = time.time(); B.build(*ARM, S, out)
            for name, su, scenes, ch in E.member_jobs(tg(), 8, "test", j, S):
                if all(E.rec(name, s) is not None for s in scenes): continue
                js.append({"name": name, "method": "average", "params": {}, "adapters": [], "adapter_paths": [], "prebuilt_dir": str(out),
                           "eval_scenes": scenes, "suite": su, "norm_stats": U.FLEET_NORM})
            print(f"[e114] built subset {j} in {time.time()-t1:.0f}s", flush=True)
        return js
    cur = build_round(rounds[0]) if rounds else []
    for r in range(len(rounds)):
        proc = subprocess.Popen([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(E.plan(cur, 10, 0, f"e114_{r}"))], cwd=REPO)
        nxt = build_round(rounds[r + 1]) if r + 1 < len(rounds) else None
        proc.wait()
        for d in {j["prebuilt_dir"] for j in cur}: shutil.rmtree(d, ignore_errors=True)
        print(f"[e114] round {r + 1}/{len(rounds)} done", flush=True)
        cur = nxt

def report():
    from scipy import stats
    sg = X.single(); rows = []
    for j, S in enumerate(X.subsets("B")):
        def ret(tag_, which="all"):
            s = E.member_success(tag_, 8, "test", j, S)
            v = [s[i] / sg[i] for i in S if i in s and sg.get(i, 0) > 0 and (which == "all" or (which == "fleet") == (i < X.NF))]
            return np.mean(v) if v else np.nan
        rows.append([ret(f"c113B-{E.tag('avg', {}, None)}"), ret(f"c113B-{E.tag('ours', {'alpha': 1/8}, None)}"), ret(tg()),
                     ret(f"c113B-{E.tag('ours', {'alpha': 1/8}, None)}", "fleet"), ret(tg(), "fleet")])
    a = np.array(rows) * 100
    L = ["=== EXP-114: condition B (12 subsets, 4 fleet + 4 zero-init external, k=8), retention ===",
         f"avg {np.nanmean(a[:,0]):.1f}%   ours (fleet S16) {np.nanmean(a[:,1]):.1f}%   ours_h (heterogeneous-pool S16) {np.nanmean(a[:,2]):.1f}%"]
    for nm, d in (("ours_h - avg", a[:, 2] - a[:, 0]), ("ours_h - ours", a[:, 2] - a[:, 1])):
        ok = np.isfinite(d); L.append(f"{nm}: {np.nanmean(d):+.1f}pp (wins {int((d[ok]>0).sum())}/{ok.sum()}, Wilcoxon p={stats.wilcoxon(d[ok]).pvalue:.3f})")
    L.append(f"fleet members: ours {np.nanmean(a[:,3]):.1f}% vs ours_h {np.nanmean(a[:,4]):.1f}%")
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp114_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    st = os.environ.get("E114_STAGE", "subspace")
    {"subspace": subspace, "run": run, "report": report}[st]()
    print(f"[e114] {st} DONE", flush=True)
