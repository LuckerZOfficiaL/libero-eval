#!/usr/bin/env python3
"""EXP-161 — what does the correction learn? Weight-space decomposition of trained corrections (no simulation, 11-18).

The deployed model is W0 + S16 + (1/k) sum_i r_i + C: the pool's shared component S16 = U Sigma V^T (full strength), every
member's two-sided residual r_i = D_i - Un Un^T D_i V V^T at 1/k, and the correction C = B_c A_c (plain LoRA, scale 1).
Per stem, in factor form (no dense k x [out, in] tensors):
  shared   ||Un Un^T C V V^T||^2 / ||C||^2 (two-sided), plus the output-only / input-only shares
  residual alignment  rc = C - P(C) against R_new = sum_new r_i and R_lib = sum_lib r_j: cosines and the least-squares fit
           rc ~ a R_new + b R_lib (explained share of ||rc||^2); per-skill cosines cos(rc, r_i)
  capture  share of ||rc||^2 inside the column span of the new adapters' B's (resp. the library's), vs the random-subspace
           expectation dim / out
  spectrum effective rank of C (exp entropy of sigma^2), and the share of ||C||^2 per VLA part (vision / language / action
           expert / heads) next to the new adapters' own residual energy per part
Groups of members come from each run's ctx_all_zfree.json (library first, then the new skills; tasks_ext.txt = #new).
Output: results/analysis/exp161_spectral.json + a printed table. Env: E161_RUNS (name=adir:train_dir,...), E161_DEV.
"""
import json, os, pathlib, sys
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
A = REPO / "results/analysis"
DEV = os.environ.get("E161_DEV", "cpu")                              # factors of up to 83 adapters: CPU memory
RUNS = dict(r.split("=") for r in os.environ.get("E161_RUNS", ",".join([
    "o0=new_loras_cons:train_consfz", "o1=new_loras_cons_s1:train_consfz", "o2=new_loras_cons_s2:train_consfz",
    "o3=new_loras_cons_s3:train_consfz", "t22=new_loras_cons_s4:train_consfz", "lib74=new_loras_cons_s6:train_consfz6",
    "c14=new_loras_cons_s19:train_consfzc", "c18=new_loras_cons_s20:train_consfzc", "c22=new_loras_cons_s4:train_consfzc",
    "base_o0=new_loras_cons:train_consb"])).split(","))

def part(stem):
    return "vision" if "vision_tower" in stem else "expert" if "gemma_expert" in stem else "language" if "language_model" in stem else "heads"

def fac(path, key):
    U.EXTPATH[key] = str(path); f = U.fold(key)
    return {s: (b.to(DEV, torch.float32), a.to(DEV, torch.float32)) for s, (b, a) in f.items()}

def ip(x, y):   # <Bx Ax, By Ay>_F in factor form
    return float(((x[0].T @ y[0]) * (x[1] @ y[1].T)).sum())

def core(x, Un, V):   # 16 x 16 shared core: Un^T Bx Ax V
    return (Un.T @ x[0]) @ (x[1] @ V)

def run(name, adir, tdir):
    ctx = json.load(open(A / adir / "ctx_all_zfree.json")); t = sum(1 for _ in open(A / adir / "tasks_ext.txt"))
    mem = [p for p, _ in ctx["members"]]; lib, new = mem[:len(mem) - t], mem[len(mem) - t:]
    Cf = fac(A / adir / tdir / "adapter_model.safetensors", f"e161c:{name}")
    Fl = [fac(p, f"e161l:{i}") for i, p in enumerate(lib)]; Fn = [fac(p, f"e161n:{i}") for i, p in enumerate(new)]
    acc = {g: dict(C=0.0, Cs=0.0, Co=0.0, Ci=0.0, rc=0.0, Rn=0.0, Rl=0.0, rRn=0.0, rRl=0.0, RnRl=0.0, capn=0.0, capl=0.0,
                   dimn=0.0, diml=0.0, rnew=0.0, n=0) for g in ("vision", "language", "expert", "heads")}
    pcos_n = np.zeros(len(new)); pcos_l = np.zeros(len(lib)); pr = np.zeros(len(new)); prl = np.zeros(len(lib)); rcn = 0.0; eff = []
    for s in U.USIG:
        if s not in Cf: continue
        g = acc[part(s)]; C = Cf[s]
        Un = U.USIG[s].to(DEV, torch.float32); Un = Un / torch.linalg.norm(Un, dim=0).clamp(min=1e-12); V = U.VS[s].to(DEV, torch.float32)
        Sc = core(C, Un, V); cc = ip(C, C); g["C"] += cc; g["Cs"] += float((Sc ** 2).sum()); g["n"] += 1
        g["Co"] += float(((Un.T @ C[0]) @ C[1]).pow(2).sum()); g["Ci"] += float((C[0] @ (C[1] @ V)).pow(2).sum())
        def rip(x, y, Sx, Sy): return ip(x, y) - float((Sx * Sy).sum())   # <r_x, r_y>
        L = [(x[s], core(x[s], Un, V)) for x in Fl if s in x]; N = [(x[s], core(x[s], Un, V)) for x in Fn if s in x]
        rcc = cc - float((Sc ** 2).sum()); g["rc"] += rcc; rcn += rcc
        cn = [rip(C, x, Sc, Sx) for x, Sx in N]; cl = [rip(C, x, Sc, Sx) for x, Sx in L]
        g["rRn"] += sum(cn); g["rRl"] += sum(cl)
        g["Rn"] += sum(rip(x, y, Sx, Sy) for x, Sx in N for y, Sy in N); g["Rl"] += sum(rip(x, y, Sx, Sy) for x, Sx in L for y, Sy in L)
        g["RnRl"] += sum(rip(x, y, Sx, Sy) for x, Sx in N for y, Sy in L)
        nn = [rip(x, x, Sx, Sx) for x, Sx in N]; g["rnew"] += sum(nn)
        if len(N) == len(new): pcos_n += np.array(cn); pr += np.array(nn)
        if len(L) == len(lib): pcos_l += np.array(cl); prl += np.array([rip(x, x, Sx, Sx) for x, Sx in L])
        # capture of rc by the column span of the new (resp. library) adapters' B, shared outputs removed
        for key, F_, dk in (("capn", N, "dimn"), ("capl", L, "diml")):
            if not F_: continue
            Bs = torch.cat([x[0] for x, _ in F_], 1); Bs = Bs - Un @ (Un.T @ Bs); Q = torch.linalg.qr(Bs)[0]
            q = int((torch.linalg.svdvals(Bs) > 1e-6 * torch.linalg.svdvals(Bs).max()).sum()); Q = Q[:, :q]
            X = Q.T @ C[0]; Y = Q.T @ Un
            num = float(((X.T @ X) * (C[1] @ C[1].T)).sum()) + float(((Y @ Sc) ** 2).sum()) - 2 * float(((X.T @ Y @ Sc) * (C[1] @ V)).sum())
            g[key] += num; g[dk] += q / C[0].shape[0] * rcc
        qb, rb = torch.linalg.qr(C[0]); qa, ra = torch.linalg.qr(C[1].T); sv = torch.linalg.svdvals(rb @ ra.T) ** 2
        p_ = sv / sv.sum().clamp(min=1e-30); eff.append((float(torch.exp(-(p_ * torch.log(p_.clamp(min=1e-30))).sum())), float(sv.sum())))
    tot = {k: sum(acc[g][k] for g in acc) for k in acc["vision"]}
    def summary(d):
        G = np.array([[d["Rn"], d["RnRl"]], [d["RnRl"], d["Rl"]]]); b = np.array([d["rRn"], d["rRl"]])
        try: coef = np.linalg.solve(G, b); expl = float(coef @ b) / max(d["rc"], 1e-30)
        except np.linalg.LinAlgError: coef, expl = (np.nan, np.nan), np.nan
        return dict(C=d["C"], shared2=d["Cs"] / max(d["C"], 1e-30), shared_out=d["Co"] / max(d["C"], 1e-30), shared_in=d["Ci"] / max(d["C"], 1e-30),
                    cos_new=d["rRn"] / np.sqrt(max(d["rc"] * d["Rn"], 1e-30)), cos_lib=d["rRl"] / np.sqrt(max(d["rc"] * d["Rl"], 1e-30)),
                    a_new=float(coef[0]), b_lib=float(coef[1]), explained=expl,
                    cap_new=d["capn"] / max(d["rc"], 1e-30), cap_new_rand=d["dimn"] / max(d["rc"], 1e-30),
                    cap_lib=d["capl"] / max(d["rc"], 1e-30), cap_lib_rand=d["diml"] / max(d["rc"], 1e-30), rnew=d["rnew"])
    out = dict(name=name, k=len(mem), t=t, total=summary(tot), parts={g: summary(acc[g]) for g in acc},
               energy_C={g: acc[g]["C"] / max(tot["C"], 1e-30) for g in acc}, energy_rnew={g: acc[g]["rnew"] / max(tot["rnew"], 1e-30) for g in acc},
               cos_new_skill=list(pcos_n / np.sqrt(np.maximum(rcn * pr, 1e-30))), cos_lib_skill=list(pcos_l / np.sqrt(np.maximum(rcn * prl, 1e-30))),
               eff_rank=float(np.average([e for e, _ in eff], weights=[w for _, w in eff])))
    return out

if __name__ == "__main__":
    res = []
    for name, spec in RUNS.items():
        adir, tdir = spec.split(":")
        if not (A / adir / tdir / "adapter_model.safetensors").exists(): print(f"[e161] {name}: {adir}/{tdir} missing", flush=True); continue
        r = run(name, adir, tdir); res.append(r); T = r["total"]
        print(f"[e161] {name:8s} k={r['k']:2d} t={r['t']:2d} | shared {100*T['shared2']:4.1f}% (out {100*T['shared_out']:4.1f} / in {100*T['shared_in']:4.1f}) | "
              f"cos(rc, R_new) {T['cos_new']:+.3f}  cos(rc, R_lib) {T['cos_lib']:+.3f}  fit a {T['a_new']:+.3f} b {T['b_lib']:+.3f} expl {100*T['explained']:4.1f}% | "
              f"capture new {100*T['cap_new']:4.1f}% (rand {100*T['cap_new_rand']:4.1f}) lib {100*T['cap_lib']:4.1f}% (rand {100*T['cap_lib_rand']:4.1f}) | eff rank {r['eff_rank']:.1f}", flush=True)
        print(f"         energy C   " + "  ".join(f"{g} {100*v:4.1f}%" for g, v in r["energy_C"].items()) +
              f"   | new residuals " + "  ".join(f"{g} {100*v:4.1f}%" for g, v in r["energy_rnew"].items()), flush=True)
        print(f"         per part cos(rc,R_new) " + "  ".join(f"{g} {d['cos_new']:+.3f}" for g, d in r["parts"].items()) +
              f"   per-skill cos new {np.mean(r['cos_new_skill']):+.3f} lib {np.mean(r['cos_lib_skill']):+.3f}", flush=True)
    json.dump(res, open(A / "exp161_spectral.json", "w"), indent=1)
