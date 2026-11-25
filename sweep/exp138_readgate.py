#!/usr/bin/env python3
"""EXP-138 — offline gate (10-25): can an external (from-scratch, foreign-A) update be made to respond only to its own
task's inputs? (user 10-25: improving retention of external tasks is the most valuable next step; brainstorm direction 1
= read-gating, direction 2 = post-hoc refit into the library's read directions.)

Network: our training-free merge (ours_lib) of order E138_SEQ (scratch condition, EXP-116 sequences: 16 library + 8
external skills, t = 8), built in place on the base pi0.5. Per adapted Linear, input activations of all 24 skills from
E138_FRAMES demo frames each, own prompt; frames alternate fit / test. Up to E138_CAP tokens per skill, split and layer.
For every external skill e and layer, with read factor R (r x d_in) and write factor Wr (d_out x r) of e's update:
  energy on a token set H = mean over tokens of ||Wr R' h||^2, reported on e's own test tokens, on the 16 library skills'
  (mean) and on the 7 other externals' (mean); own error = mean ||Wr (R' - R) h||^2 / mean ||Wr R h||^2 on own test tokens.
  direction 1  hard{q}: R' = R (I - Q_q Q_q^T), Q_q = top-q eigenvectors of the other 23 skills' mean input second moment
               (fit tokens, each skill weighted equally); rand{q}: same with a random q-dim subspace (control);
               soft{mu}: R' = R (G_e + lam I)(G_e + mu G_others + lam I)^-1, lam = 0.1 tr(G_e)/d (input-weighted least
               squares pulled toward R, RegMean-like); gen{k}: R' = R Q Q^T, Q = top-k generalized eigendirections of
               (G_e, G_others) inside e's top-512 input span (selective read; added after the 10-25 smoke test)
  direction 2  refit_V: e's update refit to read only the library's 16 shared input directions (least squares on own fit
               tokens, Wr' = Wr R G_e V (V^T G_e V)^-1), test error on own test tokens; controls refit_rand (16 random
               directions) and refit_pca (top-16 directions of e's own inputs = best any 16-dim read can do)
  also: wshared = share of e's own output energy written into the library's 16 shared output directions; bias (added
  after the 10-25 smoke test, diagnostic only): share of own output that is a constant offset, its cosine with the
  offset on library inputs and with the library shared part's output on the same inputs.
Gate (pre-registered, energy-weighted over layers, mean over the 8 externals): direction 1 GO if a variant has own error
<= 0.3 with library and other-external energy kept <= 0.3 each while rand{q} at that q keeps >= 0.6 (MAYBE if <= 0.6);
direction 2 GO if refit_V error <= 0.3. Layer-output energy is a proxy: rollouts only after a GO.
Env: E138_STAGE run|report; E138_SEQ (0); E138_FRAMES (24); E138_CAP (1536); E138_DBUDGET (sum of d_in per collection
pass, 80000); E138_SMOKE (N first layers); E112_DEV (cuda device). Output: results/analysis/exp138_gate_s{SEQ}.json, exp138_report_s{SEQ}.txt.
"""
import json, math, os, pathlib, re, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
SEQ = int(os.environ.get("E138_SEQ", "0")); NFR = int(os.environ.get("E138_FRAMES", "24"))
CAP = int(os.environ.get("E138_CAP", "1536")); DBUDGET = int(os.environ.get("E138_DBUDGET", "80000"))
QS, MUS, GKS = (16, 64, 256, 1024), (1.0, 10.0, 100.0), (16, 64, 256)
SMOKE = os.environ.get("E138_SMOKE", "")                              # smoke test: first N layers, separate files
TAG = f"s{SEQ}" + ("_smoke" if SMOKE else "")
OUT = REPO / f"results/analysis/exp138_gate_{TAG}.json"; REP = REPO / f"results/analysis/exp138_report_{TAG}.txt"
SAVE = REPO / f"results/analysis/exp138_save_{TAG}"                  # per layer: Q (top-64 shared input directions, all 24
                                                                     # skills), each external's soft1 read factor + orientation
DEMO = pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5")
PROMPT = lambda e: re.sub(r"^.*?SCENE\d+_", "", e["scene"]).replace("_", " ")         # as the eval harness (EXP-112 fix)

def frames(E, e, load_demo_samples):
    suite = e["suite"] + "_no_noops"
    try: smp = load_demo_samples(suite, e["scene"], n_demos=4, stride=4)
    except FileNotFoundError:                                        # t57: frames of a same-scene sibling (as exp112_actstats)
        scn = re.match(r"(.*?SCENE\d+)_", e["scene"]).group(1)
        sib = next(x for x in E.LIB if x is not e and x["suite"] == e["suite"] and x["scene"].startswith(scn + "_")
                   and (DEMO / suite / f"{x['scene']}_demo.hdf5").exists())
        smp = load_demo_samples(suite, sib["scene"], n_demos=4, stride=4)
    ix = np.linspace(0, len(smp) - 1, min(NFR, len(smp))).round().astype(int)
    return [smp[j][0] for j in ix]

def orth(d, q, g):
    return torch.linalg.qr(torch.randn(d, q, generator=g).to(DEVICE))[0]

def analyse(s_, d, H, mem, new, fac, Usig, Vs, wshape):
    """H[(task, split)] -> (n, d) fp16 tokens. Returns one record per external with a factor on this layer."""
    test = {j: H[(j, "test")].float() for j in mem}
    def gm(j):                                                       # skill j's mean input second moment (d x d), fit tokens
        x = H[(j, "fit")].float(); return x.T @ x / x.shape[0]
    Glib = torch.zeros(d, d, device=DEVICE)                          # one d x d at a time (d up to 16384 -> 1 GB each)
    for j in mem:
        if j not in new: Glib += gm(j)
    Gall = Glib.clone()                                              # all 24 skills, each weighted equally
    for j in new: Gall += gm(j)
    save = dict(Q=torch.svd_lowrank(Gall / len(mem), q=min(64, d // 2), niter=4)[0].half().cpu(), same={}, R_soft1={})
    del Gall                                                         # Q: the inputs' shared dominant directions (offset carrier)
    g = torch.Generator().manual_seed(13800 + d); recs = []
    for e in new:
        f = fac(e).get(s_)
        if f is None or float(f[0].abs().max()) == 0: continue
        b, a = f[0].to(DEVICE).float(), f[1].to(DEVICE).float()
        same = tuple((b @ a).shape) == tuple(wshape)                 # D = b a in weight (out, in) orientation?
        save["same"][int(e)] = bool(same)
        R, Wr = (a, b) if same else (b.T, a.T)                         # read (r x d_in), write (d_out x r)
        Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12)
        Vin = torch.linalg.qr(Vs if same else Un)[0]; Uout = torch.linalg.qr(Un if same else Vs)[0]
        Ge = gm(e); Go = Glib.clone()                                 # others summed directly: total - own cancels in fp32
        for j in new:
            if j != e: Go += gm(j)
        Go /= len(mem) - 1; MM = Wr.T @ Wr
        def energy(Rp, H_):
            Z = H_ @ Rp.T; return float(((Z @ MM) * Z).sum() / H_.shape[0])
        def evalR(Rp):
            own = energy(Rp, test[e]); Zd = test[e] @ (Rp - R).T
            lib = np.mean([energy(Rp, test[j]) for j in mem if j not in new])
            ext = np.mean([energy(Rp, test[j]) for j in new if j != e])
            return [own, float(lib), float(ext), float(((Zd @ MM) * Zd).sum() / test[e].shape[0])]
        qmax = min(max(QS), d // 2); Qo, So, _ = torch.svd_lowrank(Go, q=qmax, niter=4); tr = float(torch.trace(Go))
        V = {"base": evalR(R)}; cap = {}
        for q in QS:
            if q > qmax: continue
            Q = Qo[:, :q]; V[f"hard{q}"] = evalR(R - (R @ Q) @ Q.T); cap[q] = float(So[:q].sum()) / tr
            Qr = orth(d, q, g); V[f"rand{q}"] = evalR(R - (R @ Qr) @ Qr.T)
        lam = 0.1 * float(torch.trace(Ge)) / d                         # pull toward R (not 0) where no fit token reaches
        for mu in MUS:                                               # (10-25 smoke: a tiny ridge toward 0 blew up)
            M = Ge + mu * Go; M.diagonal().add_(lam)
            Rs = torch.linalg.solve(M, Ge @ R.T + lam * R.T).T; V[f"soft{mu:g}"] = evalR(Rs); del M
            if mu == 1.0: save["R_soft1"][int(e)] = Rs.half().cpu()
        p_ = min(512, d // 2); P = torch.svd_lowrank(Ge, q=p_, niter=4)[0]   # selective read: top generalized eigen-
        Ger, Gor = P.T @ Ge @ P, P.T @ Go @ P                        # directions of (own, others) inside own's top-p span
        wo, Uo = torch.linalg.eigh(Gor); wo = wo.clamp(min=1e-3 * float(wo.clamp(min=0).sum()) / p_)
        Li = (Uo / wo.sqrt()) @ Uo.T                                 # others^-1/2 (floored): whitened generalized eigenproblem
        w, Y = torch.linalg.eigh(Li @ Ger @ Li); Xg = Li @ Y          # ascending: own / others energy ratio
        gratio = {}
        for k in GKS:
            if k > p_: continue
            Qg = torch.linalg.qr(P @ Xg[:, -k:])[0]; V[f"gen{k}"] = evalR((R @ Qg) @ Qg.T); gratio[k] = float(w[-k])
        yt = (test[e] @ R.T) @ Wr.T; den = float((yt ** 2).sum())
        mo = yt.mean(0); ml = torch.stack([((test[j] @ R.T) @ Wr.T).mean(0) for j in mem if j not in new]).mean(0)
        bias = dict(frac=float((mo ** 2).sum()) * yt.shape[0] / max(den, 1e-30),          # share of own output = its mean
                    cos_lib=float(mo @ ml / (mo.norm() * ml.norm()).clamp(min=1e-30)))    # same offset on library inputs?
        ys = (test[e] @ Vs) @ Usig.T if same else (test[e] @ Usig) @ Vs.T   # the library's shared part on the same inputs
        bias.update(cos_S=float(mo @ ys.mean(0) / (mo.norm() * ys.mean(0).norm()).clamp(min=1e-30)),   # offset vs S's offset
                    expl_S=float((yt * ys).sum()) ** 2 / max(den * float((ys ** 2).sum()), 1e-30))    # output share along S's output
        ref = {}
        for nm, Rd in (("refit_V", Vin), ("refit_rand", orth(d, 16, g)), ("refit_pca", torch.svd_lowrank(Ge, q=24, niter=4)[0][:, :16])):
            Cxx = Rd.T @ Ge @ Rd; Cxx.diagonal().add_(1e-3 * float(torch.trace(Cxx)) / Cxx.shape[0])
            Wn = (Wr @ (R @ Ge @ Rd)) @ torch.linalg.inv(Cxx)
            ref[nm] = float((((test[e] @ Rd) @ Wn.T - yt) ** 2).sum()) / max(den, 1e-30)
        recs.append(dict(stem=s_, e=int(e), d=d, out=int(Wr.shape[0]), cap=cap, gratio=gratio, bias=bias, V=V, refit=ref,
                         wshared=float(((yt @ Uout) ** 2).sum()) / max(den, 1e-30)))
        del Ge, Go, Qo, yt
    return recs, save

def run():
    global DEVICE
    import apply_lora as al
    import unified_pool_driver as U
    from train_lora import load_demo_samples
    import exp116_continual as C
    E = C.E; DEVICE = E.DEV; t0 = time.time()
    old, new = C.sequence(SEQ); mem = old + new
    B = C.CBuilder()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(E.BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=DEVICE)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); mods = dict(model.named_modules()); keys = {k: None for k in model.state_dict(keep_vars=True)}
    S = {}
    for s_ in B.ck:
        k = al.resolve_base_key(keys, s_)
        m = mods.get(k.rsplit(".weight", 1)[0]) if k else None
        if isinstance(m, torch.nn.Linear) and k in params: S[s_] = dict(k=k, mod=m)
    with torch.no_grad():                                            # the network = our merge at t = 8 (ours_lib)
        for s_, v in S.items():
            D = B._delta("ours_lib", {}, mem, s_)
            if D is None: continue
            p = params[v["k"]]; D = D.float(); D = D if D.shape == p.shape else D.T
            p.copy_((p.float() + D).to(p.dtype))
    print(f"[e138] order {SEQ}: {len(old)} library + {len(new)} external, {len(S)} layers; merged in {(time.time()-t0)/60:.1f} min", flush=True)
    F = {j: frames(E, E.LIB[j], load_demo_samples) for j in mem}
    print(f"[e138] frames loaded ({sum(len(v) for v in F.values())}) {(time.time()-t0)/60:.1f} min", flush=True)
    calls = {}                                                       # hook calls per frame per layer (VLM once, expert per step)
    hs = [v["mod"].register_forward_pre_hook(lambda _m, _i, s_=s_: calls.__setitem__(s_, calls.get(s_, 0) + 1)) for s_, v in S.items()]
    with torch.no_grad(): policy.infer({**F[mem[0]][0], "prompt": PROMPT(E.LIB[mem[0]])})
    for h in hs: h.remove()
    stems = [s_ for s_ in S if s_ in calls][:int(SMOKE) if SMOKE else None]; batches, cur, tot = [], [], 0
    for s_ in stems:
        d = S[s_]["mod"].in_features
        if cur and tot + d > DBUDGET: batches.append(cur); cur, tot = [], 0
        cur.append(s_); tot += d
    if cur: batches.append(cur)
    res = json.load(open(OUT)) if OUT.exists() else {"records": [], "done": []}
    print(f"[e138] {len(stems)} layers seen, calls/frame {sorted(set(calls.values()))}, {len(batches)} passes; {len(res['done'])} layers done", flush=True)
    for bi, bat in enumerate(batches):
        bat = [s_ for s_ in bat if s_ not in res["done"]]
        if not bat: continue
        t1 = time.time(); store = {}; ctx = {}
        per = {s_: max(1, math.ceil(2 * CAP / max(1, len(F[mem[0]])) / calls[s_])) for s_ in bat}
        def mk(s_):
            def pre(_m, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]); n = x.shape[0]
                ix = torch.randperm(n, device=x.device)[:per[s_]]
                store.setdefault((s_, ctx["j"], ctx["sp"]), []).append(x[ix].to(torch.float16))
            return pre
        hs = [S[s_]["mod"].register_forward_pre_hook(mk(s_)) for s_ in bat]
        with torch.no_grad():
            for j in mem:
                for q, ob in enumerate(F[j]):
                    ctx.update(j=j, sp="fit" if q % 2 == 0 else "test"); policy.infer({**ob, "prompt": PROMPT(E.LIB[j])})
        for h in hs: h.remove()
        t2 = time.time(); nrec = 0; saves = {}
        with torch.no_grad():
            for s_ in bat:
                H = {}
                for j in mem:
                    for sp in ("fit", "test"):
                        x = torch.cat(store.pop((s_, j, sp)), 0); H[(j, sp)] = x[torch.randperm(x.shape[0], device=x.device)[:CAP]]
                recs, sv = analyse(s_, S[s_]["mod"].in_features, H, mem, new, B.fac, U.USIG[s_].to(DEVICE), U.VS[s_].to(DEVICE),
                                   tuple(params[S[s_]["k"]].shape))
                res["records"] += recs; res["done"].append(s_); nrec += len(recs); saves[s_] = sv; del H
                torch.cuda.empty_cache()
        SAVE.mkdir(parents=True, exist_ok=True); torch.save(saves, SAVE / f"pass{bi:02d}.pt")
        json.dump(res, open(OUT, "w"))
        print(f"[e138] pass {bi + 1}/{len(batches)}: {len(bat)} layers, {nrec} records; collect {(t2-t1)/60:.1f} min, analyse {(time.time()-t2)/60:.1f} min", flush=True)
    print(f"[e138] run DONE in {(time.time()-t0)/60:.1f} min", flush=True)

def report():
    res = json.load(open(OUT)); R = res["records"]; ex = sorted({r["e"] for r in R})
    group = lambda s_: ("expert" if "gemma_expert" in s_ else "vlm") + ("-mlp" if "mlp" in s_ else "-attn")
    def agg(rs, var):                                                # energy-weighted over the layers that have var
        rs = [r for r in rs if var in r["V"]]; own = sum(r["V"]["base"][0] for r in rs)
        if not own: return None
        k = [sum(r["V"][var][i] for r in rs) / max(sum(r["V"]["base"][i] for r in rs), 1e-30) for i in range(3)]
        err = sum(r["V"][var][3] for r in rs) / own
        return k + [err]
    vars_ = ["base"] + [f"{p}{q}" for q in QS for p in ("hard", "rand")] + [f"soft{m:g}" for m in MUS] + [f"gen{k}" for k in GKS]
    L = [f"=== EXP-138 offline gate, order {SEQ} (scratch, our merge at t=8): {len(res['done'])} layers, {len(ex)} externals ===",
         "firing of each external update, energy per token relative to its own inputs (base): library / other externals"]
    fl = [(sum(r["V"]["base"][1] for r in R if r["e"] == e) / sum(r["V"]["base"][0] for r in R if r["e"] == e),
           sum(r["V"]["base"][2] for r in R if r["e"] == e) / sum(r["V"]["base"][0] for r in R if r["e"] == e)) for e in ex]
    L.append(f"  library {np.mean([a for a, _ in fl]):.2f} (range {min(a for a, _ in fl):.2f}-{max(a for a, _ in fl):.2f}) / "
             f"other externals {np.mean([b for _, b in fl]):.2f} (range {min(b for _, b in fl):.2f}-{max(b for _, b in fl):.2f})")
    L.append("direction 1 (mean over externals): kept own / library / other externals, own error; captured = others' input energy removed")
    for v in vars_:
        a = [agg([r for r in R if r["e"] == e], v) for e in ex]; a = [x for x in a if x is not None]
        if not a: continue
        m = np.mean(a, 0); q = int(re.sub(r"\D", "", v) or 0) if v.startswith(("hard", "rand")) else None
        capn = f"  captured {np.mean([r['cap'][str(q)] for r in R if str(q) in r['cap']]):.2f}" if v.startswith("hard") else ""
        if v.startswith("gen"): capn = f"  own/others ratio of the k-th direction (median) {np.median([r['gratio'][v[3:]] for r in R if v[3:] in r['gratio']]):.2f}"
        L.append(f"  {v:9s} own {m[0]:.2f}  library {m[1]:.2f}  other ext {m[2]:.2f}  own error {m[3]:.2f}{capn}")
    L.append("direction 2 (mean over externals, energy-weighted): refit test error on own inputs; wshared = own output in shared outputs")
    for nm in ("refit_V", "refit_rand", "refit_pca"):
        L.append(f"  {nm:10s} {np.mean([sum(r['refit'][nm] * r['V']['base'][0] for r in R if r['e'] == e) / sum(r['V']['base'][0] for r in R if r['e'] == e) for e in ex]):.2f}")
    wavg = lambda key, sub: np.mean([sum(r[key][sub] * r['V']['base'][0] for r in R if r['e'] == e) / sum(r['V']['base'][0] for r in R if r['e'] == e) for e in ex])
    L.append(f"  bias: share of own output that is a constant offset {wavg('bias', 'frac'):.2f}; cosine of that offset on own vs library inputs {wavg('bias', 'cos_lib'):.2f}")
    L.append(f"  vs the library's shared part on the same inputs: cosine of offsets {wavg('bias', 'cos_S'):.2f}; share of output along the shared part's output {wavg('bias', 'expl_S'):.2f}")
    L.append(f"  wshared    {np.mean([sum(r['wshared'] * r['V']['base'][0] for r in R if r['e'] == e) / sum(r['V']['base'][0] for r in R if r['e'] == e) for e in ex]):.3f}")
    L.append("by layer group (pooled over externals): own share of output energy | base firing library / ext | soft10 kept own / lib / ext")
    for gname in sorted({group(r["stem"]) for r in R}):
        rs = [r for r in R if group(r["stem"]) == gname]; tot = sum(r["V"]["base"][0] for r in R)
        b0 = sum(r["V"]["base"][0] for r in rs); s10 = agg(rs, "soft10")
        L.append(f"  {gname:12s} {sum(r['V']['base'][0] for r in rs) / tot:.2f} | {sum(r['V']['base'][1] for r in rs) / b0:.2f} / "
                 f"{sum(r['V']['base'][2] for r in rs) / b0:.2f} | " + (" / ".join(f"{x:.2f}" for x in s10[:3]) if s10 else "-"))
    txt = "\n".join(L); print(txt); REP.write_text(txt + "\n")

if __name__ == "__main__":
    DEVICE = None
    run() if os.environ.get("E138_STAGE", "run") == "run" else report()
