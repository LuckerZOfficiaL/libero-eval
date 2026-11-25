#!/usr/bin/env python3
"""EXP-141 — offline action proxy to screen merges before rollouts (10-25, after EXP-140).

Closed-loop rollouts cost ~1 GPU-hour per merged model, too slow to iterate on ideas for external-skill retention.
For order E141_SEQ (EXP-116 sequences, t = 8: 16 library + 8 external skills), every model below predicts an action
chunk on E141_FRAMES demo frames of each of the 24 skills (own prompt, same fixed noise per frame for every model):
  base      pi0.5 W0 (no adapter)
  single_j  W0 + D_j, skill j's single-task model (the reference behaviour)
  arms      merged models: the EXP-140 arms (ours = plain at 1/k, soft_k, soft_w0.25, soft_w1, plain_w0.25) + E141_ARMS
            ("lib" = library only; "x{i}-<arm>" = library + the i-th external only, k = 17)
Proxy per (model, skill j): relative action error = mean_f ||a_model - a_single_j||^2 / mean_f ||a_base - a_single_j||^2
over the executed part of the chunk (first 5 of 10 steps, 7 action dims): 0 = acts like j's single-task model,
1 = as far from it as the base model.
PRE-REGISTERED (10-25, before running): usable as a screen if, over the 5 EXP-140 arms x 24 skills of order 0 with
closed-loop retention (EXP-134 ours, EXP-140), Spearman(proxy, retention) <= -0.6 pooled AND the 10 cell means (arm x
library / external) rank as in closed loop (Spearman <= -0.8). Otherwise the screen is not used.
Env: E141_STAGE run|report; E141_SMOKE; E141_SEQ (0); E141_FRAMES (16); E141_ARMS (extra arms, comma list); E112_DEV (cuda device).
Output: results/analysis/exp141_actions_s{SEQ}.npz (actions per model), exp141_report_s{SEQ}.txt.
"""
import os, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
SEQ = int(os.environ.get("E141_SEQ", "0")); NFR = int(os.environ.get("E141_FRAMES", "16")); STEPS, ADIM = 5, 7
os.environ.setdefault("E138_FRAMES", str(NFR))
SMOKE = bool(os.environ.get("E141_SMOKE"))                         # smoke: base, 1 single, ours; separate file
OUT = REPO / f"results/analysis/exp141_actions_s{SEQ}{'_smoke' if SMOKE else ''}.npz"; REP = REPO / f"results/analysis/exp141_report_s{SEQ}.txt"
ARMS = ["ours", "soft_k", "soft_w0.25", "soft_w1", "plain_w0.25"] + [a for a in os.environ.get("E141_ARMS", "").split(",") if a]
CL = {"ours": "c134zs{s}t8-ours", **{a: "c140s{s}t8-" + a for a in ARMS[1:5]}}   # closed-loop record tags

def arm_spec(a, new):
    """arm name -> (kind, weight, externals merged) of exp140_softgate.GBuilder.delta. "lib" = library only (k = 16);
    "x{i}-<arm>" = library + only the i-th external of the order (k = 17), e.g. x0-plain_w1, x3-soft_w0.5"""
    if a == "lib": return "plain", None, []
    sub = new
    if a.startswith("x"): i, _, a = a[1:].partition("-"); sub = [new[int(i)]]
    if a == "ours": return "plain", None, sub
    kind, _, w = a.partition("_")
    return kind, (None if w == "k" else float(w.lstrip("w"))), sub

def run():
    import apply_lora as al
    import exp138_readgate as G
    import exp140_softgate as P
    from train_lora import load_demo_samples
    E, C = P.E, P.C; dev = E.DEV; t0 = time.time()
    old, new = C.sequence(SEQ); mem = old + new
    B = P.GBuilder()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(E.BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); keys = {k: None for k in model.state_dict(keep_vars=True)}
    S = {s_: params[k] for s_ in B.ck if (k := al.resolve_base_key(keys, s_)) in params}
    W0 = {s_: p.detach().to("cpu", copy=True) for s_, p in S.items()}
    F = {j: G.frames(E, E.LIB[j], load_demo_samples) for j in mem}
    rng = np.random.default_rng(141); NZ = [rng.standard_normal((10, 32)).astype(np.float32) for _ in range(NFR)]
    print(f"[e141] order {SEQ}: {len(S)} layers, {sum(len(v) for v in F.values())} frames, {(time.time()-t0)/60:.1f} min", flush=True)
    res = dict(np.load(OUT)) if OUT.exists() else {}

    def setw(delta):
        with torch.no_grad():
            for s_, p in S.items():
                p.copy_(W0[s_].to(dev)); D = delta(s_)
                if D is None: continue
                D = D.float(); D = D if D.shape == p.shape else D.T; p.copy_((p.float() + D).to(p.dtype))
    def act():
        out = np.zeros((len(mem), NFR, STEPS, ADIM), np.float32)
        with torch.no_grad():
            for a, j in enumerate(mem):
                for f, ob in enumerate(F[j]):
                    out[a, f] = policy.infer({**ob, "prompt": G.PROMPT(E.LIB[j])}, noise=NZ[f])["actions"][:STEPS, :ADIM]
        return out
    def single(j):
        def d(s_):
            f = B.fac(j).get(s_)
            return None if f is None else f[0].to(dev).float() @ f[1].to(dev).float()
        return d
    def arm(a):
        kind, w, sub = arm_spec(a, new)
        return lambda s_: B.delta(SEQ, kind, w, old, sub, s_)
    todo = [("base", lambda s_: None)] + [(f"single_{j}", single(j)) for j in mem] + [(a, arm(a)) for a in ARMS]
    for nm, fn in (todo[:2] + todo[-len(ARMS):-len(ARMS) + 1] if SMOKE else todo):
        if nm in res: continue
        t1 = time.time(); setw(fn); res[nm] = act(); np.savez(OUT, **res)
        print(f"[e141] {nm}: {(time.time()-t1):.0f}s", flush=True)
    print(f"[e141] run DONE in {(time.time()-t0)/60:.1f} min", flush=True)

def proxy(res, mem):
    """model -> {skill j: relative action error}"""
    out = {}
    for nm, A in res.items():
        if nm.startswith("single_") or nm == "base": continue
        out[nm] = {}
        for a, j in enumerate(mem):
            ref = res[f"single_{j}"][a]; den = np.mean(np.sum((res["base"][a] - ref) ** 2, (1, 2)))
            out[nm][j] = float(np.mean(np.sum((A[a] - ref) ** 2, (1, 2))) / den)
    return out

def report():
    from scipy import stats
    import exp113_controls as X
    import exp116_continual as C
    E = C.E; sg = X.single(); old, new = C.sequence(SEQ); mem = old + new
    res = dict(np.load(OUT)); px = proxy(res, mem)
    L = [f"=== EXP-141 offline action proxy, order {SEQ} (relative action error: 0 = single-task, 1 = base) vs closed-loop retention ==="]
    xs, ys, cells = [], [], []
    for nm in px:
        cl = E.member_success(CL[nm].format(s=SEQ), len(mem), "test", SEQ, sorted(mem)) if nm in CL else {}
        line = f"{nm:16s}"; sub = arm_spec(nm, new)[2]
        for g, grp in (("library", old), ("external", sub or new)):
            e_ = np.mean([px[nm][j] for j in grp]); line += f"  {g if sub or g == 'library' else 'external(absent)'} error {e_:5.2f}"
            rr = [cl[j] / sg[j] for j in grp if j in cl and sg.get(j, 0) > 0]
            if rr: line += f" (closed loop {100*np.mean(rr):5.1f}%)"; cells.append((e_, np.mean(rr)))
        for j in old + sub:
            if j in cl and sg.get(j, 0) > 0: xs.append(px[nm][j]); ys.append(cl[j] / sg[j])
        L.append(line)
    if xs:
        r = stats.spearmanr(xs, ys); L.append(f"pooled per (arm, skill): Spearman {r.correlation:+.2f} (n={len(xs)}, p={r.pvalue:.2g})")
    if len(cells) > 2:
        r = stats.spearmanr(*zip(*cells)); L.append(f"cell means (arm x group): Spearman {r.correlation:+.2f} (n={len(cells)})")
    txt = "\n".join(L); print(txt); REP.write_text(txt + "\n")

if __name__ == "__main__":
    run() if os.environ.get("E141_STAGE", "run") == "run" else report()
