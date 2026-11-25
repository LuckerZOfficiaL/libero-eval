#!/usr/bin/env python3
"""EXP-143 — through which TOKENS must an external skill's update act? (10-25, offline, after EXP-139 partial / EXP-142)

EXP-140/142: externals need near-full weight on their own inputs, and every per-layer input gate leaks onto other tasks
because all tasks' tokens share input directions. VLA-specific handle: pi0.5's prefix = image tokens (SigLIP, the same
robot / table across tasks) + the instruction's text tokens (different words per task, full attention with the images);
the action expert reads the prefix through attention. If a skill's update only needs to act on the TEXT tokens (and the
action expert), it can be keyed on the instruction, which separates tasks far better than images.
Oracle test (position masks applied by forward hooks, not a static merge): external e's update D_e (low-rank, full
weight) is added only at chosen positions / pathways, on top of W0 (alone) or W0 + S (the library's shared part):
  all          everywhere (on W0 this IS the reference; on W0 + S it is EXP-139's sown)
  text         VLM language-model layers, text positions only (last 200 prefix positions)
  image        VLM language-model layers, image positions only, plus vision-tower layers
  expert       action-expert layers only
  text+expert, image+expert
Proxy = EXP-141 relative action error vs e's single-task model on e's 16 demo frames (same frames / noise; references
recomputed here: base, and e's update everywhere by hook; cross-checked against EXP-141's merged-weight models): 0 = acts like single-task, 1 = like base.
PRE-REGISTERED (10-25, before running): the instruction route is viable if text+expert on W0 has mean error <= 0.3 over
the 8 externals of order 0 AND clearly below image+expert; dead if text+expert >= 0.6. (Proxy validity: EXP-141.)
Env: E143_SEQ (0); E143_MASKS (comma list; parts all|text|image|lmimage|vision|expert joined by +); E143_WITHS
(0,1); E143_TAG (report suffix); E112_DEV. Output: results/analysis/exp143_report_s{SEQ}.txt.
"""
import os, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
SEQ = int(os.environ.get("E143_SEQ", "0")); NFR, STEPS, ADIM, NTXT = 16, 5, 7, 200
os.environ.setdefault("E138_FRAMES", str(NFR))
REF = REPO / f"results/analysis/exp141_actions_s{SEQ}.npz"; REP = REPO / f"results/analysis/exp143_report_s{SEQ}.txt"
MASKS = os.environ.get("E143_MASKS", "all,text,image,expert,text+expert,image+expert").split(",")
WITHS = [bool(int(x)) for x in os.environ.get("E143_WITHS", "0,1").split(",")]
if "E143_TAG" in os.environ: REP = REP.with_name(REP.stem + os.environ["E143_TAG"] + ".txt")

def path_of(s_):
    if any(t in s_ for t in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp")): return "expert"   # suffix pathway
                                                                     # (10-25 fix: the action / time projections were "lm" in run 1)
    if any(t in s_ for t in ("vision_tower", "siglip", "multi_modal_projector")): return "vision"
    return "lm"

def main():
    import apply_lora as al
    import exp138_readgate as G
    import exp140_softgate as P
    from train_lora import load_demo_samples
    E, C, U = P.E, P.C, P.U; dev = E.DEV; t0 = time.time()
    old, new = C.sequence(SEQ); mem = old + new
    B = P.GBuilder()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(E.BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); mods = dict(model.named_modules()); keys = {k: None for k in model.state_dict(keep_vars=True)}
    S = {}
    for s_ in B.ck:
        k = al.resolve_base_key(keys, s_)
        m = mods.get(k.rsplit(".weight", 1)[0]) if k else None
        if isinstance(m, torch.nn.Linear) and k in params: S[s_] = (params[k], m)
    W0 = {s_: p.detach().to("cpu", copy=True) for s_, (p, m) in S.items()}
    rng = np.random.default_rng(141); NZ = [rng.standard_normal((10, 32)).astype(np.float32) for _ in range(NFR)]
    print(f"[e143] order {SEQ}: {len(S)} layers ({sum(path_of(s_) == 'lm' for s_ in S)} lm, {sum(path_of(s_) == 'vision' for s_ in S)} vision, "
          f"{sum(path_of(s_) == 'expert' for s_ in S)} expert)", flush=True)

    def set_base(withS):
        with torch.no_grad():
            for s_, (p, m) in S.items():
                p.copy_(W0[s_].to(dev))
                if withS:
                    D = (U.USIG[s_].to(dev) @ U.VS[s_].to(dev).T).float(); D = D if D.shape == p.shape else D.T
                    p.copy_((p.float() + D).to(p.dtype))
    seen = {}
    def hooks(e, mask):
        hs = []
        for s_, (p, m) in S.items():
            f = B.fac(e).get(s_); pth = path_of(s_)
            if f is None: continue
            P_ = set(mask.split("+")); A_ = "all" in P_
            vis, ltx, lim, exp = A_ or bool(P_ & {"image", "vision"}), A_ or "text" in P_, A_ or bool(P_ & {"image", "lmimage"}), A_ or "expert" in P_
            if not {"expert": exp, "vision": vis, "lm": ltx or lim}[pth]: continue
            b, a = f[0].to(dev).float(), f[1].to(dev).float()
            L, R = (b, a) if b.shape[0] == p.shape[0] and a.shape[1] == p.shape[1] else (a.T, b.T)   # y += x R^T L^T (b @ a in
                                                                     # weight orientation when shapes allow, as the builders)
            def fwd(_m, inp, out, L=L, R=R, pth=pth, s_=s_, ltx=ltx, lim=lim):
                x = inp[0]; dy = (x.float() @ R.T) @ L.T
                if pth == "lm" and not (ltx and lim) and x.dim() == 3:
                    seen[s_] = x.shape[1]; pos = torch.zeros(x.shape[1], device=x.device, dtype=dy.dtype)
                    if ltx: pos[-NTXT:] = 1
                    if lim: pos[:-NTXT] = 1
                    dy = dy * pos[None, :, None]
                return out + dy.to(out.dtype)
            hs.append(m.register_forward_hook(fwd))
        return hs
    def act(j):
        a_ = mem.index(j); out = np.zeros((NFR, STEPS, ADIM), np.float32)
        with torch.no_grad():
            for f, ob in enumerate(F[j]): out[f] = policy.infer({**ob, "prompt": G.PROMPT(E.LIB[j])}, noise=NZ[f])["actions"][:STEPS, :ADIM]
        return out, a_
    F = {j: G.frames(E, E.LIB[j], load_demo_samples) for j in new}
    set_base(False); ref = {}                                        # references on the same frames / noise: base and
    for e in new:                                                    # e's single-task model (its update everywhere, by hook)
        ref[("base", e)] = act(e)[0]; hs = hooks(e, "all"); ref[("single", e)] = act(e)[0]
        for h in hs: h.remove()
    chk = ""
    if REF.exists():                                                 # cross-check vs EXP-141 (weights merged in bf16)
        z = np.load(REF)
        if all(f"single_{e}" in z for e in new):
            d = [np.mean(np.sum((ref[("single", e)] - z[f"single_{e}"][mem.index(e)]) ** 2, (1, 2))) / np.mean(np.sum((ref[("base", e)] - ref[("single", e)]) ** 2, (1, 2))) for e in new]
            chk = f"hook vs merged-weight single-task model (EXP-141): relative difference mean {np.mean(d):.3f}, max {np.max(d):.3f}"
    rel = lambda A, j, a_: float(np.mean(np.sum((A - ref[("single", j)]) ** 2, (1, 2))) / np.mean(np.sum((ref[("base", j)] - ref[("single", j)]) ** 2, (1, 2))))
    L_ = [f"=== EXP-143 order {SEQ}: external update applied only through some tokens / pathways (relative action error vs single-task; 0 = same, 1 = base) ==="]
    tab = {}
    for withS in WITHS:
        set_base(withS)
        for e in new:
            row = []
            for mk in MASKS:
                hs = hooks(e, mk); A, a_ = act(e)
                for h in hs: h.remove()
                v = rel(A, e, a_); tab[(withS, e, mk)] = v; row.append(f"{v:6.2f}")
            L_.append(f"{'W0+S' if withS else 'W0  '} {E.LIB[e]['tid']:9s} " + " ".join(row))
            print(L_[-1], f"({(time.time()-t0)/60:.1f} min)", flush=True)
    hdr = "                " + " ".join(f"{m:>6s}"[:6] for m in MASKS)
    L_.insert(1, hdr + "   (columns: " + ", ".join(MASKS) + ")")
    for withS in WITHS:
        L_.append(f"mean {'W0+S' if withS else 'W0  '}       " + " ".join(f"{np.mean([tab[(withS, e, m)] for e in new]):6.2f}" for m in MASKS))
    L_.append(f"prefix lengths seen at lm layers: {sorted(set(seen.values()))}; {chk}")
    txt = "\n".join(L_); print(txt); REP.write_text(txt + "\n")

if __name__ == "__main__":
    main()
