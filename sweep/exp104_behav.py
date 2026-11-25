#!/usr/bin/env python3
"""EXP-104 — four closed-loop-aware behavioural donor signals (approved 10-05), no donor rollouts.

For every (weak target t, donor d): merged policy W = base + S16 + r_t + c * r_d (c = ref / ||r_d||; the probe's
operator), built in place on the GPU exactly as EXP-96. Target alone = base + S16 + r_t. Common random flow noise
per frame across all donors of a target. Signals (normalised action space, 7 action dims):

 CI   critical-interval imitation error (Huang et al. 2026, CI-MSE; validated on pi0.5 checkpoints):
      3 target demos, every step in windows [toggle-16, toggle+8] around each gripper toggle and the last 16 steps
      (the goal-completing phase); executed action at step s = temporal ensemble of the chunks predicted at the
      window frames f <= s <= f+4 (replan every 5).  ci = mean_s ||ens(s) - a*(s)||^2 ; dci = ci - ci(target)
      corr = -mean_s <ens_m(s) - ens_0(s), ens_0(s) - a*(s)>   (> 0: donor pushes toward the expert where the
      target errs; GPC-style error-alignment term)
 RC   replan consistency (deterministic STAC, Agia et al. 2024): for window frames f, f+5:
      ||a_f[5:10] - a_{f+5}[0:5]||^2 ; drc = rc - rc(target)
 CF   counterfactual-prompt contrast (goal confusion; LIBERO-CF): on 16 window frames,
      m = mean ||a(x, prompt_t) - a(x, prompt_d)||^2 ; cf = m(merged) / m(target alone, same prompt_d)
 RP   on-policy replay on the TARGET's own rollout states (EXP-104 stage 0 dumps; no donor rollouts):
      32 frames from failed episodes (second half), 32 from successful ones.
      sel = ||da||^2 on failure frames / ||da||^2 on success frames, da = a_merged - a_target
      vinn = error reduction on failure frames vs a pseudo-label: action chunk of the nearest target-demo frame
             (centred base-SigLIP embedding, EXP-103c), only matches with cosine >= median
Output results/analysis/exp104_behav[_sN].npz: [N, N+1] per signal (col N = target alone). argv: device shard i/S
Env: B4_TARGETS (comma list) overrides sharding; B4_BATCH (default 32).
"""
import json, os, pathlib, sys, time, glob
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import apply_lora as al
import unified_pool_driver as U
from reinforce_train import scenes as SC
from train_lora import _observation_from, DEMOS, RESIZE
from subset_probe import TARGETS as WEAK
N = 76; UNIQUE = [t for t in range(N) if t not in (18, 62)]
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
si, sS = map(int, (sys.argv[2] if len(sys.argv) > 2 else "0/1").split("/"))
BASE_DIR = "/media/pinas/Chris/models/pi05_base_pytorch"
SMOKE = os.environ.get("B4_SMOKE") == "1"                                     # 3 donors, separate output
OUT = REPO / f"results/analysis/exp104_behav{'_smoke' if SMOKE else ('' if sS == 1 else f'_s{si}')}.npz"
SIG = ["ci", "dci", "corr", "rc", "drc", "cf", "sel", "vinn", "rp_fail", "rp_succ"]
REF = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
BATCH = int(os.environ.get("B4_BATCH", "32")); REPLAN = 5; AD = 7

def demo_windows(task, n_demos=3):
    """[(obs, chunk_norm, demo_idx, step)] at every step inside the critical windows of the first n_demos demos."""
    import h5py
    from openpi_client import image_tools
    from openpi.runtime import Pi05LiberoConfig
    H, ADIM = Pi05LiberoConfig().action_horizon, Pi05LiberoConfig().action_dim
    ns = json.load(open(REPO / "results/analysis/norm_assets/norm_stats.json")); st_a = ns.get("norm_stats", ns)["actions"]
    q01, q99 = np.asarray(st_a["q01"], np.float32), np.asarray(st_a["q99"], np.float32)
    out = []
    with h5py.File(DEMOS / "libero_90_no_noops" / f"{task}_demo.hdf5", "r") as h:
        d = h["data"] if "data" in h else h
        for di, k in enumerate(sorted(d.keys())[:n_demos]):
            g = d[k]; raw = np.asarray(g["actions"], np.float32); T = raw.shape[0]
            acts = raw.copy(); kq = q01.shape[-1]; acts[:, :kq] = 2.0 * (acts[:, :kq] - q01) / (q99 - q01 + 1e-6) - 1.0
            tog = np.where(np.diff(np.sign(raw[:, 6])) != 0)[0]
            steps = set()
            for s in tog: steps |= set(range(max(0, s - 16), min(T - H, s + 9)))
            steps |= set(range(max(0, T - H - 16), T - H))
            for s in sorted(steps):
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(g["obs/agentview_rgb"][s][::-1, ::-1]), RESIZE, RESIZE))
                wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(g["obs/eye_in_hand_rgb"][s][::-1, ::-1]), RESIZE, RESIZE))
                state = np.concatenate((g["obs/ee_pos"][s], g["obs/ee_ori"][s], g["obs/gripper_states"][s])).astype(np.float32)
                chunk = np.zeros((H, ADIM), np.float32); chunk[:, :acts.shape[1]] = acts[s:s + H]
                out.append(({"observation/image": img, "observation/wrist_image": wri, "observation/state": state}, chunk, di, s))
    return out

def replay_frames(t, n=32, seed=0):
    """target-alone rollout frames (EXP-104 stage 0): n from failed episodes (second half), n from successful ones."""
    fs = sorted(glob.glob(str(REPO / f"results/exp104_dumps/e104_t{t}_k16/*/ep*_*.npz")))
    rng = np.random.default_rng(10400 + t + seed); out = {"fail": [], "succ": []}
    for kind, tag in (("fail", "failure"), ("succ", "success")):
        pool = []
        for f in [f for f in fs if f.endswith(f"{tag}.npz")]:
            z = np.load(f); im, wr, sta = z["img"], z["wrist"], z["state"]        # decompress once per file
            m = len(im); lo = m // 2 if kind == "fail" else 0
            pool += [(im[i], wr[i], sta[i]) for i in range(lo, m)]
        if pool:
            sel = rng.choice(len(pool), min(n, len(pool)), replace=False)
            out[kind] = [{"observation/image": pool[i][0], "observation/wrist_image": pool[i][1], "observation/state": pool[i][2]} for i in sel]
    return out

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE_DIR, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); keys = {k: None for k in model.state_dict(keep_vars=True)}
    stem2p = {}
    for s_ in U.stems:
        k = al.resolve_base_key(keys, s_)
        if k is not None and k in params: stem2p[s_] = k
    pristine = {k: params[k].detach().to("cpu").pin_memory() for k in stem2p.values()}
    def resid_dense(key, stem, c=1.0):
        Usig = U.USIG[stem]; Un = Usig / torch.linalg.norm(Usig, dim=0).clamp(min=1e-12); V = U.VS[stem]
        b, a = U.fold(key)[stem]; b, a = b.to(dev), a.to(dev); Un, V = Un.to(dev), V.to(dev)
        D = b @ a; return c * (D - Un @ (Un.T @ D @ V) @ V.T)
    def add(delta_fn):
        with torch.no_grad():
            for s_, k in stem2p.items():
                d = delta_fn(s_)
                if d is None: continue
                p = params[k]
                if d.shape != p.shape: d = d.T
                p.add_(d.to(p.dtype))
    def restore():
        with torch.no_grad():
            for k in pristine: params[k].copy_(pristine[k], non_blocking=True)
    cfg = model.config; H = cfg.action_horizon
    def infer(obs_list, prompts, noise):
        """actions [n, H, AD] for obs_list[i] with prompts[i] and noise[i]"""
        out = []
        with torch.no_grad():
            for i in range(0, len(obs_list), BATCH):
                ob, pr = obs_list[i:i + BATCH], prompts[i:i + BATCH]
                groups = {}
                for j, p_ in enumerate(pr): groups.setdefault(p_, []).append(j)
                a = torch.empty((len(ob), H, cfg.action_dim), device=dev)
                for p_, js in groups.items():
                    o = _observation_from(policy, [ob[j] for j in js], p_, dev)
                    a[js] = model.sample_actions(dev, o, noise=noise[[i + j for j in js]].clone(), num_steps=10).float()
                out.append(a[..., :AD])
        return torch.cat(out, 0)
    # siglip embedding of the base model for the VINN labels (centred as in EXP-103c)
    fz = np.load(REPO / "results/analysis/exp103_frames.npz"); mu = torch.from_numpy(fz["mu"]).to(dev)
    pw = model.paligemma_with_expert
    def emb(obs_list):
        out = []
        with torch.no_grad():
            for cam_i, cam in enumerate(("observation/image", "observation/wrist_image")):
                x = np.stack([o[cam] for o in obs_list]); fs = []
                for i in range(0, len(x), 64):
                    xb = torch.from_numpy(x[i:i + 64]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1.0
                    with torch.autocast("cuda", dtype=torch.bfloat16): fs.append(pw.embed_image(xb).float().mean(1))
                out.append(torch.cat(fs))
        e = torch.stack(out, 1) - mu
        return torch.nn.functional.normalize(e, dim=2).flatten(1) / 2 ** 0.5
    def demo_emb(t):
        e = torch.from_numpy(fz[f"E{t}"].astype(np.float32)).to(dev) - mu
        return torch.nn.functional.normalize(e, dim=2).flatten(1) / 2 ** 0.5, torch.from_numpy(fz[f"A{t}"]).to(dev)

    E = {s: np.full((N, N + 1), np.nan) for s in SIG}
    if OUT.exists():
        z = np.load(OUT); E = {s: z[s].copy() for s in SIG}
    order = [t for t in UNIQUE if t in WEAK]
    mine = [int(x) for x in os.environ["B4_TARGETS"].split(",")] if os.environ.get("B4_TARGETS") else [t for i, t in enumerate(order) if i % sS == si]
    print(f"[b4] {len(mine)} targets {mine} on {dev}; ref {REF:.3f}; setup {time.time()-t0:.0f}s", flush=True)
    for t in mine:
        if not np.isnan(E["ci"][t, [d for d in UNIQUE if d != t]]).any(): continue
        t1 = time.time(); task = SC[t][0]; prompt_t = task.replace("_", " ")
        W = demo_windows(task); print(f"[b4] t{t}: {len(W)} window frames loaded ({time.time()-t1:.0f}s)", flush=True)
        wobs = [w[0] for w in W]; astar = torch.tensor(np.stack([w[1][:, :AD] for w in W]), device=dev)
        key = [(w[2], w[3]) for w in W]; idx = {k: i for i, k in enumerate(key)}
        # ensembling map: executed step s <- (frame f, offset s-f) for window frames f <= s <= f+4
        ens = [[(idx[(dd, s - o)], o) for o in range(REPLAN) if (dd, s - o) in idx] for (dd, s) in key]
        a_star_s = astar[:, 0]                                                   # expert action at step s
        pairs = [(idx[k], idx[(k[0], k[1] + REPLAN)]) for k in key if (k[0], k[1] + REPLAN) in idx]
        cfi = np.linspace(0, len(W) - 1, 16).round().astype(int)
        RP = replay_frames(t); print(f"[b4] t{t}: replay {len(RP['fail'])} fail / {len(RP['succ'])} succ ({time.time()-t1:.0f}s)", flush=True)
        g = torch.Generator().manual_seed(10400 + t)
        nz = lambda n: torch.randn((n, H, cfg.action_dim), generator=g).to(dev)
        NZ_W, NZ_CF, NZ_F, NZ_S = nz(len(W)), nz(16), nz(max(1, len(RP["fail"]))), nz(max(1, len(RP["succ"])))
        # VINN labels for failure frames (base SigLIP, before any adapter is applied)
        restore()
        if RP["fail"]:
            De, Da = demo_emb(t); ef = emb(RP["fail"]); s_, j_ = (ef @ De.T).max(1)
            keep = s_ >= s_.median(); lab = Da[j_]
        def signals(prompts_cf_d=None):
            a = infer(wobs, [prompt_t] * len(wobs), NZ_W)
            ex = torch.stack([torch.stack([a[f, o] for f, o in e]).mean(0) for e in ens])        # [n, AD]
            ci = (ex - a_star_s).pow(2).sum(1).mean().item()
            rc = torch.stack([(a[f][REPLAN:] - a[f2][:H - REPLAN]).pow(2).sum(1).mean() for f, f2 in pairs]).mean().item() if pairs else np.nan
            r = dict(a=a, ex=ex, ci=ci, rc=rc)
            if prompts_cf_d is not None:
                acf = infer([wobs[i] for i in cfi], [prompts_cf_d] * 16, NZ_CF)
                r["m"] = (a[cfi] - acf).pow(2).sum((1, 2)).mean().item()
            if RP["fail"]: r["af"] = infer(RP["fail"], [prompt_t] * len(RP["fail"]), NZ_F)
            if RP["succ"]: r["as"] = infer(RP["succ"], [prompt_t] * len(RP["succ"]), NZ_S)
            return r
        ft = U.fold(t); add(lambda s_: (U.USIG[s_] @ U.VS[s_].T).to(dev) + resid_dense(t, s_) if s_ in ft else None)
        snap = {k: params[k].detach().to("cpu").pin_memory() for k in stem2p.values()}
        r0 = signals(); print(f"[b4] t{t}: target-alone signals ({time.time()-t1:.0f}s)", flush=True)
        # target-alone margins for every donor's prompt (same frames / noise as the merged ones)
        a0cf = {d: infer([wobs[i] for i in cfi], [SC[d][0].replace("_", " ")] * 16, NZ_CF) for d in (UNIQUE[:4] if SMOKE else UNIQUE) if d != t}
        m0 = {d: (r0["a"][cfi] - a0cf[d]).pow(2).sum((1, 2)).mean().item() for d in a0cf}
        E["ci"][t, N], E["rc"][t, N] = r0["ci"], r0["rc"]; print(f"[b4] t{t}: target-alone prompt margins ({time.time()-t1:.0f}s)", flush=True)
        for d in (UNIQUE[:4] if SMOKE else UNIQUE):
            if d == t or not np.isnan(E["ci"][t, d]): continue
            td = time.time()
            c = REF / U.resid_norm(d); fd = U.fold(d)                              # once per donor (was per module)
            add(lambda s_: resid_dense(d, s_, c) if s_ in fd else None)
            r = signals(SC[d][0].replace("_", " "))
            E["ci"][t, d] = r["ci"]; E["dci"][t, d] = r["ci"] - r0["ci"]
            E["corr"][t, d] = -((r["ex"] - r0["ex"]) * (r0["ex"] - a_star_s)).sum(1).mean().item()
            E["rc"][t, d] = r["rc"]; E["drc"][t, d] = r["rc"] - r0["rc"]
            E["cf"][t, d] = r["m"] / max(m0[d], 1e-12)
            if RP["fail"]:
                df = (r["af"] - r0["af"]).pow(2).sum((1, 2)); E["rp_fail"][t, d] = df.mean().item()
                E["vinn"][t, d] = ((r0["af"][keep] - lab[keep]).pow(2).sum((1, 2)) - (r["af"][keep] - lab[keep]).pow(2).sum((1, 2))).mean().item()
            if RP["succ"]:
                ds_ = (r["as"] - r0["as"]).pow(2).sum((1, 2)); E["rp_succ"][t, d] = ds_.mean().item()
            if RP["fail"] and RP["succ"]: E["sel"][t, d] = E["rp_fail"][t, d] / max(E["rp_succ"][t, d], 1e-12)
            with torch.no_grad():
                for k in stem2p.values(): params[k].copy_(snap[k], non_blocking=True)
                torch.cuda.synchronize(dev)
            if SMOKE: print(f"[b4] smoke d{d}: {time.time()-td:.1f}s  " + " ".join(f"{k}={E[k][t, d]:.4g}" for k in SIG), flush=True)
        del snap; restore(); np.savez(OUT, **E)
        print(f"[b4] t{t}: {len(W)} window frames, {len(pairs)} replan pairs, replay {len(RP['fail'])} fail / {len(RP['succ'])} succ; "
              f"target ci {r0['ci']:.4f} rc {r0['rc']:.4f}; 73 donors in {(time.time()-t1)/60:.1f} min", flush=True)
    print(f"[b4] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
