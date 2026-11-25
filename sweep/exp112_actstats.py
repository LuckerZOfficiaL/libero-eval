#!/usr/bin/env python3
"""EXP-112 input statistics for the input-weighted residual merge (ours_rm, a diagonal RegMean on residuals).
For every library task (74 fleet + 30 external): apply the task's own LoRA to the base pi0.5 (dense, in place), run 32 of
its demo frames (policy.infer, its own prompt), and record per adapted Linear the mean squared input per input feature
(diag of the input second moment). Output results/analysis/exp112_actstats.npz: G [104, sum din] float32 + offsets.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import apply_lora as al
import unified_pool_driver as U
from train_lora import load_demo_samples
from exp112_bench import LIB, BASE_DIR
from la_pair_features import load
OUT = REPO / "results/analysis/exp112_actstats.npz"
import re
# the LIBERO instruction the eval harness sends: the task key WITHOUT its scene prefix (10-10 fix; earlier offline
# analyses EXP-96/101/104 sent "KITCHEN SCENE9 put ..." -- minor prompt mismatch)
PROMPT = lambda e: re.sub(r"^.*?SCENE\d+_", "", e["scene"]).replace("_", " ")
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"

def main():
    t0 = time.time()
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(str(BASE_DIR), norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model; model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    params = dict(model.named_parameters()); mods = dict(model.named_modules()); keys = {k: None for k in model.state_dict(keep_vars=True)}
    f0 = load(0); S = {}
    for s_ in U.stems:
        k = al.resolve_base_key(keys, s_)
        if k is None or k not in params or s_ not in f0: continue
        m = mods.get(k.rsplit(".weight", 1)[0])
        if isinstance(m, torch.nn.Linear): S[s_] = dict(k=k, mod=m)
    stems = sorted(S); offs, o = {}, 0
    for s_ in stems: din = S[s_]["mod"].in_features; offs[s_] = (o, din); o += din
    pristine = {v["k"]: params[v["k"]].detach().to("cpu").pin_memory() for v in S.values()}
    G = np.zeros((len(LIB), o), np.float32); done = np.zeros(len(LIB), bool)
    if OUT.exists():
        z = np.load(OUT); G[:] = z["G"]; done[:] = z["done"]
    print(f"[as] {len(stems)} modules, {o} input features; setup {(time.time()-t0)/60:.1f} min", flush=True)
    acc = {}
    def mk(s_):
        def pre(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            a = acc.setdefault(s_, [0.0, 0]); a[0] = a[0] + (x * x).sum(0); a[1] += x.shape[0]
        return pre
    hooks = [S[s_]["mod"].register_forward_pre_hook(mk(s_)) for s_ in stems]
    for i, e in enumerate(LIB):
        if done[i]: continue
        t1 = time.time()
        f = load(e["key"]) if not e["ext"] else U.fold(e["key"])
        with torch.no_grad():
            for k_, v in pristine.items(): params[k_].copy_(v.to(dev))
            for s_ in stems:
                if s_ not in f: continue
                b, a = f[s_]; p = params[S[s_]["k"]]; D = (b @ a).to(dev)
                p.add_((D if D.shape == p.shape else D.T).to(p.dtype))
        suite = e["suite"] + "_no_noops"
        try:
            smp = load_demo_samples(suite, e["scene"], n_demos=4, stride=4)
        except FileNotFoundError:                         # e.g. t57 has no demo file: frames of a same-scene sibling,
            import re                                     # run with THIS task's prompt and adapter
            scn = re.match(r"(.*?SCENE\d+)_", e["scene"]).group(1)
            sib = next(x for x in LIB if x is not e and x["suite"] == e["suite"] and x["scene"].startswith(scn + "_")
                       and (pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5") / suite / f"{x['scene']}_demo.hdf5").exists())
            smp = load_demo_samples(suite, sib["scene"], n_demos=4, stride=4)
            print(f"[as] {e['tid']}: no demos -> frames of same-scene sibling {sib['tid']}", flush=True)
        ix = np.linspace(0, len(smp) - 1, min(32, len(smp))).round().astype(int)
        acc.clear()
        for j in ix:
            obs, _ = smp[j]; policy.infer({**obs, "prompt": PROMPT(e)})
        for s_ in stems:
            if s_ in acc:
                o_, d_ = offs[s_]; G[i, o_:o_ + d_] = (acc[s_][0] / max(acc[s_][1], 1)).cpu().numpy()
        done[i] = True
        np.savez(OUT, G=G, done=done, offs=json.dumps(offs))
        print(f"[as] {i+1}/{len(LIB)} {e['tid']} ({e['suite']}): {len(ix)} frames, {len(acc)} modules seen, {time.time()-t1:.0f}s", flush=True)
    for h in hooks: h.remove()
    print(f"[as] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
