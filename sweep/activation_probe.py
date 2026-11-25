#!/usr/bin/env python
"""Offline activation probing: per-adapter activation signatures on a fixed observation bank.

Loads the base PI0.5 model ONCE, then for each probe spec (single adapter or a merge recipe)
applies the LoRA delta(s) to the weights in place, runs the whole observation bank through
`LiberoPolicy.infer` with a FIXED noise (deterministic flow sampling), captures mean-pooled
activations at selected layers via forward hooks, and restores pristine weights afterwards.

Signatures -> results/analysis/act_sigs/<name>.npz  (per hook layer: [n_obs, hidden] fp16)

Usage:
  .venv/bin/python sweep/activation_probe.py --tasks all            # 76 solo signatures
  .venv/bin/python sweep/activation_probe.py --tasks 0,1,2          # smoke test
  .venv/bin/python sweep/activation_probe.py --merges 26:34,24:69   # merged-policy signatures
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import apply_lora as al
import safetensors.torch as st
from openpi.runtime import LiberoPolicy

BASE_DIR = "/media/pinas/Chris/models/pi05_base_pytorch"
NORM = str(REPO / "results/analysis/norm_assets")
HOOKS = {
    "vision": "paligemma_with_expert.paligemma.model.vision_tower",
    "llm_00": "paligemma_with_expert.paligemma.model.language_model.layers.0",
    "llm_05": "paligemma_with_expert.paligemma.model.language_model.layers.5",
    "llm_11": "paligemma_with_expert.paligemma.model.language_model.layers.11",
    "llm_17": "paligemma_with_expert.paligemma.model.language_model.layers.17",
    "exp_00": "paligemma_with_expert.gemma_expert.model.layers.0",
    "exp_08": "paligemma_with_expert.gemma_expert.model.layers.8",
    "exp_17": "paligemma_with_expert.gemma_expert.model.layers.17",
    "act_out": "action_out_proj",
}


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            t = first_tensor(e)
            if t is not None:
                return t
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="")
    ap.add_argument("--merges", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plan", default="sweep/plan.json",
                    help="adapter plan; use plan106.json to include the S16-init experts (76-105)")
    ap.add_argument("--bank", default="results/analysis/obs_bank.npz", help="observation bank npz")
    ap.add_argument("--base", default="", help="EXP-82: run the BASE policy (no adapter) once, save under this name")
    ap.add_argument("--outdir", default="results/analysis/act_sigs")
    args = ap.parse_args()

    plan = json.load(open(REPO / args.plan))["adapters"]
    afile = {a["task_index"]: a["adapter_file"] for a in plan}
    specs = []
    if args.tasks:
        ts = sorted(afile) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
        specs += [(f"task{t}", [t]) for t in ts]
    if args.base:
        specs += [(args.base, [])]
    if args.merges:
        for g in args.merges.split(","):
            grp = [int(x) for x in g.split(":")]
            specs += [("merge_" + "_".join(map(str, grp)), grp)]

    bank = np.load(REPO / args.bank, allow_pickle=True)
    n_obs = len(bank["state"])
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((10, 32)).astype(np.float32)

    import time
    t0 = time.time()
    policy = LiberoPolicy(BASE_DIR, norm_stats_path=NORM, pytorch_device=args.device)
    model = policy._model
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
    # 3 denoise steps instead of 10: signatures only need cross-adapter consistency, not
    # action quality; same fixed noise + same step count = comparable and ~3x faster.
    _orig_sample = model.sample_actions
    model.sample_actions = lambda device, obs, **kw: _orig_sample(device, obs, **{"num_steps": 3, **kw})
    params = dict(model.named_parameters())

    # LoRA stem -> model parameter, via the base state-dict key mapping
    probe_state = st.load_file(afile[0])
    stems = [s for s, g in al.extract_lora_groups(probe_state).items() if g.a is not None and g.b is not None]
    sd = model.state_dict(keep_vars=True)
    stem2param = {}
    skipped = []
    for s_ in stems:
        k = al.resolve_base_key({k: None for k in sd.keys()}, s_)
        if k is None:
            raise KeyError(f"no model tensor for stem {s_}")
        if k not in params:
            # tied tensor (paligemma lm_head shares storage with embed_tokens); lm_head is unused
            # at VLA inference, and mutating the tied storage would corrupt the embeddings -> skip.
            skipped.append(s_)
            continue
        stem2param[s_] = k
    stems = [s_ for s_ in stems if s_ in stem2param]
    if skipped:
        print(f"skipping {len(skipped)} tied/unused stems: {[s.split('.')[-1] for s in skipped]}", flush=True)
    pristine = {k: params[k].detach().clone().cpu() for k in set(stem2param.values())}
    print(f"{len(stem2param)} target modules; pristine copy "
          f"{sum(v.numel()*v.element_size() for v in pristine.values())/1e9:.1f} GB", flush=True)

    acts = {}
    handles = []
    for tag, modname in HOOKS.items():
        mod = dict(model.named_modules())[modname]
        def mk(tag):
            def hook(_m, _i, out):
                t = first_tensor(out)
                if t is not None:
                    acts[tag] = t.float().mean(dim=tuple(range(t.ndim - 1))).detach().cpu()
            return hook
        handles.append(mod.register_forward_hook(mk(tag)))

    def apply_group(grp):
        if not grp:
            return                                   # --base: pristine weights, nothing to add
        groups = {t: al.extract_lora_groups(st.load_file(afile[t])) for t in grp}
        with torch.no_grad():
            for s_ in stems:
                pk = stem2param[s_]
                p = params[pk]
                acc = None
                for t in grp:
                    g = groups[t][s_]
                    sc = (g.alpha / al.infer_rank(g.a)) if g.alpha is not None else 1.0
                    d = sc * al.compute_delta(p, g.a.to(torch.float32), g.b.to(torch.float32))
                    acc = d if acc is None else acc + d
                p.add_((acc / len(grp)).to(p.dtype).to(p.device))

    def restore():
        with torch.no_grad():
            for k, v in pristine.items():
                params[k].copy_(v.to(params[k].device))

    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    import time
    for name, grp in specs:
        out = outdir / f"{name}.npz"
        if out.exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        t0 = time.time()
        apply_group(grp)
        t1 = time.time()
        sig = {tag: [] for tag in HOOKS}
        for i in range(n_obs):
            obs = {"observation/image": bank["img"][i], "observation/wrist_image": bank["wrist"][i],
                   "observation/state": bank["state"][i], "prompt": str(bank["prompt"][i])}
            policy.infer(obs, noise=noise)
            for tag in HOOKS:
                sig[tag].append(acts[tag].numpy().astype(np.float16))
            if i == 9:
                print(f"  [{name}] apply {t1-t0:.0f}s; {(time.time()-t1)/10:.2f}s/obs "
                      f"-> est {(time.time()-t1)/10*n_obs:.0f}s/adapter", flush=True)
        restore()
        np.savez_compressed(out, **{k: np.stack(v) for k, v in sig.items()},
                            scene=bank["scene"], prompt=bank["prompt"])
        print(f"{name}: {n_obs} obs in {time.time()-t0:.0f}s", flush=True)
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
