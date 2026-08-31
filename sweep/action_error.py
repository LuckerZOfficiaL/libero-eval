#!/usr/bin/env python3
"""PHASE 1 — rollout-free labels: action-prediction error on demonstrations.

Borrowed from Text-to-LoRA's SFT branch: instead of scoring a merged adapter by *rolling it out*, apply
it and take the model's loss on task data. Here the "loss" is open-loop action-prediction error against
the demonstrations the LoRA was trained on.

WHY THIS IS WORTH TRYING (the argument changed after EXP-22.4). Originally the case was "labels become
cheap". The stronger case is that they become *cleaner*: a 10-rollout probe has test-retest reliability
rho = 0.41 and ~13pp of noise, because a rollout is a stochastic, chaotic, closed-loop process. An
action-error label is a DETERMINISTIC forward pass -- zero measurement noise. Even if it correlates
imperfectly with true affinity, it can carry more usable signal per unit of compute than a noisy probe.

THE RISK, STATED UP FRONT. This project's recurring finding is that merge damage is a CLOSED-LOOP
phenomenon: tiny per-step drift amplified over a rollout. Open-loop measures have failed before
(activation distance, within-target r = -0.043). Action error may fail the same way. It differs in being
TASK-GROUNDED -- "does this merged policy still produce the right action for THIS task" rather than "are
these two adapters similar" -- but that is a hypothesis, and Phase 1 exists to kill it cheaply.

VALIDATION BEFORE SCALING. The pipeline has a built-in correctness check: the target's OWN adapter must
score much lower action error on its own demos than the un-adapted base model. If it does not, the
observation preprocessing is wrong (image orientation, state layout) and every downstream number would be
silent garbage.
"""
import json
import os
import pathlib
import sys

# The P6000s are CUDA capability 6.1, below Triton's 7.0 minimum, so torch.compile explodes on
# sample_actions. merge_orchestrator.py sets this for every policy server it launches; standalone
# scripts must set it too, BEFORE torch is imported.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

REPO = pathlib.Path("/home/lzhou/libero-eval")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import apply_lora as al
import safetensors.torch as st
from openpi_client import image_tools

DEMOS = pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5/libero_90_no_noops")
RESIZE = 224
plan = json.load(open(REPO / "sweep/plan.json"))["adapters"]
afile = {a["task_index"]: a["adapter_file"] for a in plan}
scenes = {a["task_index"]: a["scenes"] for a in plan}
lang = {a["task_index"]: a["language"] for a in plan}


def scene_to_hdf5(scene):
    """plan.json scene names correspond 1:1 to '<SCENE>_demo.hdf5' in the LIBERO-90 demo dump."""
    p = DEMOS / f"{scene}_demo.hdf5"
    return p if p.exists() else None


def load_obs(scene, n_demos=3, per_demo=8, seed=0):
    """Return a list of observation dicts + the ground-truth action at each of those timesteps.

    Preprocessing mirrors examples/libero/main.py exactly: 180-degree image rotation, resize_with_pad to
    224, and state = concat(ee_pos, ee_ori(axis-angle), gripper). Any deviation here silently corrupts
    every label, which is what the base-vs-adapter check below is for.
    """
    import h5py
    p = scene_to_hdf5(scene)
    if p is None:
        return [], []
    rng = np.random.default_rng(seed)
    out_obs, out_act = [], []
    with h5py.File(p, "r") as h:
        d = h["data"] if "data" in h else h
        keys = sorted(d.keys())[:n_demos]
        for k in keys:
            g = d[k]
            T = g["actions"].shape[0]
            idx = np.sort(rng.choice(T, min(per_demo, T), replace=False))
            for t in idx:
                img = np.ascontiguousarray(g["obs/agentview_rgb"][t][::-1, ::-1])
                wri = np.ascontiguousarray(g["obs/eye_in_hand_rgb"][t][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE))
                state = np.concatenate((g["obs/ee_pos"][t], g["obs/ee_ori"][t],
                                        g["obs/gripper_states"][t]))
                out_obs.append({"observation/image": img, "observation/wrist_image": wri,
                                "observation/state": state})
                out_act.append(np.asarray(g["actions"][t], dtype=np.float32))
    return out_obs, out_act


def fold(path):
    out = {}
    for stem, g in al.extract_lora_groups(st.load_file(str(path))).items():
        if g.a is None or g.b is None:
            continue
        r = al.infer_rank(g.a)
        s = (g.alpha / r) if g.alpha is not None else 1.0
        a, b = g.a.float(), g.b.float()
        if b.shape[1] != a.shape[0]:
            a, b = b.T.contiguous(), a.T.contiguous()
        out[stem] = (b * s, a)
    return out


def action_error(policy, obs_list, act_list, prompt, noise=None):
    """Mean per-dimension L1 between the policy's first predicted action and the demo action."""
    errs = []
    for o, a in zip(obs_list, act_list):
        el = dict(o); el["prompt"] = prompt
        pred = policy.infer(el, noise=noise)["actions"]
        pred = np.asarray(pred)
        first = pred[0] if pred.ndim == 2 else pred
        errs.append(float(np.abs(first[:len(a)] - a).mean()))
    return float(np.mean(errs)) if errs else float("nan")


class Applier:
    """Apply / restore LoRA deltas in place on a loaded policy (same pattern as activation_probe.py)."""

    def __init__(self, policy, ref_adapter):
        self.model = policy._model
        self.params = dict(self.model.named_parameters())
        sd = self.model.state_dict(keep_vars=True)
        groups = al.extract_lora_groups(st.load_file(str(ref_adapter)))
        self.stem2param, skipped = {}, []
        for s_ in [s for s, g in groups.items() if g.a is not None and g.b is not None]:
            k = al.resolve_base_key({k: None for k in sd.keys()}, s_)
            if k is None or k not in self.params:
                skipped.append(s_)          # tied lm_head shares storage with embeddings -> never touch
                continue
            self.stem2param[s_] = k
        self.pristine = {k: self.params[k].detach().clone().cpu()
                         for k in set(self.stem2param.values())}
        print(f"  {len(self.stem2param)} modules applied, {len(skipped)} tied/unused skipped", flush=True)

    def apply(self, folds_with_coeff):
        """folds_with_coeff: list of (fold, coefficient). Deltas are summed, matching our operator."""
        with torch.no_grad():
            for s_, pk in self.stem2param.items():
                p = self.params[pk]
                acc = None
                for f, c in folds_with_coeff:
                    if s_ not in f:
                        continue
                    b, a = f[s_]
                    d = c * al.compute_delta(p, a.to(torch.float32), b.to(torch.float32))
                    acc = d if acc is None else acc + d
                if acc is not None:
                    p.add_(acc.to(p.dtype).to(p.device))

    def restore(self):
        with torch.no_grad():
            for k, v in self.pristine.items():
                self.params[k].copy_(v.to(self.params[k].device))


if __name__ == "__main__":
    # CORRECTNESS CHECK: the task's own adapter must beat the un-adapted base model on its own demos.
    # If it does not, preprocessing is wrong and every downstream label would be silent garbage.
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sc = scenes[t][0]
    obs, act = load_obs(sc, n_demos=2, per_demo=6)
    print(f"task t{t}  scene {sc}")
    print(f"  demo timesteps: {len(obs)}   action dim {act[0].shape}   "
          f"state dim {obs[0]['observation/state'].shape}   image {obs[0]['observation/image'].shape}")
    print(f"  prompt: {lang[t]!r}")
    if not obs:
        sys.exit("no demo file")
    from openpi.runtime import LiberoPolicy
    BASE = "/media/pinas/Chris/models/pi05_base_pytorch"
    NORM = str(REPO / "results/analysis/norm_assets")
    print("  loading policy ...", flush=True)
    # NOTE: torch device ordering != nvidia-smi ordering here. nvidia-smi 0 is the 2GB K620 display card,
    # but torch sees the two P6000s as cuda:0/cuda:1 and the K620 as cuda:2. Pick the emptier P6000.
    import subprocess as _sp
    used = [int(x) for x in _sp.run(["nvidia-smi", "--query-gpu=memory.used",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True).stdout.split()]
    dev = "cuda:0" if used[1] <= used[2] else "cuda:1"   # nvidia-smi 1,2 == torch 0,1
    print(f"  using {dev} (nvidia-smi used: {used})", flush=True)
    policy = LiberoPolicy(BASE, norm_stats_path=NORM, pytorch_device=dev)
    ap = Applier(policy, afile[t])
    g = torch.Generator(device="cpu").manual_seed(0)
    noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)
    e_base = action_error(policy, obs, act, lang[t], noise=noise)
    ap.apply([(fold(afile[t]), 1.0)])
    e_own = action_error(policy, obs, act, lang[t], noise=noise)
    ap.restore()
    e_back = action_error(policy, obs, act, lang[t], noise=noise)
    print(f"\n  base model            : {e_base:.4f}")
    print(f"  + task's own LoRA     : {e_own:.4f}   ({(e_base-e_own)/e_base*100:+.1f}% vs base)")
    print(f"  after restore         : {e_back:.4f}   (must match base: "
          f"{'OK' if abs(e_back-e_base) < 1e-6 else 'RESTORE BROKEN'})")
    print(f"\n  VERDICT: {'pipeline is sound' if e_own < e_base else 'PREPROCESSING WRONG — do not use'}")


def generate_labels(pairs, n_demos=3, per_demo=8, out=None, device=None):
    """Action error for W = S_16-equivalent full merge: target delta + ALPHA*donor delta.

    NOTE ON THE OPERATOR. Rollout probes used W = S_16 + r_target + 0.5*r_donor. Applied to the base
    weights, S_16 + r_target is exactly the target's own delta reconstructed, so the merged weight is
    base + D_target + 0.5*r_donor. Here we apply base + D_target + 0.5*D_donor instead: the donor's
    SHARED component is included rather than projected out. Within a target that is a constant offset
    plus the donor-specific part, so donor RANKING -- the only thing we use -- is preserved, and it
    avoids reconstructing the soup basis per pair.
    """
    import collections
    from openpi.runtime import LiberoPolicy
    BASE = "/media/pinas/Chris/models/pi05_base_pytorch"
    NORM = str(REPO / "results/analysis/norm_assets")
    if device is None:
        import subprocess as sp
        used = [int(x) for x in sp.run(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True).stdout.split()]
        device = "cuda:0" if used[1] <= used[2] else "cuda:1"
    print(f"[labels] device {device}; {len(pairs)} pairs", flush=True)
    policy = LiberoPolicy(BASE, norm_stats_path=NORM, pytorch_device=device)
    ap = Applier(policy, afile[0])
    noise = np.random.default_rng(0).standard_normal((10, 32)).astype(np.float32)
    obs_cache, fold_cache, res = {}, {}, {}
    if out and pathlib.Path(out).exists():
        res = json.load(open(out))
    def get_obs(t):
        if t not in obs_cache:
            o, a = load_obs(scenes[t][0], n_demos=n_demos, per_demo=per_demo)
            obs_cache[t] = (o, a)
        return obs_cache[t]
    def get_fold(t):
        if t not in fold_cache:
            if len(fold_cache) > 12:
                fold_cache.clear()
            fold_cache[t] = fold(afile[t])
        return fold_cache[t]
    import time
    t0 = time.time()
    for i, (T, D) in enumerate(pairs):
        key = f"{T}_{D}" if D is not None else f"{T}_solo"
        if key in res:
            continue
        o, a = get_obs(T)
        if not o:
            continue
        terms = [(get_fold(T), 1.0)] + ([(get_fold(D), 0.5)] if D is not None else [])
        ap.apply(terms)
        res[key] = action_error(policy, o, a, lang[T], noise=noise)
        ap.restore()
        if out and (i % 10 == 0 or i == len(pairs) - 1):
            json.dump(res, open(out, "w"))
            el = time.time() - t0
            print(f"[labels] {i+1}/{len(pairs)}  {el/(i+1):.1f}s/pair  "
                  f"eta {(len(pairs)-i-1)*el/(i+1)/60:.0f}min", flush=True)
    if out:
        json.dump(res, open(out, "w"))
    return res
