#!/usr/bin/env python3
"""EXP-36 — train NEW LoRA experts to enlarge the fleet.

The learning curve is the reason this exists. Every attempt to learn donor selection has failed, and
the measured requirement is ~140 training targets against our 76, with a slope of 0.018pp/target.
More features, more labels, more rollout budget and a different representation have each been tested
and rejected; the ONLY axis that moves the constraint is more real tasks.

The material is on the NAS: libero_goal / libero_object / libero_spatial each hold 10 tasks with
demonstrations, none of which our fleet covers. Training those 30 would take the pool from 76 to ~106.

RECIPE — copied exactly from the adapters we already have, so new experts are interchangeable with the
old ones (same 458 stems, same rank, same scale). Anything else would make the pool heterogeneous and
quietly invalidate every comparison that treats adapters as exchangeable:
    r=16, lora_alpha=16, lora_dropout=0.0, init_lora_weights="gaussian",
    target_modules = the 17 names in the fleet's adapter_config.json

LOSS — the model's own training forward: PI0Pytorch.forward(observation, actions) implements the
flow-matching objective, so nothing needs reimplementing. Only the LoRA parameters get gradients.

DATA — hdf5 demos; each sample is one timestep's observation plus the next `action_horizon` actions,
padded from LIBERO's 7 dims to the model's 32. Preprocessing mirrors the eval harness exactly
(180-degree image rotation, resize_with_pad to 224, 8-dim state); a mismatch here would train an
adapter that cannot be evaluated by the same pipeline.

VALIDATE BEFORE SCALING. This script trains ONE task by default. It is only worth training 30 if the
first adapter (a) trains, (b) loads through our merge machinery with all 458 stems, and (c) actually
performs on its task. Committing days of GPU before checking those is how you discover on day three
that the adapters are unusable.
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
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

DEMOS = pathlib.Path("/media/pinas/palma/inspire_4suite_regenerated_hdf5")
BASE = "/media/pinas/Chris/models/pi05_base_pytorch"
RESIZE = 224
REF_CFG = "/media/pinas/mehdi/libero90_lora/task0/adapter_config.json"



def _clear_orphaned_builds():
    """Adapters in decomp_adapters/ are build->eval->delete transients. If no orchestrator is running,
    any that exist were abandoned by a killed run -- and 30+ of them is a blown disk quota (happened
    twice). Clear them at startup rather than trusting every kill path to clean up."""
    import subprocess as _sp
    if _sp.run(["pgrep", "-f", "merge_orchestrat[o]r.py"], capture_output=True).returncode != 0:
        for _f in (REPO / "results/analysis/decomp_adapters").glob("*.safetensors"):
            _f.unlink(missing_ok=True)


_clear_orphaned_builds()

def lora_recipe():
    """The fleet's own LoRA configuration, so new experts are drop-in compatible."""
    c = json.load(open(REF_CFG))
    return dict(r=c["r"], lora_alpha=c["lora_alpha"], lora_dropout=c.get("lora_dropout", 0.0),
                target_modules=list(c["target_modules"]),
                init_lora_weights=c.get("init_lora_weights", "gaussian"))


def load_demo_samples(suite, task, n_demos=None, stride=2):
    """[(obs_dict, action_chunk)] from one task's hdf5, preprocessed exactly as the eval harness does."""
    import h5py
    from openpi_client import image_tools
    from openpi.runtime import Pi05LiberoConfig
    H = Pi05LiberoConfig().action_horizon
    ADIM = Pi05LiberoConfig().action_dim
    p = DEMOS / suite / f"{task}_demo.hdf5"
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    with h5py.File(p, "r") as h:
        d = h["data"] if "data" in h else h
        keys = sorted(d.keys())[:n_demos] if n_demos else sorted(d.keys())
        for k in keys:
            g = d[k]
            acts = np.asarray(g["actions"], dtype=np.float32)
            T = acts.shape[0]
            for t in range(0, T - H, stride):
                img = np.ascontiguousarray(g["obs/agentview_rgb"][t][::-1, ::-1])
                wri = np.ascontiguousarray(g["obs/eye_in_hand_rgb"][t][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE))
                state = np.concatenate((g["obs/ee_pos"][t], g["obs/ee_ori"][t],
                                        g["obs/gripper_states"][t])).astype(np.float32)
                chunk = np.zeros((H, ADIM), dtype=np.float32)
                chunk[:, :acts.shape[1]] = acts[t:t + H]
                out.append(({"observation/image": img, "observation/wrist_image": wri,
                             "observation/state": state}, chunk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal_no_noops")
    ap.add_argument("--task", default="put_the_bowl_on_the_stove")
    ap.add_argument("--prompt", default=None, help="defaults to the task name with underscores removed")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_demos", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    prompt = a.prompt or a.task.replace("_", " ")
    out = pathlib.Path(a.out or (REPO / f"results/analysis/new_loras/{a.suite}__{a.task}"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] {a.suite}/{a.task}", flush=True)
    samples = load_demo_samples(a.suite, a.task, a.n_demos)
    print(f"[data] {len(samples)} training samples  (chunk {samples[0][1].shape})", flush=True)

    if a.device is None:
        import subprocess
        used = [int(x) for x in subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()]
        a.device = f"cuda:{int(np.argmin(used))}"
    print(f"[model] loading base on {a.device}", flush=True)

    from openpi.runtime import LiberoPolicy
    from peft import LoraConfig, get_peft_model
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"),
                          pytorch_device=a.device)
    model = policy._model
    rec = lora_recipe()
    print(f"[lora] {rec['target_modules']}  r={rec['r']} alpha={rec['lora_alpha']}", flush=True)
    model = get_peft_model(model, LoraConfig(**rec))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_tr/1e6:.1f}M / {n_all/1e9:.2f}B  ({100*n_tr/n_all:.3f}%)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    rng = np.random.default_rng(0)
    losses = []
    for step in range(a.steps):
        idx = rng.choice(len(samples), a.batch, replace=False)
        obs_b = [samples[i][0] for i in idx]
        act_b = torch.tensor(np.stack([samples[i][1] for i in idx])).to(a.device)
        observation = policy.batch_observation(obs_b, prompt) if hasattr(policy, "batch_observation") \
            else _observation_from(policy, obs_b, prompt, a.device)
        loss = model(observation, act_b)
        loss = loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 50 == 0 or step == a.steps - 1:
            print(f"[train] step {step}/{a.steps}  loss {np.mean(losses[-50:]):.4f}", flush=True)
    model.save_pretrained(str(out))
    json.dump({"suite": a.suite, "task": a.task, "prompt": prompt, "steps": a.steps,
               "final_loss": float(np.mean(losses[-50:]))}, open(out / "train_meta.json", "w"), indent=1)
    print(f"[done] saved to {out}", flush=True)


def _observation_from(policy, obs_list, prompt, device):
    """Build a batched Observation the same way LiberoPolicy.infer builds a single one."""
    from openpi.runtime import Observation, normalize_inputs, Pi05LiberoConfig
    batch = {}
    per = []
    for o in obs_list:
        inp = policy._input_transform(dict(o, prompt=prompt))
        inp = normalize_inputs(inp, policy._norm_stats)
        tok, msk = policy._tokenizer.tokenize(str(inp.pop("prompt")))
        inp["tokenized_prompt"] = tok
        inp["tokenized_prompt_mask"] = msk
        inp["state"] = np.pad(inp["state"], (0, Pi05LiberoConfig().action_dim - inp["state"].shape[-1]))
        per.append(inp)
    for k in per[0]:
        if isinstance(per[0][k], dict):
            batch[k] = {kk: torch.from_numpy(np.stack([np.asarray(p[k][kk]) for p in per])).to(device)
                        for kk in per[0][k]}
        else:
            batch[k] = torch.from_numpy(np.stack([np.asarray(p[k]) for p in per])).to(device)
    return Observation.from_dict(batch)


if __name__ == "__main__":
    main()
