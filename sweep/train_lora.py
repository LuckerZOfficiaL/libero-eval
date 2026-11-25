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
import re
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
RAW90 = pathlib.Path("/media/pinas/datasets/LIBERO/libero_90")                # original LIBERO-90 demos (fallback, see load_demo_samples)
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


# NOTE: not called at import time. A chunked driver builds its adapters BEFORE its orchestrator
# starts, so an import from an analysis script in that window deleted two live probe files
# (EXP-32, 09-04 21:48). Drivers call it from main() only.

def lora_recipe():
    """The fleet's own LoRA configuration, so new experts are drop-in compatible."""
    c = json.load(open(REF_CFG))
    return dict(r=c["r"], lora_alpha=c["lora_alpha"], lora_dropout=c.get("lora_dropout", 0.0),
                target_modules=list(c["target_modules"]),
                init_lora_weights=c.get("init_lora_weights", "gaussian"))


def load_demo_samples(suite, task, n_demos=None, stride=2):
    """[(obs_dict, action_chunk)] from one task's hdf5, preprocessed exactly as the eval harness does.

    Action targets MUST be quantile-normalized into [-1,1]: at inference the model's output is pushed
    through unnormalize_outputs, so a model trained on raw actions gets unnormalized twice and commands
    far outside the action range. This exact bug produced 0/25 on the first trial adapter while an
    unrelated fleet adapter scored 20/25 on the same task.
    """
    import h5py
    from openpi_client import image_tools
    from openpi.runtime import Pi05LiberoConfig
    H = Pi05LiberoConfig().action_horizon
    ADIM = Pi05LiberoConfig().action_dim
    ns = json.load(open(REPO / "results/analysis/norm_assets/norm_stats.json"))
    st_a = ns.get("norm_stats", ns)["actions"]
    q01, q99 = np.asarray(st_a["q01"], dtype=np.float32), np.asarray(st_a["q99"], dtype=np.float32)
    p, noop = DEMOS / suite / f"{task}_demo.hdf5", False
    if not p.exists() and suite == "libero_90_no_noops":                # 11-07 EXP-150: LIVING_ROOM_SCENE2 butter -> basket is missing
        p, noop = RAW90 / f"{task}_demo.hdf5", True                     # from the regenerated set: original LIBERO demos (128 px), no-ops dropped
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    with h5py.File(p, "r") as h:
        d = h["data"] if "data" in h else h
        keys = sorted(d.keys())[:n_demos] if n_demos else sorted(d.keys())
        for k in keys:
            g = d[k]
            acts = np.asarray(g["actions"], dtype=np.float32)
            idx = np.arange(len(acts))
            if noop:                                                    # OpenVLA's no-op filter: no motion and the gripper unchanged
                idx = np.array([t for t in idx if not (t > 0 and np.linalg.norm(acts[t, :-1]) < 1e-4 and acts[t, -1] == acts[t - 1, -1])])
                acts = acts[idx]
            k = q01.shape[-1]
            acts[:, :k] = 2.0 * (acts[:, :k] - q01) / (q99 - q01 + 1e-6) - 1.0
            T = acts.shape[0]
            for t in range(0, T - H, stride):
                img = np.ascontiguousarray(g["obs/agentview_rgb"][idx[t]][::-1, ::-1])
                wri = np.ascontiguousarray(g["obs/eye_in_hand_rgb"][idx[t]][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
                wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE))
                state = np.concatenate((g["obs/ee_pos"][idx[t]], g["obs/ee_ori"][idx[t]],
                                        g["obs/gripper_states"][idx[t]])).astype(np.float32)
                chunk = np.zeros((H, ADIM), dtype=np.float32)
                chunk[:, :acts.shape[1]] = acts[t:t + H]
                out.append(({"observation/image": img, "observation/wrist_image": wri,
                             "observation/state": state}, chunk))
    return out


def load_rollout_samples(pattern, stride=2):
    """[(obs_dict, action_chunk)] from rollout dumps (examples/libero/main_dense.py --args.dump-stride k: frames every k
    steps as served to the policy, + every executed action), e.g. a skill's own successful episodes (EXP-144). Same
    format and action normalisation as load_demo_samples; frames whose next H executed actions are not all recorded
    are skipped. stride must be a multiple of the dump stride (frames exist only there)."""
    import glob
    from openpi.runtime import Pi05LiberoConfig
    H, ADIM = Pi05LiberoConfig().action_horizon, Pi05LiberoConfig().action_dim
    ns = json.load(open(REPO / "results/analysis/norm_assets/norm_stats.json"))
    st_a = ns.get("norm_stats", ns)["actions"]
    q01, q99 = np.asarray(st_a["q01"], dtype=np.float32), np.asarray(st_a["q99"], dtype=np.float32)
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    out = []
    for f in fs:
        # 10-26: read each array ONCE per file — z[key] re-reads the npz member on every access, and every kept frame
        # pinned its own full copy of the episode's images (~60 GB RSS and ~10 min load for ~2,700 samples)
        z = np.load(f); img, wri, stt, steps = z["img"], z["wrist"], z["state"], z["step"]
        if "chunk" in z.files:                                         # 11-07 EXP-155: expert-relabeled frames, one chunk each
            ch = z["chunk"].astype(np.float32)[:, :H]; k = q01.shape[-1]
            ch[..., :k] = 2.0 * (ch[..., :k] - q01) / (q99 - q01 + 1e-6) - 1.0
            for j in range(len(steps)):
                chunk = np.zeros((H, ADIM), dtype=np.float32); chunk[:, :ch.shape[-1]] = ch[j]
                out.append(({"observation/image": img[j], "observation/wrist_image": wri[j],
                             "observation/state": stt[j].astype(np.float32)}, chunk))
            continue
        acts = z["act"].astype(np.float32); k = q01.shape[-1]
        acts[:, :k] = 2.0 * (acts[:, :k] - q01) / (q99 - q01 + 1e-6) - 1.0
        pos = {int(s): i for i, s in enumerate(z["act_step"])}; st0 = int(steps[0])
        for j, s in enumerate(steps):
            i = pos.get(int(s))
            if i is None or i + H > len(acts) or (int(s) - st0) % stride: continue
            chunk = np.zeros((H, ADIM), dtype=np.float32); chunk[:, :acts.shape[1]] = acts[i:i + H]
            out.append(({"observation/image": img[j], "observation/wrist_image": wri[j],
                         "observation/state": stt[j].astype(np.float32)}, chunk))
    print(f"[data] {len(out)} rollout samples from {len(fs)} episodes ({pattern})", flush=True)
    return out


def main():
    _clear_orphaned_builds()
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
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds data order and the flow-matching noise/time draws; S16 init is deterministic")
    ap.add_argument("--wp_lambda", type=float, default=1.0, help="s16soft: weight of the soft write-protection penalty")
    ap.add_argument("--tasks_file", default=None, help="EXP-136 joint training: file of 'suite task' lines; each step draws one "
                                                        "task uniformly and a batch from its demos (with its own prompt); "
                                                        "EXP-147: an optional third field = rollout glob, used instead of the demos")
    ap.add_argument("--rank", type=int, default=None, help="override the LoRA rank (alpha set equal -> scale 1, like the fleet)")
    ap.add_argument("--stride", type=int, default=2, help="frame stride when reading demos (2 = previous behaviour)")
    ap.add_argument("--rollouts", default=None, help="EXP-144: glob of rollout dumps (ep*_success.npz) to train on INSTEAD "
                         "of the task's demos (--task / --prompt still name the task)")
    ap.add_argument("--context", default=None, help="EXP-145 (--init ctx): JSON {members: [[adapter path, weight], ...], w, "
                         "init_adapter[, init_zero]}: the merge of the OTHER members (S16 + sum weight * residual, as PBuilder builds it) is "
                         "folded into the frozen base; the trained adapter's private part acts at weight w, starting from init_adapter's")
    ap.add_argument("--replay_tasks", default=None, help="EXP-145: file of 'suite task' lines (the library's own demos) mixed into "
                         "training with probability --replay_frac per step (each step one task, its own prompt)")
    ap.add_argument("--replay_frac", type=float, default=0.5)
    ap.add_argument("--replay_demos", type=int, default=5, help="demos per replay task")
    ap.add_argument("--ewc_lambda", type=float, default=0.0, help="11-18 EXP-159 online EWC: + lambda/2 * sum F (theta - theta*)^2 over "
                         "the trainable parameters; F = diagonal Fisher (squared minibatch gradients, mean-normalised), summed over rounds")
    ap.add_argument("--ewc_prev", default=None, help="the previous round's ewc.pt (F, theta*); without it F is estimated at the start "
                         "on the replay groups (the library), theta* = the initial parameters")
    ap.add_argument("--ewc_batches", type=int, default=50, help="minibatches per Fisher estimate")
    ap.add_argument("--olora_lambda", type=float, default=0.0, help="11-25 EXP-174 O-LoRA (Wang et al. 2023; official L1 form, default "
                         "0.5): + lambda * sum |A_old A^T| over the target modules, A_old = the lora_A rows of every --context member (the "
                         "previous tasks' LoRAs, folded into the base); its gradient is added after the task-gradient clip")
    ap.add_argument("--inflora", type=float, default=0.0, help="11-25 EXP-174 InfLoRA (Liang & Li 2024; official design): threshold "
                         "(lamb, official 0.95) of the old tasks' input subspace M, DualGPM criterion on the input covariance of the "
                         "--replay_tasks groups (the library); A frozen at the top-r left singular vectors of (I - M M^T) C_new / sqrt(3), "
                         "C_new = the input covariance of the skill groups; only B trains (from zero)")
    ap.add_argument("--inflora_batches", type=int, default=50, help="minibatches per input-covariance estimate")
    ap.add_argument("--inflora_rows", type=int, default=4096, help="input rows (tokens) sampled per module for each covariance")
    ap.add_argument("--init", default="zero", choices=["zero", "s16", "s16wp", "s16soft", "readout", "ctx"],
                    help="zero = peft default (B=0); s16 = start from the fleet's shared rank-16 component; "
                         "s16wp = s16 + WRITE-PROTECTED: A frozen at V^T, B's shared-output component held at U*Sigma "
                         "(only (I-UU^T)B learns -> D = S16 + B_perp V^T, never writes into the shared outputs; EXP-123); "
                         "s16soft = s16 + penalty wp_lambda * sum ||Un^T (B - U*Sigma)||^2 / sum ||U*Sigma||^2, A trainable "
                         "(soft write-protection; EXP-126); readout = S16 merged into the frozen base, LoRA A frozen at random "
                         "rows orthogonal to the shared input directions V, B from zero (reads outside the shared inputs; EXP-126; "
                         "the saved adapter EXCLUDES S16); ctx = in-context (EXP-145): A frozen at V^T, only (I-UU^T)B learns, "
                         "inside the merge given by --context (the saved adapter is S16 + B_perp V^T, like s16wp)")
    a = ap.parse_args()
    if (a.init == "ctx") != bool(a.context): raise SystemExit("--init ctx and --context go together")

    prompt = a.prompt or a.task.replace("_", " ")
    out = pathlib.Path(a.out or (REPO / f"results/analysis/new_loras/{a.suite}__{a.task}"))
    out.mkdir(parents=True, exist_ok=True)

    if a.tasks_file:                                                    # joint training over several tasks
        pairs = [ln.split() for ln in open(a.tasks_file) if ln.strip()]
        groups = []
        for su, tk, *ro in pairs:
            smp = load_rollout_samples(ro[0], stride=a.stride) if ro else load_demo_samples(su, tk, a.n_demos, stride=a.stride)
            groups.append((re.sub(r"^[A-Z_]*SCENE\d+_", "", tk).replace("_", " "), smp))   # bare instruction = eval prompt
            print(f"[data] {su}/{tk}: {len(smp)} samples", flush=True)
        print(f"[data] joint training over {len(groups)} tasks, {sum(len(g[1]) for g in groups)} samples", flush=True)
    elif a.rollouts:                                                    # EXP-144: the skill's own rollouts, no demos
        groups = [(prompt, load_rollout_samples(a.rollouts, stride=a.stride))]
    else:
        print(f"[data] {a.suite}/{a.task}", flush=True)
        groups = [(prompt, load_demo_samples(a.suite, a.task, a.n_demos, stride=a.stride))]
        print(f"[data] {len(groups[0][1])} training samples  (chunk {groups[0][1][0][1].shape})", flush=True)
    gp = None                                                           # per-step group probabilities (None = uniform)
    if a.replay_tasks:                                                  # EXP-145: the skill group(s) first, then the library replay
        n_s = len(groups)
        for su, tk, *ro in (ln.split() for ln in open(a.replay_tasks) if ln.strip()):   # 11-06: optional 3rd field = rollout glob (no demos)
            groups.append((re.sub(r"^[A-Z_]*SCENE\d+_", "", tk).replace("_", " "),
                           load_rollout_samples(ro[0], stride=a.stride) if ro else load_demo_samples(su, tk, a.replay_demos, stride=a.stride)))
        n_r = len(groups) - n_s; gp = np.array([(1 - a.replay_frac) / n_s] * n_s + [a.replay_frac / n_r] * n_r)
        print(f"[data] replay: {n_r} library tasks x {a.replay_demos} demos, {sum(len(g[1]) for g in groups[n_s:])} samples, "
              f"frac {a.replay_frac}", flush=True)

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
    if a.rank: rec["r"], rec["lora_alpha"] = a.rank, a.rank
    print(f"[lora] {rec['target_modules']}  r={rec['r']} alpha={rec['lora_alpha']}", flush=True)
    if a.init == "readout":
        fold_s16_into_base(model)
    ctx = json.load(open(a.context)) if a.context else None
    lrc = None
    if ctx and ctx.get("lrc"):                                          # 11-18 EXP-162: S16 folded, residuals with learned weights
        fold_s16_into_base(model); lrc = ResidualCoefs(model, ctx["members"], a.device)
    elif ctx and ctx["members"]: fold_context_into_base(model, ctx["members"], ctx.get("op", "ours"))   # 11-19 EXP-164 consbc: no members = base
    model = get_peft_model(model, LoraConfig(**rec))
    wp = None; pen = None; fin = None
    if ctx and ctx.get("free"):                                         # 11-05 EXP-147 baseline
        print("[ctx] free: plain LoRA (default init, A and B trainable, scale 1) on the folded merge", flush=True)
        if ctx.get("resume"):                                            # 11-07 EXP-149: continue a trained correction (round 2)
            from peft import set_peft_model_state_dict; from safetensors.torch import load_file
            r = set_peft_model_state_dict(model, load_file(ctx["resume"]))
            nb = sum(float(p.abs().sum()) for n_, p in model.named_parameters() if "lora_B" in n_)
            print(f"[ctx] resumed {ctx['resume']} (unexpected keys {len(r.unexpected_keys)}, |lora_B| {nb:.1f})", flush=True)
            if r.unexpected_keys or nb == 0: raise SystemExit("[ctx] resume did not load")
    elif ctx: wp, fin = in_context(model, ctx["init_adapter"], ctx["w"], zero=bool(ctx.get("init_zero")))
    if a.init == "s16soft":
        init_from_s16(model); pen = soft_write_penalty(model)
    if a.init == "readout":
        read_outside(model)
    if a.init in ("s16", "s16wp"):
        init_from_s16(model)
    if a.init == "s16wp":
        wp = write_protect(model)
    if lrc is not None and ctx.get("lrc_only"):                         # the residual weights alone (no LoRA correction)
        for p in model.parameters(): p.requires_grad_(False)
    def batch_loss(prompt, samples, idx):
        obs_b = [samples[i][0] for i in idx]
        act_b = torch.tensor(np.stack([samples[i][1] for i in idx])).to(a.device)
        observation = policy.batch_observation(obs_b, prompt) if hasattr(policy, "batch_observation") \
            else _observation_from(policy, obs_b, prompt, a.device)
        loss = model(observation, act_b)
        return loss.mean() if torch.is_tensor(loss) and loss.ndim > 0 else loss
    if a.inflora > 0:                                                   # 11-25 EXP-174: A designed and frozen before the optimizer
        if not a.replay_tasks: raise SystemExit("--inflora needs --replay_tasks (the library groups)")
        inflora_design(model, batch_loss, groups[n_s:], groups[:n_s], a)
    olo = olora_penalty(model, ctx["members"]) if a.olora_lambda > 0 else None   # 11-25 EXP-174
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {n_tr/1e6:.1f}M / {n_all/1e9:.2f}B  ({100*n_tr/n_all:.3f}%)" +
          (f" + {lrc.alpha.numel()} residual weights (lr {ctx.get('alpha_lr', 1e-3):g})" if lrc is not None else ""), flush=True)

    pg = [{"params": [p for p in model.parameters() if p.requires_grad]}] if n_tr else []
    if lrc is not None: pg.append({"params": [lrc.alpha], "lr": ctx.get("alpha_lr", 1e-3), "weight_decay": 0.0})
    opt = torch.optim.AdamW(pg, lr=a.lr)
    model.train()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    ewc = None
    if a.ewc_lambda > 0:                                                # 11-18 EXP-159
        if not a.replay_tasks: raise SystemExit("--ewc_lambda needs --replay_tasks (the library groups)")
        tr = {k: p for k, p in model.named_parameters() if p.requires_grad}
        if a.ewc_prev:
            e = torch.load(a.ewc_prev, map_location=a.device); F0 = {k: e["F"][k].float() for k in tr}; th0 = {k: e["theta"][k].float() for k in tr}
        else: F0 = ewc_fisher(tr, batch_loss, groups[n_s:], a); th0 = {k: p.detach().float().clone() for k, p in tr.items()}
        ewc = lambda: a.ewc_lambda / 2 * sum((F0[k] * (p.float() - th0[k]) ** 2).sum() for k, p in tr.items())
        print(f"[ewc] lambda {a.ewc_lambda:g}, F from {a.ewc_prev or 'the replay groups'} (mean {float(sum(f.sum() for f in F0.values())) / sum(f.numel() for f in F0.values()):.2f})", flush=True)
    losses = []
    for step in range(a.steps):
        if gp is not None: prompt, samples = groups[int(rng.choice(len(groups), p=gp))]
        else: prompt, samples = groups[int(rng.integers(len(groups)))] if len(groups) > 1 else groups[0]
        idx = rng.choice(len(samples), a.batch, replace=False)
        loss = batch_loss(prompt, samples, idx)
        if pen is not None: loss = loss + a.wp_lambda * pen()
        if ewc is not None: loss = loss + ewc()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if olo is not None:                                             # unclipped: at 458 modules its norm would starve the task gradient
            po = olo(); (a.olora_lambda * po).backward()
        opt.step()
        if wp is not None: wp()
        losses.append(float(loss))
        if step % 50 == 0 or step == a.steps - 1:
            print(f"[train] step {step}/{a.steps}  loss {np.mean(losses[-50:]):.4f}" + (f"  orth {float(po):.4g}" if olo is not None else ""), flush=True)
    if fin is not None: fin()
    if ewc is not None:                                                 # accumulate this round's Fisher (its own new-skill groups); bf16: mean-normalised F overflows fp16
        F1 = ewc_fisher(tr, batch_loss, groups[:n_s], a)
        torch.save({"F": {k: (F0[k] + F1[k]).bfloat16() for k in tr}, "theta": {k: p.detach().float().cpu() for k, p in tr.items()}}, out / "ewc.pt")
        print(f"[ewc] saved F (+ this round's) and theta* to {out / 'ewc.pt'}", flush=True)
    if lrc is not None: lrc.save(out / "lrc.pt")
    model.save_pretrained(str(out))
    json.dump({"suite": a.suite, "task": a.task, "prompt": groups[0][0], "steps": a.steps, "seed": a.seed, "init": a.init,
               "tasks_file": a.tasks_file, "rank": rec["r"], "stride": a.stride, "context": ctx,
               "replay_tasks": a.replay_tasks, "replay_frac": a.replay_frac if a.replay_tasks else None,
               "final_loss": float(np.mean(losses[-50:]))}, open(out / "train_meta.json", "w"), indent=1)
    print(f"[done] saved to {out}", flush=True)


class ResidualCoefs:
    """11-18 EXP-162: the merge's residual part with LEARNED weights. The frozen base already holds W0 + S16; a forward hook on every
    target Linear adds sum_i alpha[stem, i] r_i x, r_i = B_i A_i - Un (Un^T B_i)(A_i V) V^T (member i's two-sided residual, kept in
    factor form), alpha initialised at the context weights (1/k = exactly the deployed merge, MM.ours). alpha (k per stem) is the
    only new parameter; save() writes {members, alpha: {stem: [k]}} for ZBuilder privx "alpha"."""
    def __init__(self, model, members, device):
        import unified_pool_driver as U
        F = []
        for n, (p, c) in enumerate(members):
            U.EXTPATH[f"lrc:{n}"] = p; F.append(U.fold(f"lrc:{n}"))
        self.members = [p for p, _ in members]; self.c0 = torch.tensor([float(c) for _, c in members])
        self.stems, self.idx, hooked = [], [], []
        for stem in U.USIG:
            idx = [i for i, f in enumerate(F) if stem in f]
            if not idx: continue
            try: mod = model.get_submodule(stem.split("base_model.model.", 1)[-1])
            except AttributeError: continue
            Un = U.USIG[stem].float(); Un = Un / torch.linalg.norm(Un, dim=0).clamp(min=1e-12); V = U.VS[stem].float()
            Bs = [F[i][stem][0].float() for i in idx]; As = [F[i][stem][1].float() for i in idx]
            Cs = torch.stack([(Un.T @ B) @ (A @ V) for B, A in zip(Bs, As)]); Bc, Ac = torch.cat(Bs, 1), torch.cat(As, 0)
            if (Bc.shape[0], Ac.shape[1]) != tuple(mod.weight.shape):          # stored transposed: dW = (B A)^T
                if (Ac.shape[1], Bc.shape[0]) != tuple(mod.weight.shape): raise RuntimeError(f"residual shape mismatch at {stem}")
                Bc, Ac, Un, V, Cs = Ac.T, Bc.T, V, Un, Cs.transpose(1, 2)
            dt = mod.weight.dtype; rk = Bs[0].shape[1]
            hooked.append((mod, len(self.stems), torch.tensor(idx, device=device), Bc.to(device, dt), Ac.to(device, dt),
                           Un.to(device, dt), V.to(device, dt), Cs.to(device), rk))
            self.stems.append(stem); self.idx.append(idx)
        self.alpha = torch.nn.Parameter(self.c0[None].repeat(len(self.stems), 1).to(device))
        for mod, j, ix, Bc, Ac, Un, V, Cs, rk in hooked:
            def hook(m_, inp, out, j=j, ix=ix, Bc=Bc, Ac=Ac, Un=Un, V=V, Cs=Cs, rk=rk):
                x = inp[0]; w = self.alpha[j, ix]
                h = (x @ Ac.T) * w.repeat_interleave(rk).to(x.dtype)
                M = (w[:, None, None] * Cs).sum(0).to(x.dtype)
                return out + h @ Bc.T - ((x @ V) @ M.T) @ Un.T
            mod.register_forward_hook(hook)
        print(f"[lrc] {len(members)} members' residuals hooked into {len(self.stems)} modules, weights from {sorted({round(float(c), 4) for c in self.c0})}", flush=True)

    def save(self, path):
        al = self.alpha.detach().float().cpu()
        torch.save({"members": self.members, "alpha": {s: al[j].clone() for j, s in enumerate(self.stems)}}, path)   # members absent at a stem keep c0
        d = al - self.c0[None]
        print(f"[lrc] saved {path}: alpha mean {float(al.mean()):.4f} (init {float(self.c0.mean()):.4f}), |change| mean {float(d.abs().mean()):.4f} max {float(d.abs().max()):.4f}", flush=True)


def ewc_fisher(tr, batch_loss, groups, a):
    """11-18 EXP-159: diagonal empirical Fisher of the trainable parameters tr = mean squared minibatch gradient over
    a.ewc_batches batches (group uniform, batch a.batch), normalised to mean 1 over all entries (so --ewc_lambda is scale-free)"""
    F = {k: torch.zeros_like(p, dtype=torch.float32) for k, p in tr.items()}; rng = np.random.default_rng(a.seed + 1)
    for _ in range(a.ewc_batches):
        prompt, samples = groups[int(rng.integers(len(groups)))]
        for p in tr.values(): p.grad = None
        batch_loss(prompt, samples, rng.choice(len(samples), a.batch, replace=False)).backward()
        for k, p in tr.items():
            if p.grad is not None: F[k] += p.grad.float() ** 2
    for p in tr.values(): p.grad = None
    m = float(sum(f.sum() for f in F.values())) / sum(f.numel() for f in F.values())
    print(f"[ewc] Fisher over {a.ewc_batches} batches of {len(groups)} groups (raw mean {m / a.ewc_batches:.3g})", flush=True)
    return {k: f / max(m, 1e-30) for k, f in F.items()}


def olora_penalty(model, members):
    """11-25 EXP-174 O-LoRA: per LoRA module, A_old = the stacked lora_A rows (PEFT layout r x in) of the member adapters, i.e. the
    previous tasks' LoRAs; returns pen() = sum over modules |A_old A^T|.sum(), the official L1 orthogonality loss"""
    from safetensors import safe_open
    params = dict(model.named_parameters()); A_old = {}
    for p, _ in members:
        with safe_open(p, "pt") as f:
            for k in f.keys():
                kk = k[:-len("weight")] + "default.weight"
                if k.endswith("lora_A.weight") and kk in params: A_old.setdefault(kk, []).append(f.get_tensor(k))
    A_old = {k: torch.cat(v).to(params[k].device, torch.bfloat16) for k, v in A_old.items()}
    n_A = sum(1 for k in params if "lora_A" in k)
    print(f"[olora] A_old for {len(A_old)} / {n_A} modules from {len(members)} members ({sum(v.shape[0] for v in A_old.values()) // max(len(A_old), 1)} rows per module)", flush=True)
    if len(A_old) != n_A: raise SystemExit("[olora] member adapters do not cover every LoRA module")
    return lambda: sum((A_old[k].float() @ params[k].float().T).abs().sum() for k in A_old)


def inflora_design(model, batch_loss, old, new, a):
    """11-25 EXP-174 InfLoRA (Liang & Li, CVPR 2024) with one previous session (the library, folded into the base) and one new
    one. Per LoRA module, C = the input covariance, from a.inflora_rows tokens sampled over a.inflora_batches minibatches (group
    uniform). Old: M = C_old's top eigenvectors until the energy of its singular values reaches a.inflora (official DualGPM
    criterion, first session). New: A = top-r left singular vectors of (I - M M^T) C_new / sqrt(3) (official), frozen; B stays
    at zero and is the only part that trains. With X the sampled rows, C = X^T X: M from the SVD of X (C's singular values =
    S^2), and (I - M M^T) X^T X = Q (R X) with Q R = (I - M M^T) X^T, so its left singular vectors are Q times those of R X"""
    import math
    mods = {n: m for n, m in model.named_modules() if isinstance(getattr(m, "lora_A", None), torch.nn.ModuleDict) and "default" in m.lora_A}
    def collect(groups, seed):
        buf, cnt, q = {n: [] for n in mods}, dict.fromkeys(mods, 0), -(-a.inflora_rows // a.inflora_batches)
        def mk(n):
            def pre(m_, args):
                if cnt[n] >= a.inflora_rows: return
                x = args[0].detach().reshape(-1, args[0].shape[-1])
                x = x[torch.randperm(x.shape[0], device=x.device)[:min(q, a.inflora_rows - cnt[n])]]
                buf[n].append(x.to("cpu", torch.bfloat16)); cnt[n] += x.shape[0]
            return pre
        hs = [m.register_forward_pre_hook(mk(n)) for n, m in mods.items()]
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        with torch.no_grad():
            for _ in range(a.inflora_batches):
                prompt, samples = groups[int(rng.integers(len(groups)))]
                batch_loss(prompt, samples, rng.choice(len(samples), a.batch, replace=False))
        for h in hs: h.remove()
        return {n: torch.cat(b) for n, b in buf.items() if b}
    X_old, M, fr = collect(old, a.seed + 2), {}, []
    for n, X in X_old.items():
        _, S, Vh = torch.linalg.svd(X.to(a.device, torch.float32), full_matrices=False)
        e = S ** 4; r = max(int((torch.cumsum(e, 0) / e.sum() < a.inflora).sum()), 1)
        M[n] = Vh[:r].T.cpu(); fr.append(r / X.shape[1])
    del X_old
    print(f"[inflora] old (library) input subspace: {len(M)} modules, rank / dim median {np.median(fr):.3f} max {max(fr):.3f} (threshold {a.inflora:g})", flush=True)
    X_new, n_d = collect(new, a.seed + 3), 0
    for n, m in mods.items():
        if n in X_new:
            X = X_new[n].to(a.device, torch.float32); Mn = M[n].to(a.device); rk = m.lora_A["default"].weight.shape[0]
            Q, R = torch.linalg.qr(X.T - Mn @ (Mn.T @ X.T))
            U2 = torch.linalg.svd(R @ X, full_matrices=False)[0]
            m.lora_A["default"].weight.data.copy_(((Q @ U2[:, :rk]).T / math.sqrt(3)).to(m.lora_A["default"].weight.dtype)); n_d += 1
        m.lora_A["default"].weight.requires_grad_(False)
        if float(m.lora_B["default"].weight.abs().max()) != 0: raise SystemExit(f"[inflora] {n}: B does not start at zero")
    print(f"[inflora] A designed from the new skills' inputs in {n_d} / {len(mods)} modules (the rest see no input); all A frozen", flush=True)


def init_from_s16(model):
    """Warm-start every LoRA module at the fleet's shared component S16 = U*Sigma @ V^T (rank 16).

    The 76 fleet adapters are ~87% one common vector (pairwise cosine 0.87); adapters trained from the
    standard B=0 init are mutually orthogonal (cosine 0.003) and orthogonal to that component. The
    only mechanism that produces the fleet's signature is a shared initial adapter, so this makes new
    experts pool-consistent by construction -- if the hypothesis is right, capture@16 should land near
    the fleet's 0.80 instead of the 0.001 that zero-init gives.
    """
    from reinforce_train import USIG, VS
    n = 0
    params = dict(model.named_parameters())
    for stem in USIG:
        ka = f"{stem}.lora_A.default.weight"
        kb = f"{stem}.lora_B.default.weight"
        if ka not in params or kb not in params:
            continue
        with torch.no_grad():
            params[kb].copy_(USIG[stem].to(params[kb].dtype))
            params[ka].copy_(VS[stem].T.to(params[ka].dtype))
        n += 1
    print(f"[init] warm-started {n} modules from S16", flush=True)
    if n == 0:
        raise RuntimeError("S16 init matched no LoRA parameters -- check key naming")


def write_protect(model):
    """EXP-123: freeze every S16-initialised lora_A at V^T and return a projector that, after each optimizer step,
    resets B's component in the shared output directions to U*Sigma: B <- B - Un Un^T (B - U*Sigma). The trained update
    is then D = S16 + B_perp V^T with B_perp orthogonal to U: it reads the shared input directions and writes only
    outside the shared output directions (the structure of the LIBERO-90 adapters' task-specific parts)."""
    from reinforce_train import USIG
    params = dict(model.named_parameters()); held = []
    for stem in USIG:
        ka, kb = f"{stem}.lora_A.default.weight", f"{stem}.lora_B.default.weight"
        if ka not in params or kb not in params: continue
        params[ka].requires_grad_(False)
        Us = USIG[stem].to(params[kb].device, torch.float32)
        held.append((params[kb], Us / torch.linalg.norm(Us, dim=0).clamp(min=1e-12), Us))
    print(f"[wp] write-protected {len(held)} modules (A frozen, B shared-output component held)", flush=True)
    if not held: raise RuntimeError("write-protect matched no LoRA parameters")
    def project():
        with torch.no_grad():
            for B, Un, Us in held:
                Bf = B.float(); B.copy_((Bf - Un @ (Un.T @ (Bf - Us))).to(B.dtype))
    return project


def soft_write_penalty(model):
    """EXP-126 (soft write-protection): returns sum_m ||Un^T (B_m - U*Sigma_m)||^2 / sum_m ||U*Sigma_m||^2 over the
    S16-initialised modules; A stays trainable (the SkillsCrafter-style soft alternative to the hard projection)."""
    from reinforce_train import USIG
    params = dict(model.named_parameters()); held = []; den = 0.0
    for stem in USIG:
        kb = f"{stem}.lora_B.default.weight"
        if kb not in params: continue
        Us = USIG[stem].to(params[kb].device, torch.float32)
        held.append((params[kb], Us / torch.linalg.norm(Us, dim=0).clamp(min=1e-12), Us)); den += float(Us.norm() ** 2)
    print(f"[soft] write penalty over {len(held)} modules", flush=True)
    if not held: raise RuntimeError("soft write penalty matched no LoRA parameters")
    def pen():
        return sum(((Un.T @ (B.float() - Us)) ** 2).sum() for B, Un, Us in held) / den
    return pen


def fold_s16_into_base(model):
    """EXP-126 (read-outside): W_base += U*Sigma @ V^T for every stem, so S16 is part of the frozen base during training."""
    from reinforce_train import USIG, VS
    n = 0
    with torch.no_grad():
        for stem in USIG:
            name = stem.split("base_model.model.", 1)[-1]
            try: mod = model.get_submodule(name)
            except AttributeError: continue
            W = mod.weight; D = (USIG[stem].float() @ VS[stem].float().T).to(W.device)
            if D.shape != W.shape: raise RuntimeError(f"S16 shape {tuple(D.shape)} != weight {tuple(W.shape)} at {name}")
            W.add_(D.to(W.dtype)); n += 1
    print(f"[readout] folded S16 into {n} base weights", flush=True)
    if n == 0: raise RuntimeError("fold_s16_into_base matched no modules")


def read_outside(model):
    """EXP-126 (read-outside): LoRA A = random rows projected out of span(V), orthonormalised, frozen; B stays at peft's
    zero init and is trained -> the update B A reads only input directions OUTSIDE the library's shared inputs."""
    from reinforce_train import VS
    params = dict(model.named_parameters()); n = 0
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for stem in VS:
            ka = f"{stem}.lora_A.default.weight"
            if ka not in params: continue
            A = params[ka]; V = VS[stem].float().to(A.device)
            R = torch.randn(A.shape, generator=g).to(A.device); R = R - (R @ V) @ V.T
            Q, _ = torch.linalg.qr(R.T); A.copy_(Q.T[:A.shape[0]].to(A.dtype)); A.requires_grad_(False); n += 1
    print(f"[readout] A frozen outside the shared input directions in {n} modules", flush=True)
    if n == 0: raise RuntimeError("read_outside matched no LoRA parameters")


def fold_context_into_base(model, members, op="ours"):
    """EXP-145: W_base += S16 + sum_m c_m r_m over the context members ([adapter path, c_m]), r_m the two-sided residual
    outside the pool's shared subspace — the library-anchored merge (MM.ours on c_m-weighted B's), built per stem exactly as
    PBuilder / Builder.build build it (fp32 sum, cast back to the base dtype). 11-18 EXP-160: op "avg" = plain weighted sum
    sum_m c_m B_m A_m (vanilla task arithmetic, no pool anchor), as ZBuilder privx with "avg" deploys it."""
    import unified_pool_driver as U, merge_methods as MM
    F = []
    for n, (p, c) in enumerate(members):
        U.EXTPATH[f"ctx:{n}"] = p; F.append((U.fold(f"ctx:{n}"), float(c)))
    dev = next(model.parameters()).device; n = 0
    with torch.no_grad():
        for stem in U.USIG:
            if not any(stem in f and float(f[stem][0].abs().max()) > 0 for f, _ in F): continue
            try: mod = model.get_submodule(stem.split("base_model.model.", 1)[-1])
            except AttributeError: continue
            if op == "avg": D = sum(c * f[stem][0].to(dev) @ f[stem][1].to(dev) for f, c in F if stem in f)
            else: D = MM.ours([c * f[stem][0].to(dev) for f, c in F], [f[stem][1].to(dev) for f, _ in F], 1.0,
                              U.USIG[stem].to(dev), U.VS[stem].to(dev))
            W = mod.weight; D = D if D.shape == W.shape else D.T
            W.copy_((W.float() + D).to(W.dtype)); n += 1
    print(f"[ctx] folded the {op} merge of {len(F)} members (weights {sorted({round(c, 4) for _, c in F})}) into {n} base weights", flush=True)
    if n == 0: raise RuntimeError("fold_context_into_base matched no modules")


def in_context(model, init_path, w, zero=False):
    """EXP-145: LoRA A frozen at V^T; B starts at init_path's private part B_perp = (I - Un Un^T) B (EXP-147: zero=True
    -> B_perp = 0; init_path then only fixes the module set) and its output is scaled by w (the adapter's weight in the merge), so training sees exactly the merged model. Returns (project, finalize):
    project keeps B orthogonal to the shared outputs after each step; finalize adds U*Sigma back and resets the scale, so
    the saved adapter is S16 + B_perp V^T (the s16wp format the priv merge expects)."""
    from reinforce_train import USIG, VS
    import unified_pool_driver as U
    U.EXTPATH["ctx:init"] = init_path; f0 = U.fold("ctx:init")
    params = dict(model.named_parameters()); held = []; da = 0.0
    with torch.no_grad():
        for stem in USIG:
            ka, kb = f"{stem}.lora_A.default.weight", f"{stem}.lora_B.default.weight"
            if ka not in params or kb not in params or stem not in f0: continue
            dev = params[kb].device; Us = USIG[stem].to(dev, torch.float32); Un = Us / torch.linalg.norm(Us, dim=0).clamp(min=1e-12)
            B0, A0 = (x.to(dev, torch.float32) for x in f0[stem]); At = VS[stem].T.to(dev, torch.float32)
            da = max(da, float((A0 - At).abs().max()))
            if zero: B0 = torch.zeros_like(B0)
            params[ka].copy_(At.to(params[ka].dtype)); params[ka].requires_grad_(False)
            params[kb].copy_((B0 - Un @ (Un.T @ B0)).to(params[kb].dtype)); held.append((params[kb], Un, Us))
    if not held: raise RuntimeError("in_context matched no LoRA parameters")
    if da > 1e-3: raise RuntimeError(f"init adapter is not write-protected (max |A - V^T| = {da:.3g})")
    mods = [m for m in model.modules() if hasattr(m, "scaling") and isinstance(m.scaling, dict) and "default" in m.scaling]
    for m in mods: m.scaling["default"] = float(w)
    print(f"[ctx] {len(held)} modules: A = V^T frozen, B = private part of {init_path}, output x{w:g} ({len(mods)} scaled)", flush=True)
    def project():
        with torch.no_grad():
            for B, Un, _ in held: Bf = B.float(); B.copy_((Bf - Un @ (Un.T @ Bf)).to(B.dtype))
    def finalize():
        with torch.no_grad():
            for B, _, Us in held: B.copy_((B.float() + Us).to(B.dtype))
        for m in mods: m.scaling["default"] = 1.0
    return project, finalize


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
