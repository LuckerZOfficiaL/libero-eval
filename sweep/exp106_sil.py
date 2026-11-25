#!/usr/bin/env python3
"""EXP-106 stage B (arm 1): self-imitation fine-tuning of the TARGET on its own successful rollouts — the standard
way to spend a rollout budget on improving one policy (filtered behaviour cloning / "RL from autonomous data").

Data: the target's 80 collection rollouts (EXP-106 stage A, initial states 25..49, never scored), successful
episodes only: frames every 2 steps with the next action_horizon EXECUTED actions (quantile-normalised exactly like
the demos), mixed 50/50 per batch with the task's own demonstrations (stride 2) so the policy does not drift.
Model: base + LoRA with the fleet recipe (r=16, alpha=16, same 17 module types), WARM-STARTED from the target's own
adapter (all tensors copied and checked), then fine-tuned. Output results/analysis/exp106_sil/t{t}/ (peft format,
drop-in for the orchestrator).
Usage: exp106_sil.py TARGET [--steps 1000 --lr 5e-5 --batch 8 --device cuda:0]
"""
import argparse, glob, json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sweep"))
import safetensors.torch as st
from train_lora import load_demo_samples, lora_recipe, _observation_from, BASE
import unified_pool_driver as U
from reinforce_train import scenes as SC

def rollout_samples(t, H, ADIM, q01, q99):
    out, n_ep = [], 0
    for f in sorted(glob.glob(str(REPO / f"results/exp106_dumps/e106_t{t}_self/*/ep*_success.npz"))):
        z = np.load(f); img, wri, sta, stp = z["img"], z["wrist"], z["state"], z["step"]
        act, ast = z["act"].astype(np.float32), z["act_step"]; pos = {int(s): i for i, s in enumerate(ast)}
        a = act.copy(); k = q01.shape[-1]; a[:, :k] = 2.0 * (a[:, :k] - q01) / (q99 - q01 + 1e-6) - 1.0
        n_ep += 1
        for j, s in enumerate(stp):
            i = pos.get(int(s))
            if i is None or i + H > len(a): continue
            chunk = np.zeros((H, ADIM), np.float32); chunk[:, :a.shape[1]] = a[i:i + H]
            out.append(({"observation/image": img[j], "observation/wrist_image": wri[j], "observation/state": sta[j].astype(np.float32)}, chunk))
    return out, n_ep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=int); ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t = a.target; t0 = time.time()
    task = SC[t][0]; prompt = task.replace("_", " ")
    from openpi.runtime import Pi05LiberoConfig
    H, ADIM = Pi05LiberoConfig().action_horizon, Pi05LiberoConfig().action_dim
    ns = json.load(open(REPO / "results/analysis/norm_assets/norm_stats.json")); sa = ns.get("norm_stats", ns)["actions"]
    q01, q99 = np.asarray(sa["q01"], np.float32), np.asarray(sa["q99"], np.float32)
    demos = load_demo_samples("libero_90_no_noops", task, None, stride=2)
    rolls, n_ep = rollout_samples(t, H, ADIM, q01, q99)
    print(f"[sil] t{t}: {len(demos)} demo samples, {len(rolls)} rollout samples from {n_ep} successful episodes", flush=True)
    if not rolls: raise SystemExit("no successful rollouts")
    from openpi.runtime import LiberoPolicy
    from peft import LoraConfig, get_peft_model
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=a.device)
    model = get_peft_model(policy._model, LoraConfig(**lora_recipe()))
    src = st.load_file(str(U.afile[t])); params = dict(model.named_parameters()); n = 0
    with torch.no_grad():
        for k, v in src.items():
            kk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
            if kk not in params: raise KeyError(f"adapter key {k} -> {kk} not in model")
            if params[kk].shape != v.shape: raise ValueError(f"{kk}: {tuple(params[kk].shape)} vs {tuple(v.shape)}")
            params[kk].copy_(v.to(params[kk].dtype)); n += 1
    print(f"[sil] warm-started {n}/{len(src)} tensors from the target adapter ({(time.time()-t0)/60:.1f} min)", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr); model.train()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); hb = a.batch // 2; losses = []
    for step in range(a.steps):
        bd = [demos[i] for i in rng.choice(len(demos), hb, replace=False)]
        br = [rolls[i] for i in rng.choice(len(rolls), a.batch - hb, replace=len(rolls) < a.batch - hb)]
        b = bd + br
        obs = _observation_from(policy, [x[0] for x in b], prompt, a.device)
        act = torch.tensor(np.stack([x[1] for x in b])).to(a.device)
        loss = model(obs, act); loss = loss.mean() if loss.ndim > 0 else loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        losses.append(float(loss))
        if step % 100 == 0 or step == a.steps - 1:
            print(f"[sil] t{t} step {step}/{a.steps} loss {np.mean(losses[-100:]):.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    out = REPO / f"results/analysis/exp106_sil/t{t}"; out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    json.dump(dict(target=t, steps=a.steps, lr=a.lr, batch=a.batch, demo_samples=len(demos), rollout_samples=len(rolls),
                   success_episodes=n_ep, final_loss=float(np.mean(losses[-100:]))), open(out / "sil_meta.json", "w"), indent=1)
    print(f"[sil] t{t} DONE -> {out} ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
