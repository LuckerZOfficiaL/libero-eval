#!/usr/bin/env python3
"""EXP-83 step 6a: the query embeddings -- DA-MergeLoRA's conditioning signal.

Their query is the frozen CLIP encoder's mean embedding of a batch of TARGET-DOMAIN images. Ours is the
frozen base policy's vision-tower activation, averaged over N of that task's OWN demo frames. One vector
per TARGET keyed by tag -- the goal expert is ONE adapter serving 10 different tasks, so an
adapter-keyed query cannot tell them apart. 86 targets (68 donors + 18 held-out receivers); receivers are embedded here but their adapters
are never in the donor pool and their frames never train the hypernetwork.
"""
import json, pathlib, sys, time
import numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from train_lora import load_demo_samples, BASE, _observation_from

K = 16
def suite_dir(s): return s if s.endswith("_no_noops") else s + "_no_noops"

def main():
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    dev = f"cuda:{max(free)[1]}"; print(f"[q] {dev}", flush=True)
    split = json.load(open(REPO / "results/analysis/exp83_split.json"))
    tasks = split["donors"] + split["receivers"]
    from openpi.runtime import LiberoPolicy
    policy = LiberoPolicy(BASE, norm_stats_path=str(REPO / "results/analysis/norm_assets"), pytorch_device=dev)
    model = policy._model
    for p in model.parameters(): p.requires_grad_(False)
    # The vision tower is called 3x per infer: base camera, wrist camera, and a THIRD slot that is
    # bitwise constant across every task (an empty/padding camera). Keeping only the last call -- which is
    # what a naive hook does -- produced 85 IDENTICAL query vectors and a hypernetwork that correctly
    # learned one answer for everyone. Concatenate the varying calls instead and drop constant ones.
    acts = {"calls": []}
    def hook(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        acts["calls"].append(t.detach().float().mean(tuple(range(t.ndim - 1))))
    mods = dict(model.named_modules())
    vt = mods.get("paligemma_with_expert.paligemma.model.vision_tower")
    if vt is None: raise RuntimeError("vision tower not found")
    h = vt.register_forward_hook(hook)
    Q, keys, ncalls, t0 = [], [], [], time.time()
    missing = []
    for i, d in enumerate(tasks):
        try:
            S = load_demo_samples(suite_dir(d["suite"]), d["task"], n_demos=2, stride=6)
        except FileNotFoundError:
            # a few LIBERO-90 tasks have no regenerated hdf5. A DONOR without demos simply cannot serve
            # as a pseudo-target; a RECEIVER without demos cannot be conditioned on at all and must be
            # dropped from the split. Recorded either way rather than crashing the whole build.
            missing.append(dict(key=str(d["key"]), tag=d["tag"], task=d["task"],
                                role="receiver" if d in split["receivers"] else "donor"))
            print(f"[q] MISSING demos: {d['tag']} ({d['task'][:40]})", flush=True)
            continue
        ix = np.linspace(0, len(S) - 1, min(K, len(S))).round().astype(int)
        vs = []
        for j in ix:
            obs, _ = S[j]
            acts["calls"].clear()
            policy.infer({**{k: v for k, v in obs.items()}, "prompt": d["task"].replace("_", " ")})
            if not acts["calls"]:
                raise RuntimeError("vision-tower hook never fired -- wrong module path?")
            ncalls.append(len(acts["calls"]))
            vs.append(torch.cat(acts["calls"]).cpu().numpy())      # ALL calls; constants dropped below
        Q.append(np.mean(vs, 0)); keys.append(d["tag"])   # TAG: unique per (adapter, task)
        if i % 10 == 0: print(f"[q] {i+1}/{len(tasks)}  ({(time.time()-t0)/(i+1):.1f}s/task)", flush=True)
    h.remove()
    # Calls can differ in number between tasks; keep the common prefix so rows are comparable.
    m = min(len(v) for v in Q)
    Q = np.stack([v[:m] for v in Q]).astype(np.float32)
    # Drop dimensions that are CONSTANT across tasks -- the vision tower has a padding camera slot whose
    # embedding is identical for every task, and keeping only that slot is what produced 85 identical
    # queries on 09-18. Removing zero-variance dims makes the build immune to call ordering.
    sd = Q.std(0)
    keep = sd > 1e-6
    print(f"[q] tower calls per infer: min {min(ncalls)} max {max(ncalls)}; "
          f"{int(keep.sum())}/{len(keep)} dims vary across tasks, {int((~keep).sum())} constant (dropped)", flush=True)
    if keep.sum() < 64:
        raise RuntimeError(f"only {int(keep.sum())} query dims vary across tasks -- the embedding carries "
                           f"no task information, refusing to write it")
    Q = Q[:, keep]
    Q = (Q - Q.mean(0)) / (Q.std(0) + 1e-6)
    np.savez(REPO / "results/analysis/exp83_queries.npz", Q=Q)
    json.dump(keys, open(REPO / "results/analysis/exp83_queries.json", "w"))
    json.dump(missing, open(REPO / "results/analysis/exp83_missing_demos.json", "w"), indent=1)
    if missing:
        nr = sum(1 for m in missing if m["role"] == "receiver")
        print(f"[q] {len(missing)} tasks without demos ({nr} receivers, {len(missing)-nr} donors) "
              f"-> results/analysis/exp83_missing_demos.json", flush=True)
    print(f"[q] done: {Q.shape} for {len(keys)} tasks in {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
