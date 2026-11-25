#!/usr/bin/env python3
"""EXP-82 input: LoRA 'column' tokens for every plan_ext adapter, DA-MergeLoRA style keys.
Per stem, dW = B A has rank 16; its 16 column-directions are the columns of B scaled by the row norms
of A. Each is JL-projected (fixed seed) to 64-d so all stems share one key space. 458 stems x 16 = 7,328
tokens per adapter, plus per-token stem index (-> layer / band embeddings at train time)."""
import pathlib, sys, numpy as np, torch
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
import unified_pool_driver as U
from reinforce_train import group_of
D = 64; g = torch.Generator().manual_seed(82)
proj = {}
def P(n):
    if n not in proj: proj[n] = torch.randn(n, D, generator=g) / np.sqrt(D)
    return proj[n]
stems = list(U.stems); band = np.array([group_of(s) for s in stems], dtype=np.int8)
tasks = sorted(int(x) for x in np.load(REPO / "results/analysis/hypernet_featsexta48.npz", allow_pickle=True)["tasks"])
ext = {a["task_index"]: a["adapter_file"] for a in __import__("json").load(open(REPO / "sweep/plan_ext.json"))["adapters"]}
for t, f in ext.items():
    if t >= 76: U.EXTPATH[f"np:{t}"] = f
out = np.zeros((len(tasks), len(stems), 16, D), dtype=np.float16)
for i, t in enumerate(tasks):
    key = t if t < 76 else f"np:{t}"
    fd = U.fold(key)
    for j, s in enumerate(stems):
        if s not in fd: continue
        b, a = fd[s]                                   # b: [d_out, r], a: [r, d_in]
        cols = b * torch.linalg.norm(a, dim=1)[None, :]  # [d_out, r] column directions of dW
        r = cols.shape[1]
        out[i, j, :r] = (cols.T @ P(cols.shape[0])).numpy().astype(np.float16)
    if i % 25 == 0: print(f"[tok] {i}/{len(tasks)}", flush=True)
np.savez(REPO / "results/analysis/lora_col_tokens.npz", tok=out, tasks=np.array(tasks), stem_band=band, stems=np.array(stems))
print(f"[tok] done {out.shape}", flush=True)
