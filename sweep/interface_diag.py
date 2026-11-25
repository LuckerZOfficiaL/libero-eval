"""11-20 EXP-165 diagnostic (CPU, weight space): PolicyWeave (2609.33125) finds that VLA experts which adapt the ACTION INTERFACE
(action encoder / decoder; normalization) stop being mergeable, because each expert's hidden layers compensate its own interface
change and averaging breaks the pairing. Our adapters adapt action_in_proj / action_out_proj (+ time_mlp, adaRMS dense) like every
other module; normalization is shared. Question: do the adapters disagree MORE on the interface than on the hidden layers?
Per stem, Gram G_ij = <B_i A_i, B_j A_j>_F = sum(B_i^T B_j * A_i A_j^T) (r x r, no dense dW); reported per module family:
mean pairwise cosine (library-library / new-new / library-new) and averaging retention ||mean_i dW_i|| / mean_i ||dW_i||
(1 = all agree, 1/sqrt(k) = orthogonal). Members = order 0's merge (16 library + 8 new), E145_SEQ picks another order."""
import json, os, re, sys
import torch
from safetensors import safe_open

REPO = "/home/lzhou/libero-eval"
SEQ = int(os.environ.get("E145_SEQ", "0"))
ADIR = f"{REPO}/results/analysis/new_loras_cons" + ("" if SEQ == 0 else f"_s{SEQ}")
MEM = [m[0] for m in json.load(open(f"{ADIR}/ctx_all_zfree.json"))["members"]]
NLIB = 16
def fam(s):
    if "action_in_proj" in s or "action_out_proj" in s: return "interface (action enc/dec)"
    if "time_mlp" in s: return "time mlp"
    if "layernorm" in s or s.endswith("expert.model.norm"): return "expert adaRMS dense"
    if "gemma_expert" in s: return "expert attn/mlp"
    if "language_model" in s: return "language attn/mlp"
    if "vision_tower" in s: return "vision attn/mlp"
    return "other (" + s.rsplit(".", 1)[-1] + ")"
fs = [safe_open(p, "pt") for p in MEM]
stems = sorted({k.rsplit(".lora_", 1)[0] for k in fs[0].keys()})
acc = {}
for s in stems:
    A = torch.stack([f.get_tensor(s + ".lora_A.weight").float() for f in fs])   # k, r, din
    B = torch.stack([f.get_tensor(s + ".lora_B.weight").float() for f in fs])   # k, dout, r
    G = (torch.einsum("iar,jas->ijrs", B, B) * torch.einsum("ird,jsd->ijrs", A, A)).sum((2, 3)).double()
    n = G.diag().clamp_min(1e-30).sqrt(); C = G / n[:, None] / n[None, :]
    k = len(MEM); L, N = slice(0, NLIB), slice(NLIB, k)
    def offmean(M): m = M.shape[0]; return ((M.sum() - M.diag().sum()) / (m * (m - 1))).item()
    ret = (G.sum().clamp_min(0).sqrt() / k / n.mean()).item()
    a = acc.setdefault(fam(s), {"n": 0, "ll": 0., "nn": 0., "ln": 0., "ret": 0., "norm": 0.})
    a["n"] += 1; a["ll"] += offmean(C[L, L]); a["nn"] += offmean(C[N, N]); a["ln"] += C[L, N].mean().item(); a["ret"] += ret
    a["norm"] += n.mean().item()
print(f"order {SEQ}: {len(MEM)} members ({NLIB} library + {len(MEM) - NLIB} new), {len(stems)} stems; k = {len(MEM)}, "
      f"orthogonal retention 1/sqrt(k) = {len(MEM) ** -0.5:.3f}")
print(f"{'family':30s} {'stems':>5s} {'cos lib-lib':>11s} {'cos new-new':>11s} {'cos lib-new':>11s} {'avg retention':>13s} {'mean ||dW||':>11s}")
for f_, a in sorted(acc.items(), key=lambda x: -x[1]["n"]):
    m = a["n"]; print(f"{f_:30s} {m:5d} {a['ll']/m:11.3f} {a['nn']/m:11.3f} {a['ln']/m:11.3f} {a['ret']/m:13.3f} {a['norm']/m:11.3f}")
