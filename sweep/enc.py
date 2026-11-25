import torch, sys, json, numpy as np
sys.path.insert(0, "sweep")
from lora2coef import LoRA2Coef
tok = torch.load("results/analysis/lora_tokens_raw.pt")
X, tasks = tok["X"], list(tok["tasks"]); idx = {t: i for i, t in enumerate(tasks)}
net = LoRA2Coef(n_mod=X.shape[1], in_h=X.shape[2], in_w=X.shape[3], n_groups=1)
net.load_state_dict(torch.load("results/analysis/rl_v1_net.pt")); net.eval()
with torch.no_grad():
    E = torch.stack([net.encoder(X[i].unsqueeze(0))[0] for i in range(len(tasks))])
E = E.numpy()
print("ENCODER embeddings of the 76 adapters (does the trunk distinguish them at all?)")
print(f"  embedding dim            : {E.shape[1]}")
print(f"  mean pairwise L2 distance: {np.mean([np.linalg.norm(E[i]-E[j]) for i in range(0,40,4) for j in range(0,40,4) if i<j]):.4f}")
print(f"  per-dim std across adapters (mean): {E.std(0).mean():.5f}")
print(f"  identical-adapter check t7 vs t18  : {np.linalg.norm(E[7]-E[18]):.6f}  (should be ~0)")
print(f"  distinct-adapter  check t6 vs t19  : {np.linalg.norm(E[6]-E[19]):.6f}  (should be >0)")
