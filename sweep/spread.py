import torch, sys, json, numpy as np
sys.path.insert(0, "sweep")
from lora2coef import LoRA2Coef
tok = torch.load("results/analysis/lora_tokens_raw.pt")
X, tasks = tok["X"], list(tok["tasks"])
idx = {t: i for i, t in enumerate(tasks)}
net = LoRA2Coef(n_mod=X.shape[1], in_h=X.shape[2], in_w=X.shape[3], n_groups=1)
net.load_state_dict(torch.load("results/analysis/rl_v1_net.pt"))
net.eval()
split = json.load(open("results/analysis/rl_v1_split.json"))
ho, pool = split["holdout"], split["train_pool"]
print("%-8s %8s %8s %8s %9s" % ("target", "mean c", "min", "max", "spread"))
sp = []
for t in ho[:5]:
    with torch.no_grad():
        cs = np.array([float(net(X[idx[t]].unsqueeze(0), X[idx[d]].unsqueeze(0))) for d in pool[:30]])
    sp.append(cs.max() - cs.min())
    print("t%-7d %8.3f %8.3f %8.3f %9.4f" % (t, cs.mean(), cs.min(), cs.max(), cs.max() - cs.min()))
print("\n  mean spread across 30 donors: %.4f" % np.mean(sp))
print("  ~0 => a constant (abstention guard relearned); large => donor-dependent dosing")
