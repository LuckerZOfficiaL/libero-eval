import torch, sys, json, numpy as np
sys.path.insert(0, "sweep")
from lora2coef import LoRA2Coef
tok = torch.load("results/analysis/lora_tokens_raw.pt")
X, tasks = tok["X"], list(tok["tasks"]); idx = {t: i for i, t in enumerate(tasks)}
net = LoRA2Coef(n_mod=X.shape[1], in_h=X.shape[2], in_w=X.shape[3], n_groups=1)
net.load_state_dict(torch.load("results/analysis/rl_v1_net.pt")); net.eval()
split = json.load(open("results/analysis/rl_v1_split.json"))
E27 = json.loads(open("results/analysis/probe_selector76_report.txt").read().split("probe rankings: ")[1].strip())
good, rand = [], []
for t in split["holdout"]:
    if str(t) not in E27: continue
    g = [p for p, _ in E27[str(t)][:3]]                       # known-good donors for this target
    b = [p for p, _ in E27[str(t)][-3:]]                      # known-bad donors
    with torch.no_grad():
        cg = [float(net(X[idx[t]].unsqueeze(0), X[idx[d]].unsqueeze(0))) for d in g]
        cb = [float(net(X[idx[t]].unsqueeze(0), X[idx[d]].unsqueeze(0))) for d in b]
    good += cg; rand += cb
good, rand = np.array(good), np.array(rand)
print("coefficient assigned on HELD-OUT targets:")
print(f"  known-GOOD donors (e27 top-3)   : {good.mean():.4f}  (n={len(good)})")
print(f"  known-BAD  donors (e27 bottom-3): {rand.mean():.4f}  (n={len(rand)})")
print(f"  difference                      : {good.mean()-rand.mean():+.4f}")
print()
print("  >0 => it is learning to dose GOOD donors higher: the intended behaviour")
print("  <=0 => it is only learning a global low dose (the 'do not merge' attractor, bounded)")
