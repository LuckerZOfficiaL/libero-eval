"""Is a CONTINUOUS per-rollout score (progress / time-to-success) a more reliable probe than binary success?
Same e93 data: 15 targets x 8 candidates x 25 states, all three scores per episode."""
import sys, json, numpy as np
sys.path.insert(0,'sweep'); sys.path.insert(0,'scripts')
from scipy import stats
from subset_probe import TARGETS
from draw_variance import cands
rng=np.random.default_rng(87)
S={}; P={}; T={}
for t in TARGETS:
    s=[];p=[];tt=[]
    for d in cands(t,1):
        recs=[json.loads(l) for l in open(f"results/merge/e93_t{t}_p{d}.jsonl") if l.strip()]
        r=[x for x in recs if x.get("status")=="ok" and isinstance(x.get("success_step"),list)][0]
        ss=np.array(r["success_step"][:25],float); pp=np.array(r["progress"][:25],float)
        s.append((ss>=0).astype(float)); p.append(pp)
        tt.append(np.where(ss>=0, 1.0-ss/300.0, 0.0))            # speed: 0 for failure, higher = earlier success
    S[t]=np.array(s); P[t]=np.array(p); T[t]=np.array(tt)
ts=list(S)
print("per-episode score correlations (pooled): success~progress %.2f, success~speed %.2f"%(
    np.corrcoef(np.concatenate([S[t].ravel() for t in ts]),np.concatenate([P[t].ravel() for t in ts]))[0,1],
    np.corrcoef(np.concatenate([S[t].ravel() for t in ts]),np.concatenate([T[t].ravel() for t in ts]))[0,1]))
# ground truth for a candidate = its binary success over the HELD-OUT states. Score on k probed states by each criterion.
scores={"success (binary)":S,"progress":P,"speed (1 - t_success/300)":T,
        "success + 0.5*progress":{t:S[t]+0.5*P[t] for t in ts}}
for k in (5,10):
    out={n:[] for n in scores}; top={n:[] for n in scores}; regret={n:[] for n in scores}
    for t in ts:
        for _ in range(200):
            pk=list(rng.choice(25,k,replace=False)); rest=[s for s in range(25) if s not in pk]
            truth=S[t][:,rest].mean(1)
            for n,X in scores.items():
                a=X[t][:,pk].mean(1)
                if a.std()>0 and truth.std()>0: out[n].append(stats.spearmanr(a,truth).statistic)
                o=np.argsort(-a); top[n].append(float(truth[o[0]]>=truth.max()-1e-9))
                regret[n].append(truth.max()-truth[o[:3]].mean())        # top-3 pick vs best-3 (what we commit)
    print(f"k={k} probed states (truth = success on the other {25-k}):")
    for n in scores:
        print(f"   {n:28s} rho {np.mean(out[n]):+.3f}   P(top1 best) {np.mean(top[n]):.2f}   top-3 regret {np.mean(regret[n]):.3f}")
# paired significance, k=10, success vs progress, per target
d=[]
for t in ts:
    a=[];b=[]
    for _ in range(200):
        pk=list(rng.choice(25,10,replace=False)); rest=[s for s in range(25) if s not in pk]; truth=S[t][:,rest].mean(1)
        for X,acc in ((S,a),(P,b)):
            v=X[t][:,pk].mean(1); acc.append(stats.spearmanr(v,truth).statistic if v.std()>0 and truth.std()>0 else np.nan)
    d.append(np.nanmean(b)-np.nanmean(a))
print(f"paired per-target (k=10) progress - success: {np.mean(d):+.3f}, better on {sum(x>0 for x in d)}/{len(d)}, p={stats.ttest_1samp(d,0).pvalue:.3f}")
