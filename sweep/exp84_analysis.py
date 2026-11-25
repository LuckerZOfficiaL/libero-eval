import sys, json, numpy as np
sys.path.insert(0,'sweep'); sys.path.insert(0,'scripts')
from scipy import stats
from subset_probe import TARGETS
from draw_variance import cands
rng=np.random.default_rng(84)
M={}
for t in TARGETS:
    rows=[]
    for d in cands(t,1):
        recs=[json.loads(l) for l in open(f"results/merge/e93_t{t}_p{d}.jsonl") if l.strip()]
        r=[x for x in recs if x.get("status")=="ok" and isinstance(x.get("success_step"),list)]
        rows.append([1.0 if s>=0 else 0.0 for s in r[0]["success_step"][:25]])
    M[t]=np.array(rows)
ts=list(M); print(f"{len(M)} targets x 8 candidates x 25 states")
# per-state discrimination
disc=[]; dead=0
for t,X in M.items():
    tot=X.mean(1)
    for s in range(25):
        c=X[:,s]; disc.append(0.0 if c.std()==0 else np.corrcoef(c,tot)[0,1]); dead+= c.std()==0
disc=np.array(disc)
print(f"item-total r: mean {disc.mean():.2f}, p10 {np.percentile(disc,10):.2f}, p90 {np.percentile(disc,90):.2f}; dead states {dead}/{len(disc)}")
# does per-state discrimination correlate across targets? (is state s hard/informative everywhere?)
D=np.zeros((len(ts),25)); H=np.zeros((len(ts),25))
for i,t in enumerate(ts):
    X=M[t]; tot=X.mean(1)
    for s in range(25):
        c=X[:,s]; D[i,s]=0.0 if c.std()==0 else np.corrcoef(c,tot)[0,1]; H[i,s]=c.mean()
# leave-one-target-out: correlate a target's per-state disc with mean over other targets
lo=[]
for i in range(len(ts)):
    oth=np.delete(D,i,0).mean(0); lo.append(stats.spearmanr(D[i],oth).statistic)
print(f"state discrimination transfer across targets (LOTO spearman): mean {np.mean(lo):+.2f}, {sum(x>0 for x in lo)}/{len(lo)} positive")
# split-half reliability of candidate ranking
def rc(X,pick,rest):
    a=X[:,pick].mean(1); b=X[:,rest].mean(1)
    return np.nan if a.std()==0 or b.std()==0 else stats.spearmanr(a,b).statistic
def top1(X,pick,rest):
    a=X[:,pick].mean(1); b=X[:,rest].mean(1)
    return float(b[np.argmax(a)]>=b.max()-1e-9)
res={}
def add(k,v): res.setdefault(k,[]).append(v)
for i,t in enumerate(ts):
    X=M[t]; oth=np.delete(D,i,0).mean(0)
    for k in (5,10):
        top=list(np.argsort(-oth)[:k]); rest=[s for s in range(25) if s not in top]
        add(f"{k} states, fixed most-discriminative (LOTO)",(rc(X,top,rest),top1(X,top,rest)))
        for _ in range(40):
            p=list(rng.choice(25,k,replace=False)); r=[s for s in range(25) if s not in p]
            add(f"{k} states, random",(rc(X,p,r),top1(X,p,r)))
    # ADAPTIVE: spend 5 random states on all 8 candidates, pick the 5 most discriminative from THOSE observed (within-target),
    # then... we only have 25 states total; adaptive = choose next 5 states among the 20 unseen, using the target's own signal?
    # Can't -- unseen states have no signal yet. Instead: adaptive across CANDIDATES within states: rollout 5 random states on all 8,
    # then 5 more random states only on the top-4. Budget 5*8+5*4=60 vs 80 for 10 states x 8.
    for _ in range(40):
        p=list(rng.choice(25,10,replace=False)); p1,p2=p[:5],p[5:]; r=[s for s in range(25) if s not in p]
        a1=X[:,p1].mean(1); keep=np.argsort(-a1)[:4]
        a=a1.copy(); a[keep]=(X[keep][:,p1].sum(1)+X[keep][:,p2].sum(1))/10.0
        a[[j for j in range(8) if j not in keep]]-=1e-3  # pruned ones stay below
        b=X[:,r].mean(1)
        rho=np.nan if a.std()==0 or b.std()==0 else stats.spearmanr(a,b).statistic
        add("adaptive 5 on 8 then 5 on top-4 (60 rollouts)",(rho,float(b[np.argmax(a)]>=b.max()-1e-9)))
print("candidate-ranking reliability vs the remaining states (rho) and P(top-1 pick is truly best):")
for k,v in res.items():
    v=np.array(v); ok=~np.isnan(v[:,0])
    print(f"  {k:50s} rho {v[ok,0].mean():+.2f}   top1 {v[:,1].mean():.2f}   (n={len(v)})")
# how much does the best candidate beat the baseline-like median? spread across candidates
sp=[M[t].mean(1).max()-M[t].mean(1).min() for t in ts]
print(f"candidate spread within target (max-min success over 25 states): mean {np.mean(sp):.2f}")
np.save("results/analysis/exp84_M.npy",np.stack([M[t] for t in ts]))
