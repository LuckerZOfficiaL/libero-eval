#!/usr/bin/env python3
"""Q1: does the RESIDUAL's effective rank differ between fragile and robust tasks?
r = D - P(D) has an exact rank<=32 factorization r = [b | -U C] @ [a ; V^T] (C = U^T b a V),
so its spectrum comes from a small SVD (QR trick). Effective rank = participation ratio
(sum s^2)^2 / sum s^4, energy-weighted across modules. Rank-deficient modules (7/458 have a
~zero 16th soup direction) are skipped."""
import json, pathlib, sys
import numpy as np, torch
sys.path.insert(0,"/home/lzhou/libero-eval/scripts")
import safetensors.torch as st, apply_lora as al
torch.set_num_threads(10)
REPO=pathlib.Path("/home/lzhou/libero-eval"); K=16
plan=json.load(open(REPO/"sweep/plan.json"))["adapters"]
afile={a["task_index"]:a["adapter_file"] for a in plan}
z=np.load(REPO/"results/analysis/loo_soup_factors_t1.npz")
stems=[str(s) for s in z["stems"]]
US={s:torch.from_numpy(z[f"U_{i}"]).float()[:,:K] for i,s in enumerate(stems)}
VS={s:torch.from_numpy(z[f"V_{i}"]).float()[:,:K] for i,s in enumerate(stems)}
GOOD=[s for s in stems if float(torch.linalg.norm(US[s],dim=0).min())>1e-6][::3]
print(f"[modules] {len(GOOD)} usable (of {len(stems)}; {len(stems)-len([s for s in stems if float(torch.linalg.norm(US[s],dim=0).min())>1e-6])} rank-deficient skipped)",flush=True)
def pr(sv2):
    s1=float(sv2.sum()); s2=float((sv2**2).sum())
    return (s1*s1/s2) if (s2>0 and np.isfinite(s2)) else np.nan
out={}
for t in range(76):
    g=al.extract_lora_groups(st.load_file(afile[t]))
    nr=nd=w=0.0
    for stem in GOOD:
        gg=g[stem]; r=al.infer_rank(gg.a); s=(gg.alpha/r) if gg.alpha is not None else 1.0
        a,b=gg.a.float(),gg.b.float()
        if b.shape[1]!=a.shape[0]: a,b=b.T.contiguous(),a.T.contiguous()
        b=b*s
        U=US[stem]; U=U/torch.linalg.norm(U,dim=0); V=VS[stem]
        Cc=(U.T@b)@(a@V)
        Br=torch.cat([b,-(U@Cc)],1); Ar=torch.cat([a,V.T],0)
        svr=torch.linalg.svdvals(torch.linalg.qr(Br,mode="reduced")[1]@torch.linalg.qr(Ar.T,mode="reduced")[1].T)**2
        svd=torch.linalg.svdvals(torch.linalg.qr(b,mode="reduced")[1]@torch.linalg.qr(a.T,mode="reduced")[1].T)**2
        er,ed=pr(svr),pr(svd); e=float(svr.sum())
        if np.isfinite(er) and np.isfinite(ed) and e>0:
            nr+=er*e; nd+=ed*e; w+=e
    out[t]=(nr/w, nd/w)
    if t%15==0: print(f"  t{t} done  resid_eff={out[t][0]:.2f}",flush=True)
json.dump({str(k):list(v) for k,v in out.items()}, open(REPO/"results/analysis/residual_rank.json","w"), indent=1)
rr=np.array([out[t][0] for t in range(76)]); dd=np.array([out[t][1] for t in range(76)])
FR=[0,6,15,50]; RO=[2,4,5,10,13,16]
print(f"\nRESIDUAL effective rank (all 76): mean {rr.mean():.3f}  sd {rr.std():.3f}  range [{rr.min():.3f}, {rr.max():.3f}]")
print(f"FULL-DELTA effective rank:        mean {dd.mean():.3f}  sd {dd.std():.3f}")
print(f"  fragile {FR}: {[round(float(rr[t]),2) for t in FR]}  mean {rr[FR].mean():.3f}")
print(f"  anchors {RO}: {[round(float(rr[t]),2) for t in RO]}  mean {rr[RO].mean():.3f}")
d=json.load(open(REPO/"results/analysis/full76_capture.json")); ks=d["ks"]
resid=1-np.array([d["curves"][str(t)][ks.index(16)] for t in range(76)])
base={}
for ln in open(REPO/"results/baseline_results.jsonl"):
    rj=json.loads(ln)
    if rj.get("status")=="ok": base[rj["scene"]]=rj["success_rate"]
scenes={a["task_index"]:a["scenes"] for a in plan}
bt=np.array([np.mean([base[s] for s in scenes[t] if s in base]) for t in range(76)])
print(f"  corr(resid eff.rank, residual fraction) = {np.corrcoef(rr,resid)[0,1]:+.3f}")
print(f"  corr(resid eff.rank, baseline perf)     = {np.corrcoef(rr,bt)[0,1]:+.3f}")
