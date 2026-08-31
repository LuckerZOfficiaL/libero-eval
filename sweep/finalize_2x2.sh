#!/bin/bash
cd /home/lzhou/libero-eval || exit 1
until ! pgrep -f 'merge_orchestrat[o]r.py' >/dev/null 2>&1; do sleep 120; done
.venv/bin/python - <<'PY' > results/analysis/matrix_2x2_report.txt 2>&1
import json, glob
import numpy as np
plan=json.load(open("sweep/plan.json"))["adapters"]
scenes={a["task_index"]:a["scenes"] for a in plan}
base={}
for ln in open("results/baseline_results.jsonl"):
    r=json.loads(ln)
    if r.get("status")=="ok": base[r["scene"]]=r["success_rate"]
def tr(sr,t):
    ss=[s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss)/len(ss) if ss else None
R={}
for f in glob.glob("results/merge/e18e_*.jsonl"):
    for ln in open(f):
        d=json.loads(ln)
        if d.get("status")=="ok": R.setdefault(d["job"],{})[d["scene"]]=d["success_rate"]
print("=== THE 2x2 MATRIX (delta vs own baseline, 25 rollouts) ===")
for m,mn in (("sh","SHARED merged"),("re","RESIDUAL merged")):
    for tag,ts,lab in (("twin",(24,69),"similar"),("dissim",(5,33),"dissimilar")):
        ds=[]
        for t in ts:
            v=tr(R.get(f"e18e_{m}_{tag}_t{t}",{}),t)
            if v is not None: ds.append((t,v,tr(base,t)))
        detail="  ".join(f"t{t}:{b*100:.0f}->{v*100:.0f}%" for t,v,b in ds)
        mean=np.mean([v-b for _,v,b in ds])*100 if ds else float('nan')
        print(f"  {mn:<16} x {lab:<11}: mean {mean:+5.1f}pp   ({detail})")
PY
echo "$(date '+%H:%M') 2x2 matrix written" >> results/merge_logs/chain_2x2.log
