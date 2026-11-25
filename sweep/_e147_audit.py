import json, glob, sys, collections
st = collections.Counter(); bad = []
for f in glob.glob(f"results/merge/e112_{sys.argv[1]}*.jsonl"):
    for ln in open(f):
        try: r = json.loads(ln)
        except Exception: continue
        if "success_rate" not in r: continue
        if r["success_rate"] is None: st[r.get("status")] += 1; continue
        st[r.get("status", "ok")] += 1
        if r.get("secs", 99) / 10 < 8 and r["success_rate"] < 1: bad.append((f.split("/")[-1], r.get("scene")))
print(st, "suspicious:", bad[:5])
