#!/usr/bin/env python3
"""Consolidate EVERY pairwise-operator measurement into one label file.

Sources (all use W = S_16 + 1.0*r_target + 0.5*r_donor on the target's scenes):
  e27 probes (10 rollouts), e29 symmetry (10r), e30 test-retest (10r x up to 4 repeats),
  e31 re-probes (25 rollouts).
Repeated measurements of the same (target, donor) are averaged with rollout-count weights,
so pairs seen in both e27 and e31 get labels at effective reliability ~0.6 instead of 0.41.
Output: results/analysis/pair_labels.json  {"t_p": {"aff": pp, "w": total_rollouts}}
"""
import glob, json, re, collections, pathlib
REPO = pathlib.Path("/home/lzhou/libero-eval")
plan = json.load(open(REPO/"sweep/plan.json"))["adapters"]
scenes = {a["task_index"]: a["scenes"] for a in plan}
base = {}
for ln in open(REPO/"results/baseline_results.jsonl"):
    r = json.loads(ln)
    if r.get("status") == "ok": base[r["scene"]] = r["success_rate"]
def brate(t):
    ss = [s for s in scenes[t] if s in base]; return sum(base[s] for s in ss)/len(ss)
def jrate(f, t):
    sr = {}
    for ln in open(f):
        d = json.loads(ln)
        if d.get("status") == "ok": sr[d["scene"]] = d["success_rate"]
    ss = [s for s in scenes[t] if s in sr]
    return sum(sr[s] for s in ss)/len(ss) if len(ss) == len(scenes[t]) else None

acc = collections.defaultdict(list)   # (t,p) -> [(affinity_pp, rollouts)]
# e27 + e29 + e31: affinity vs the fleet baseline
for pat, nroll in ((r"e27_probe_t(\d+)_p(\d+)", 10), (r"e29_sym_t(\d+)_p(\d+)", 10),
                   (r"e31_probe_t(\d+)_p(\d+)", 25)):
    for f in glob.glob(str(REPO/"results/merge/*.jsonl")):
        m = re.match(pat, f.split("/")[-1][:-6])
        if not m: continue
        t, p = int(m.group(1)), int(m.group(2))
        v = jrate(f, t)
        if v is not None: acc[(t, p)].append(((v - brate(t))*100, nroll))
# e30: affinity vs the SAME checkpoint's own solo (4 repeats of near-identical adapters)
solo = {}
for f in glob.glob(str(REPO/"results/merge/e30_t*_solo.jsonl")):
    m = re.match(r"e30_t(\d+)_s(\d+)_solo", f.split("/")[-1][:-6])
    v = jrate(f, int(m.group(1)))
    if v is not None: solo[(int(m.group(1)), int(m.group(2)))] = v
for f in glob.glob(str(REPO/"results/merge/e30_t*_p*.jsonl")):
    m = re.match(r"e30_t(\d+)_s(\d+)_p(\d+)", f.split("/")[-1][:-6])
    if not m: continue
    t, s, p = int(m.group(1)), int(m.group(2)), int(m.group(3))
    v = jrate(f, t)
    if v is not None and (t, s) in solo:
        acc[(t, p)].append(((v - solo[(t, s)])*100, 10))
out = {}
for (t, p), vals in acc.items():
    w = sum(r for _, r in vals)
    out[f"{t}_{p}"] = {"aff": sum(a*r for a, r in vals)/w, "w": w}
json.dump(out, open(REPO/"results/analysis/pair_labels.json", "w"))
multi = sum(1 for v in acc.values() if len(v) > 1)
print(f"pairs: {len(out)}  |  multi-measured (averaged): {multi}  |  "
      f"total rollout-equivalents: {sum(v['w'] for v in out.values())}")
