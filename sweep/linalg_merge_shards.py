#!/usr/bin/env python3
"""Sum the sharded accumulators of linalg_structure.py into linalg_structure.npz (all fields are sums)."""
import glob, pathlib, numpy as np, shutil
R = pathlib.Path(__file__).resolve().parent.parent / "results/analysis"
fs = sorted(glob.glob(str(R / "linalg_structure_s[0-9].npz"))); zs = [np.load(f) for f in fs]
keep = {"KR", "KS", "NN"}
out = {k: zs[0][k] for k in keep}
for k in zs[0].files:
    if k in keep: continue
    out[k] = sum(z[k] for z in zs)
out["done"] = int(zs[0]["done"])      # each shard records the total module count at the end
np.savez(R / "linalg_structure.npz", **out)      # meta json is written by every shard to linalg_structure.json
print(f"merged {len(fs)} shards, {out['done']} modules")
