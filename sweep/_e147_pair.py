"""paired raw-success differences a - b over lib / new skills, pooled over the _e147_abs.py JSON lines on stdin.
usage: ... | _e147_pair.py cons,consz consz,joint"""
import json, sys
from scipy.stats import wilcoxon
pairs = [tuple(x.split(",")) for x in sys.argv[1:]]; D = {}
for ln in sys.stdin:
    d = json.loads(ln)
    for a, b in pairs:
        if a not in d or b not in d: continue
        for g in ("lib", "new"):
            for sc, v in d[a][g].items():
                w = d[b][g].get(sc)
                if v is not None and w is not None: D.setdefault((a, b, g), []).append(v - w)
for (a, b, g), x in D.items():
    print(f"{a} - {b} {g}: n={len(x)} mean {100*sum(x)/len(x):+.1f}pp wins {sum(y>0 for y in x)}/{sum(y<0 for y in x)}"
          f" p={wilcoxon(x).pvalue if any(x) else 1:.3f}")
