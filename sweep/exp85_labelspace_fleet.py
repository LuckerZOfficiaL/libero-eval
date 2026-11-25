"""EXP-85: fleet-fold slice of the label-space shortlist check (variants have no fingerprint, so only t<76 count)."""
import json, sys, pathlib, numpy as np
from itertools import combinations
REPO = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO/"scripts")); sys.path.insert(0, str(REPO/"sweep"))
from reinforce_train import target_baseline
from scipy import stats
RANK = json.loads(open(REPO/"results/analysis/probe_selector76_report.txt").read().split("probe rankings: ")[1].strip())
for tag in sys.argv[1:]:
    picks = {int(k): v for k, v in json.load(open(REPO/f"results/analysis/exp85_picks_{tag}.json")).items()}
    hyb, rnd4, prb, hit, mdl = [], [], [], [], []
    for t, pk in picks.items():
        if t >= 76: continue
        pr = {int(p): s for p, s in RANK[str(t)]}; b = target_baseline(t); best = max(pr, key=pr.get)
        order = sorted(pk["scores"], key=pk["scores"].get, reverse=True); top4 = [int(c) for c in order[:4]]
        hyb.append(100*(np.mean([pr[c] for c in pk["hybrid"]]) - b)); prb.append(100*(np.mean([pr[c] for c in pk["probe"]]) - b))
        mdl.append(100*(np.mean([pr[c] for c in pk["model"]]) - b))
        # exact expectation of "probe within a RANDOM shortlist of 4": mean over all 4-subsets of the value of its top-3
        vals = [np.mean(sorted([pr[c] for c in S], reverse=True)[:3]) for S in combinations(pr, 4)]
        rnd4.append(100*(np.mean(vals) - b)); hit.append(float(best in top4))
    hyb, rnd4, prb, mdl = map(np.array, (hyb, rnd4, prb, mdl)); n = len(hyb)
    print(f"[{tag}] fleet fold, n={n} weak targets: model top-3 {mdl.mean():+.2f} | hybrid (probe within model top-4) {hyb.mean():+.2f} (SE {stats.sem(hyb):.2f})"
          f" | random shortlist-of-4 {rnd4.mean():+.2f} | full probe {prb.mean():+.2f} | hybrid-random4 {np.mean(hyb-rnd4):+.2f} p={stats.ttest_rel(hyb,rnd4).pvalue:.2f}"
          f" | top-4 holds probe-best {np.mean(hit):.2f} (chance 0.50)")
