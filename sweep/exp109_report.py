#!/usr/bin/env python3
"""EXP-109 report: clean scaling curve of plain averaging (retention vs same-period single-task runs, panel of 24)."""
import pathlib, sys
import numpy as np
from scipy import stats
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
from exp109_consolidate import PANEL, KS, panel_groups, rec

def merged(k, i, t):
    return rec(f"e109_avg_k{k}_g{i}_t{t}", t) or rec(f"e109_avg_k{k}_g{i}", t)

def main():
    base = {t: rec(f"e109_base_t{t}", t) for t in PANEL}
    L = ["=== EXP-109 clean scaling curve of plain averaging; panel of 24 tasks, 25 episodes each, states 0..24 ===",
         f"single-task (same period): {100*np.mean([b['success_rate'] for b in base.values() if b]):.1f}% over {sum(b is not None for b in base.values())} tasks",
         f"{'k':>3} {'tasks':>5} {'merged':>7} {'single':>7} {'retention':>9} {'gap pp':>7} {'p':>6} {'worst tasks (gap)':s}"]
    for k in KS:
        rows = []
        for i, g, pt in panel_groups(k):
            for t in pt:
                m = merged(k, i, t)
                if m is not None and base[t] is not None: rows.append((t, m["success_rate"], base[t]["success_rate"]))
        if not rows: L.append(f"{k:>3} (no data)"); continue
        M = np.array([r[1] for r in rows]); S = np.array([r[2] for r in rows]); gap = 100 * (M - S)
        p = stats.wilcoxon(gap).pvalue if np.any(gap) else float("nan")
        worst = sorted(rows, key=lambda r: r[1] - r[2])[:3]
        L.append(f"{k:>3} {len(rows):>5} {100*M.mean():6.1f}% {100*S.mean():6.1f}% {100*M.mean()/S.mean():8.1f}% {gap.mean():+7.1f} {p:6.3f} "
                 + ", ".join(f"t{t} {100*(a-b):+.0f}" for t, a, b in worst))
    txt = "\n".join(L); print(txt); (REPO / "results/analysis/exp109_report.txt").write_text(txt + "\n")

if __name__ == "__main__":
    main()
