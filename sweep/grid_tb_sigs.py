#!/usr/bin/env python3
"""EXP-59 — finish the shared-donor grid with TARGET-BANK features, to settle a genuine conflict.

dnorm_llm_00 (relative activation-norm change at the first LLM layer) survives a donor-prior control on
the e27 pairs: within-target rho +0.26/+0.29 on two independent observation banks, still +0.21 (p=0.004)
after removing each donor's global quality, while correlating only +0.08-0.10 with that quality. It is
the first feature to do so. But EXP-49's balanced grid put its interaction correlation at +0.090
(p=0.33) -- measured with a different statistic, on 8 fixed donors, and with shared-bank features.

Only 28 of the grid's 120 cells currently have target-bank features. This computes the other 92 so the
exact EXP-49 decomposition can be rerun on target-bank features: same design, same statistic, no
ambiguity left about whether the feature tracks the interaction.
"""
import json, pathlib, sys
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep"))
D = json.load(open(REPO / "results/analysis/shared_donor_design.json"))
need = [(t, d) for t in D["targets"] for d in D["donors"]
        if not (REPO / f"results/analysis/merged_sigs_tb/m{t}_{d}.npz").exists()]
print(f"[gridtb] {len(need)} grid cells need target-bank signatures", flush=True)
if need:
    import merged_sigs as MS
    MS.PAIR_OVERRIDE = need
    sys.argv = ["merged_sigs.py", "--targets", ",".join(str(t) for t in D["targets"]), "--bank", "target", "--device", "cuda:0"]
    MS.main()
print("[gridtb] done")
