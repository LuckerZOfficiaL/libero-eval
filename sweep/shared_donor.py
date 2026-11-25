#!/usr/bin/env python3
"""EXP-49 — shared-donor design: separate DONOR PRIOR from TARGET-DONOR INTERACTION.

Route 1 found one rollout-free feature that ranks a target's own candidates at rho +0.285 and captures
31% of the oracle gap: dnorm_llm_00, the relative activation-norm change at the first LLM layer of the
merged model. Whether that is a real interaction or just "this donor is broadly good" could not be
decided, because e27 gave every target its OWN random 8 candidates -- 24 donors appeared once, only 12
appeared three or more times, and just 4 targets had enough repeated donors to test.

This fixes that by construction: ONE set of 8 donors, applied to ALL 15 weak fleet targets, giving a
complete 15x8 grid. In a full grid the two effects separate exactly, by two-way decomposition
    y(t,d) = mu + a_t + b_d + e_td
where b_d is the donor prior and e_td the interaction. A selector only needs e_td: a donor prior ranks
the same donors first for every target and cannot explain why the probe's picks differ per target.

Donors chosen as the 8 most frequent in the weak targets' existing candidate lists, so 28 of the 120
cells reuse probes and signatures we already own. Stage 1 probes the 92 missing cells at 10 rollouts;
stage 2 computes the 92 missing merged signatures; stage 3 does the decomposition.
"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweep")); sys.path.insert(0, str(REPO / "scripts"))
D = json.load(open(REPO / "results/analysis/shared_donor_design.json"))
TARGETS, DONORS = D["targets"], D["donors"]
RANK = json.loads((REPO / "results/analysis/probe_selector76_report.txt").read_text().split("probe rankings: ")[1].strip())


def existing_probe(t, d):
    """e27 already probed this pair? Its score is the label; the job file is the source of truth."""
    f = REPO / f"results/merge/e27_probe_t{t}_p{d}.jsonl"
    if not f.exists():
        return None
    import unified_pool_driver as U
    return U.get(f"e27_probe_t{t}_p{d}", U.scenes[t][0]) if hasattr(U, "scenes") else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="probe", choices=["probe", "sigs"])
    a = ap.parse_args()
    import unified_pool_driver as U
    from reinforce_train import scenes as SC
    U._clear_orphaned_builds()
    ref = 0.5 * float(np.mean([U.resid_norm(t) for t in range(0, 76, 8)]))
    todo = [(t, d) for t in TARGETS for d in DONORS
            if U.get(f"e27_probe_t{t}_p{d}", SC[t][0]) is None and U.get(f"e59_t{t}_p{d}", SC[t][0]) is None]
    if a.stage == "probe":
        print(f"[sd] grid {len(TARGETS)}x{len(DONORS)}={len(TARGETS)*len(DONORS)}; {len(todo)} cells need probes", flush=True)
        for i in range(0, len(todo), U.CHUNK):
            jobs, built = [], []
            for t, d in todo[i:i + U.CHUNK]:
                p = U.build(f"sd_{t}_{d}", t, [(d, ref / U.resid_norm(d))]); built.append(p)
                jobs.append({"name": f"e59_t{t}_p{d}", "method": "average", "params": {}, "adapters": [],
                             "adapter_paths": [p], "eval_scenes": SC[t]})
            planp = REPO / "sweep/_sd_current.json"
            json.dump(dict(U.META, n_trials=10, jobs=jobs), open(planp, "w"), indent=1)
            subprocess.run([str(REPO / ".venv/bin/python"), str(REPO / "sweep/merge_orchestrator.py"), str(planp)], cwd=REPO)
            for p in built:
                pathlib.Path(p).unlink(missing_ok=True)
            print(f"[sd] probe chunk {i//U.CHUNK+1}/{(len(todo)+U.CHUNK-1)//U.CHUNK} done", flush=True)
        print("[sd] probes done")
    else:
        import merged_sigs as MS
        need = [(t, d) for t in TARGETS for d in DONORS
                if not (REPO / f"results/analysis/merged_sigs/m{t}_{d}.npz").exists()]
        print(f"[sd] {len(need)} merged signatures to compute", flush=True)
        sys.argv = ["merged_sigs.py", "--targets", ",".join(map(str, TARGETS)), "--device", "cuda:0"]
        MS.PAIR_OVERRIDE = need
        MS.main()


if __name__ == "__main__":
    main()
