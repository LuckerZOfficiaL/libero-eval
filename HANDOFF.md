# Handoff — read this first on the new server

Migration mechanics: `MIGRATION.md`. Full experiment log: `EXPERIMENTS.md`. Presentation-ready
findings: `results/content_for_slides_31_08.md`. This file is the 2-minute orientation.

## The project in one paragraph

Given a **new LoRA and a pool of other LoRAs**, merge pool adapters *into* it to improve **its own
task** (not multi-task merging — the target is the only thing scored). Fleet: 76 task-specific LoRAs
for PI0.5 on LIBERO-90, all rank 16, all sharing a backbone.

## The method, as it stands (all numbers measured, all 76 tasks)

**Operator** `W = S₁₆ + 1.0·r_target + 0.5·mean(r_donors)` where `S₁₆` = rank-16 SVD of the adapter
soup (the shared component, causally interchangeable) and `r = D − P₁₆(D)` (task-specific residual).
Verified local optimum: moving the donor dose either way costs ~8.8pp; un-protecting the target costs
2.5pp; halving the whole operator costs 8.8pp.

**Selector** — behavioural probe: evaluate 8 candidate donors with the *pairwise* operator at
**10 rollouts each**, merge the top-3. Fleet result: **+2.22pp over all 76 targets (p = 0.001),
+9.78pp on the 15 weak ones, 0 weak targets harmed**, fleet success 94.5% → 96.7%.
Add the free abstention guard (merge only if the best probe beats baseline) → **+2.77pp**, harmed
targets 3 → 1.

## What is settled, so you don't re-run it

- **Selection is irreducibly behavioural.** Five selector families failed on held-out targets: weight
  cosine, donor history (LOOCV R² −0.075), activation distance (r = −0.043), a learned hypernetwork,
  and an action-error proxy on demonstrations (ρ = −0.08).
- **The probe budget is an interior optimum**: 3 → +5.6, 5 → +6.8, **10 → +10.5**, 25 → +9.4 pp.
  Cheaper is too coarse, sharper buys nothing.
- **Affinity is not reciprocal** (r = −0.115, sign agreement 50%, mean gap 17.7pp) → no symmetric
  metric (cosine, any embedding) can represent it.
- **The method is single-shot.** A second round loses 9pp: once the incumbent is good, "commit if the
  best probe beats it" selects for *measurement error* (probe overstated by 20.5pp).
- **Probe reliability is ρ = 0.41** — the same adapter measured 4× gives 70/80/100/50%. Our reported
  numbers are honest (commits are independent 25-rollout measurements) but are **lower bounds**.
- **Winner's curse = 5.0pp** at 10 rollouts: any probe-selected maximum overstates by ~5pp.

## The hypernetwork (closed here — but this is why you have better GPUs)

Best version: shared-trunk MLP (43→32→32) + asymmetric pair head on `[f_T, f_p, f_T−f_p, f_T⊙f_p]`,
**binary** "is this donor good" loss on 634 rollout-derived labels. First real transfer:
+1.31pp over random in label space (p = 0.0065). **But deployment failed**: +2.80pp vs baseline
(n.s., p = 0.36) and **−7.00pp vs the probe (p = 0.030)**. It learned a **donor prior** (picks collapse
onto 2 universal donors; +5.6pp on mid-tier targets, −1.3pp on fragile ones), not the
(target, donor) **interaction**.

**Read this before scaling it up on the new hardware.** The measured blocker is *not* model capacity
and *not* compute — it is **training targets**: the learning curve slope is ~0.018 pp/target and needs
**~140 tasks; the fleet has 76**. Verified dead ends: more candidates per target (flat), more labels
(no change), and **synthetic targets** — perturbed adapters were refuted at a pre-registered checkpoint
(−0.42pp against +0.45 predicted). A bigger hypernetwork on the same 76 tasks will overfit harder, not
generalize better. **The only lever that moves it is more real tasks** — e.g. training a LIBERO-130
fleet, which the new GPUs make feasible. That, not architecture, is the experiment worth running.

## In flight at migration time (both resumable — just re-run the driver)

| run | driver | state |
|---|---|---|
| EXP-26 external **donors** into our targets | `sweep/external_driver.py` | 3/5 targets |
| EXP-27 external LoRA as **target**, our pool as donors | `sweep/external_target_driver.py` | queued behind it |

Every driver skips work already on disk (`results/merge/*.jsonl`, status `ok`), so re-launching after
migration continues rather than restarts. If a job died mid-write, purge its non-`ok` rows first — a
`status: "error"` record is the one thing that can poison a resume.

**EXP-27 is the open scientific question**: capture@16 of our soup basis is 0.84–0.87 for our adapters
but only **0.19–0.39 for external ones**, so its operator keeps the foreign adapter intact
(`W = D_external + α·r_donor`, donors energy-matched) rather than swapping in our shared component.
If our pool improves a foreign LoRA, the pool-consistency caveat that scopes every result in this
project drops away.

## Operational rules that carry over

- **One eval orchestrator at a time.** Two concurrently → `server_died` and build failures.
- **Build → eval → delete** each synthetic adapter (~0.9–1.7 GB each; disk hit 100% twice).
- **Always run experiments in `tmux`** — plain background jobs get reaped; tmux survived a reboot.
- **Watchers must count distinct targets**, not log lines (appended logs double-count on resume).
- Machine-specific constants (`n_workers`, `TORCHDYNAMO_DISABLE`, device ordering, throughput) are
  tuned to the old GPUs — see the last section of `MIGRATION.md` and re-derive them.
