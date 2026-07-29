# Weekly slides — mergeability: what's going on under the hood
<!-- add items here through the week; figures live in results/figs/ -->

## Context (1 setup slide)
- Base: PI0.5 (PyTorch) + 76 single-task LoRAs (r=16, α=16) on LIBERO-90 (76 consolidated tasks).
- Merge = weight-average N task LoRAs into one policy (average won the method election vs task-arithmetic & TIES).
- Eval: merged model on the union of the merged tasks' own scenes, 10 rollouts/scene.
- Baseline (upper bound) = each parent LoRA on its own task.

## Headline scaling result (context slide, from last week + full sweep)
- Retention vs merge size: **95% (2-way, n=6) → 92% (3-way, n=25, full sweep) → 86% (4-way, n=5)**.
- Full triplet sweep (all 25 consecutive triplets, tasks 0–74): merged 87.1% vs ceiling 94.7%; **0 collapses**, 13/25 ≥95% retention, **5/25 triplets BEAT their parents** (up to 112%).
- Rank-position view (fig `rank_3way_full.png`, `rank_all.png`): best-preserved task ≈ **99–100% at every size**; damage concentrates in the single most-fragile task (68% at 3-way, 45% at 4-way).
- Anchor property: **every** merge keeps a ≥90% task (22/25 triplets keep a 100% one); zero all-fragile groups.

## Q1 — Is task similarity detectable in LoRA weight space? → YES, but tiny
- All 75 adapters share **one dominant direction**: raw pairwise cosine 0.795 ± 0.01 for *every* pair. ~80% of any adapter is a task-agnostic update.
- After removing it (double-centered Gram → **residual cosine**), similarity IS semantic:
  - same-physical-scene pairs: residual cos +0.022 vs −0.016 (permutation p < 1e-4)
  - language overlap ↔ residual cos: Pearson **+0.42**
  - top residual pairs = semantic twins: butter↔butter (+0.10), book↔book, ketchup↔ketchup, pan↔pan
- But the whole residual range is only **[−0.04, +0.10]** — a whisper on top of the shared component.
- Method: 75×75 Frobenius Gram via r×r Gram-trace identity (never materialize ΔW); `sweep/similarity_analysis.py`, cache `results/analysis/lora_gram.npz`.

## Q2 — Do similar tasks merge better? → NO (observationally); headroom is what matters
- 107 (task, group) transfer instances from all 36 average merges.
- Transfer Δ vs similarity-to-partners: **r = +0.06** (Spearman ≈ 0). Group retention vs within-group similarity: r = +0.05. No signal.
- What DOES structure transfer: **headroom** — positive-transfer cases (n=11) have mean own-baseline **78%**, vs 98% (neutral) / 92% (negative). Positive transfer only exists below ceiling.
- Interpretation: the shared 0.8-direction carries most competence (lifts under-trained tasks); the tiny task-specific residual is what gets diluted 1/N and breaks — idiosyncratically.
- Fig: `results/figs/transfer_vs_similarity.png`.

## Supporting under-the-hood findings
- **Fragility is idiosyncratic, not semantic**: twins with near-identical language/scene have opposite fates (t0 moka-pot fragile vs t9 stove-pan robust; t12 fragile vs t13 robust). Not the "odd one out": fragile tasks sit mid-pack in within-group similarity.
- Fragility is partner-dependent, not just size-dependent: t0 = 80% (pair) → 40% (triplet) → 60% (4-way, different partners); t6 = 50 → 10 → 0% (monotonic death).
- Norms don't explain it either: all ΔW norms ≈ 59.5–60.4 (fragile tasks mid-pack).
- t6 (most fragile) is also the most **isolated** adapter in residual space (4 most-dissimilar pairs all involve t6).
- **Data-hygiene discovery**: t7≡t18 and t29≡t62 are byte-identical adapter files (same MD5) — duplicates from the 90→76 consolidation. Found blind by the similarity analysis (residual cos = 1.0) → pipeline validation. (Tell mehdi.)

## Cluster analysis of residual weight space (new)
- Setup: average-linkage clustering + classical MDS on residual cosine (75×75 Gram, duplicates removed → 73).
- **Residual space is high-dimensional**: top eigen-direction = 2.1% of variance; 33 dims for 50%. No global axes — structure is local.
- **Clusters are strongly semantic** (from weights alone, no language input!):
  - k=5: STUDY-book tasks | KITCHEN cabinet/drawer | LIVING-ROOM pick→tray/basket | black-bowl tasks | mug/plate tasks
  - k=8 refines: open/close-drawer (articulated), frying-pan, moka-pot, mug-caddy vs mug-plate
- **Nearest neighbor in weight space = the task's semantic twin**: t0→t23 (both "moka pot on stove"), t6→t64, t12→t30, t24↔t69. NN retrieval ≈ task identification.
- **Fragility does not cluster**: every cluster mixes fragile + robust members → weight-space semantics predicts *similarity*, not *merge-survival*. (Consistent with fragility being idiosyncratic.)
- Fig: `results/figs/lora_residual_mds.png`. Follow-up idea: merge a full coherent cluster (7 book tasks) vs a random same-size group.
- **UPDATED FIG (use this one): `results/figs/residual_map_76.png`** — all 76 tasks (74 unique; duplicates t18/t62 merged into t7/t29), classical MDS of residual weight space, colored by k=8 average-linkage clusters computed from weights alone. All 8 clusters are semantically coherent: book/compartment, drawer/cabinet, frying-pan, wine/moka, pick→tray/basket, black-bowl, mug, mug→plate. 2D shows ~5% of residual variance (space is high-dim) yet the cluster separation is clearly visible.

## First working mergeability predictor: fragility history (new)
- Q: is a task's merge-behavior a stable property? → **YES.** Leave-one-out over 52 instances (20 tasks with ≥2 merges): transfer in a task's *other* merges predicts a held-out merge (different partners!) with **pearson r = +0.71**; LOO error 10.2pp vs 15.3pp for predicting the mean.
- Per-task transfer is bimodal and consistent: fragile tail (t15 −56pp, t6 −44, t0 −40) · exact-0 anchors (t2, t4, t5, t13, t16) · consistent positives (t19 **+35pp** — +30 and +40 in its two merges; t9 +11).
- All a-priori weight-geometry features FAIL as predictors of per-task transfer (73 tasks): headroom r=+0.05, residual isolation r=+0.05, NN similarity r=+0.06.
- Practical rule: **probe a task once — its merge behavior generalizes across partner sets.** Don't merge the fragile tail; anchors and positive-transfer tasks merge for free.
- Open question → can we skip even the probe merge? EXP-5 (running): apply each adapter SOLO at λ=1/3 (what 3-way averaging does to its task-specific component) and eval on its own task. If solo-λ failure = merge failure, we have a partner-free, a-priori fragility test.

## EXP-4b RESULT — do similar tasks merge better? → NO, the OPPOSITE (headline slide)
- Designed test: 24 pairs stratified by adapter residual similarity (8 LOW / 8 MID / 8 HIGH = semantic twins). 64/64 evals, no errors.
- **LOW (dissimilar): 100.8% retention — merging unrelated tasks is FREE. HIGH (twins): 86.4%, std 21.8 — risky.** MID: 96.4%. Continuous r = −0.41 over 30 pairs. Fig: `retention_vs_similarity_pairs.png`.
- HIGH failures are **goal confusion between near-twins** differing in one spatial word: butter back↔front → 60% ret; book front↔right compartment → **t26 −93pp**; book back↔front → 69%.
- But similarity is a *variance* amplifier, not just harm: fragile t6 (−14/−24pp with dissimilar partners) is **rescued by its twin t64: +26pp** — its best result ever; t19 shows the reverse (+40 dissimilar, −10 similar).
- **Mergeability rule v1: merge dissimilar tasks freely; treat near-twins as high-variance (rescue or confusion — worst when twins differ only in target position in the same scene).**

### Rescue vs confusion — what separates them (per-pair anatomy of the 8 HIGH twins)
- **Confusion signature = same scene + same object + goals differ only by a fine PLACEMENT offset in the same receptacle** (butter back/front −50/−30; book front/left −93; book back/front −37; butter-at-front vs pudding, same drawer −30). All task-distinguishing info sits in a cm-scale late-trajectory choice over a near-identical visual context → averaging blends placements (both fail) or commits winner-take-all (one collapses).
- **Rescue/free = the twin differs WITHOUT a placement conflict:**
  - distinct motor programs: pan ON vs UNDER shelf → t6 (fragile, 64% base) **+26pp, its best merge ever** — twin acts as data augmentation, no blendable middle ground;
  - pick-phase difference w/ visual grounding: mirror bowl-stacking (identical word sets!) free +4/0; book middle-vs-right only mild −10/+8;
  - no shared scene: ketchup→tray vs →basket (different rooms) free +4/+8 — confusion needs shared visual context at test time.
- Rescue also needs HEADROOM: it lifts undertrained tasks (t6 64%, t19 60% base); ceiling tasks have nothing to gain.
- **Mergeability rule v2: never merge twins that differ only in target position within the same receptacle; twins with distinct motion types or pick-phase differences are near-free and can rescue undertrained tasks.** (n=8 pairs — clear mechanism, small sample.)

## EXP-5: why merging works at all — partners are load-bearing (mechanism slide)
- Probe: each adapter applied ALONE at λ=1/3 (the dilution a 3-way average applies), eval on its own task.
- **Solo 1/3-scale destroys every task** (17/19 → 0%, including perfect anchors t2/t4/t13) — yet the same tasks keep 90–100% in real 3-way merges. Zero correlation with merge outcomes (r ≈ −0.05).
- ⇒ The "merge = dilute each task to 1/N" picture is **wrong**. Merged model = **full-strength shared component** (all adapters share the 0.795 direction; averaging preserves it) + 1/N task-specific residuals.
- **Partners replenish the shared backbone** — that's why: dissimilar merges are free, anchors stay at 100% (their tasks run on the backbone), fragile tasks are the ones whose success hinges on their own residual, and under-trained tasks get lifted (positive transfer).
- One picture: merging is safe by default because ~80% of every adapter is the same update; the risk is confined to the 20% that's task-specific.
- Confirmation (EXP-5b): the bare base model scores **0% on all 23 probe scenes** — the backbone is not base-model skill, it IS the shared LoRA direction. λ-ladder: base 0% → solo λ⅓ ≈ 0% → solo λ1 = 60–100% → N-way merge 90–100% (partners restore what solo-scaling destroys).

## SOUP75: merge ALL 75 tasks into ONE policy (finale slide)
- The extreme scaling point: weight-average all 75 adapters → single policy, evaluated on 23 probe scenes (10 rollouts each).
- **Retention curve: 95% (2-way) → 92% (3-way) → 86% (4-way) → 81% (75-way).** Sub-linear degradation — merging 75 tasks costs only ~14pp more than merging 2.
- Absolute: 72.2% mean success vs 89.0% baseline on the probe scenes. One policy, 75 tasks, zero training.
- **All 6 anchors stay at exactly 100% even at 75-way** — strongest confirmation of the backbone theory: their tasks run on the shared component, which survives any amount of averaging.
- **Fragility is NOT monotonic in N**: t6 = 40% in the soup vs 10% at 3-way (bad partner averaged out); t12 = 100% vs 80% at 3-way; t0 = 40% = same as 3-way. A 75-way soup can be SAFER than a bad triplet — dilution washes out pairwise conflicts.
- New victims at 75-way: the STUDY-book cluster (t8: 30/10/30% on its 3 scenes vs 70–100% at 3-way; t17 60%) and the usual fragile tail (t15 → 0%). Consistent with cluster-level residual crowding: 7 similar book-tasks compete for one diluted direction.
- Caveats: 23 probe scenes are a hand-picked (not random) subset overweighting interesting tasks; baseline on them is 89% vs 94.7% overall.
- Engineering note: 75-way build OOM-killed the naive merge (loads all adapters + full delta lists); fixed with streaming merge (safe_open handles + running-sum combine, bit-exact validated) + build-cache in the orchestrator (4 probe jobs share 1 build).

## Soup-probe = the practical mergeability predictor (answer-to-the-week slide)
- Validation on 20 tasks with both soup75 + real-merge data (soup held out of all real merges): soup-transfer predicts a task's mean real-merge transfer with **pearson r = +0.83, spearman +0.70** — as accurate as the fragility-history predictor (r=+0.71) but needs **ONE merge build total** instead of one probe-merge per task.
- Binary screen ("soup drop >10pp → do-not-merge"): 16/20 correct, and errors are asymmetric in the SAFE direction — 3 false alarms (t8/t17/t11: book-cluster tasks that crowd only at 75-way) vs 1 missed fragile (t12, partner-specific).
- **Practical recipe: build one grand soup, eval every task once (76 evals), read off the merge-safe list.** Refinement: discount soup-failures that have many same-cluster siblings (crowding, not fragility).
- Caveat: n=20 tasks, 10 rollouts/eval; screen thresholds not tuned out-of-sample.

## EXP-6: merge a coherent cluster vs dissimilar tasks — crowding head-to-head (new)
- Design: 7-way merge of the FULL book cluster (most coherent in weight space, res-cos +0.024) vs a greedy max-dissimilar 7-task control (res-cos −0.020). Same size, same method; compare retention (bases 86% vs 97%).
- **BOOK cluster: 56% retention. CONTROL: 79%.** Merging a coherent cluster costs ~23pp extra at matched size.
- Inside the book merge: **winner-take-all over confusable goals** — caddy compartments: t34 "left" 97%, t17 "back" 80%, t26 "front" 17%, t8 "right" **3%**. The merged policy commits to some compartments, abandons the rest (EXP-4b confusion anatomy at scale).
- t19 rescued AGAIN (+30pp, 60→90%) — 3rd independent merge that lifts this undertrained task.
- Honest surprises: (1) dissimilar is NOT free at 7-way (79% vs ~101% at pairs; new fragile task discovered: t22 mug-by-caddy −80pp); (2) 7-way dissimilar (79%) ≈ 75-way soup (81%) → retention roughly **plateaus after the first few partners**.

## FINAL RESULTS (all runs landed 2026-07-29)

### Grand eval — one policy, ALL 76 tasks (clean headline)
- 76-adapter soup evaluated on **all 90 scenes**: **77.2% vs 94.2% baseline → 82% retention.** Final retention curve: **95% (2) / 92-93% (3) / 86% (4) / 82% (76)**.
- t75 (the last held-out task, a book task): 90% in the soup (−6pp) — crowding hits specific confusable goals, not whole clusters uniformly.

### 3-way retention REPLICATES across partitions
- Shifted partition ([1,2,3]…[73,74,75], 25 new triplets, 93/93 evals): **93% retention** vs 92% on the original partition → the triplet number is robust to grouping (n=50 triplets total).

### Soup-probe validation at full scale (n=76, non-circular ground truth)
- Ground truth = each task's mean transfer over ≥2 independent real merges (orig + shifted partitions + pairs/quads).
- **r = +0.67 (pearson), +0.68 (spearman)** on all 76 tasks (the n=20 preview was +0.83 — full-set is the honest number). Fig: `soup_probe_validation.png`.
- Binary screen (soup drop >10pp → don't merge): **61/75 correct; catches 18/20 true-fragile tasks (90% recall), only 2 misses**; 12 false alarms.
- False alarms are SYSTEMATIC, not random: almost all are cluster-crowded tasks (book family t8/t17/t19, caddy-adjacent placements t22/t35/t72/t74, plate-mug t55/t59) that fail at 75-way from crowding but are fine in small merges → probe is a **conservative screen**.
- **Cluster-discount refinement: tested and REJECTED (honest negative).** Centering each task's soup drop on its weight-space cluster peers drops correlation (+0.67→+0.54) and halves recall (18/20→11/20): intrinsic fragility itself clusters (wine/moka, drawer families), so the discount explains away real signal, and some false alarms live in small clusters it can't touch. **The raw soup-probe stands as the best screen**; separating crowding from fragility needs a small-merge follow-up probe on flagged tasks, not a cluster correction.
- **Final recipe: build one grand soup (~10 min), eval each task once (~1 GPU-day), get: (1) the merge-safe list (90% recall on fragile tasks), (2) which tasks need their own residual, (3) the multi-task policy itself.**

## In flight (may land before the presentation)
