# libero-eval — Experiment Log

Project goal: go beyond single-task LoRA application and develop **model-merging methods** that
combine multiple task-specific LoRA adapters into one PI0.5 policy that solves multiple LIBERO
tasks — evaluated under the existing simulator harness.

Each experiment below is a *branch*: **motivation** (why) → **progress** (what we did) → **result**.
Status legend: 🟢 done · 🔵 running · ⬜ planned (root only) · ⚪ dropped.

---

## EXP-1 · Fix the pi0.5 normalization bug 🟢 done

**Motivation.** Single-task LoRA baselines scored **0%** on their own tasks — the eval harness was
unusable, so nothing downstream could be trusted. Had to find out whether it was a bad checkpoint or
a pipeline bug before any research.

**Progress.**
- Ruled out setup causes: our rebuilt merge is *bit-identical* to the canonical one; norm-stats file
  byte-identical; action-decode correct; EGL frames render fine; `torch.compile` disabled for the
  Pascal P6000s (`TORCHDYNAMO_DISABLE=1`).
- Root cause: `src/openpi/runtime.py` normalized state/actions with **mean/std**, but pi0.5 is trained
  with **quantile (q01/q99 → [-1,1])** normalization (upstream openpi sets `use_quantile_norm` for all
  non-PI0 models). Wrong state normalization corrupted proprioception → degenerate actions (gripper
  frozen open). Confirmed by probe: raw model output was bounded in [-1,1] (quantile signature).
- Fixed both `normalize_inputs` / `unnormalize_outputs` to quantile (mean/std fallback).

**Result.** task0 (moka pot) and task1 (mug) both went **0/5 → 5/5 (100%)**. Change pushed to
`fix/quantile-normalization` on the fork (`LuckerZOfficiaL/libero-eval`).

---

## EXP-2 · Single-task LoRA baseline sweep 🟢 done

**Motivation.** Establish the **reference baseline**: each of the 76 single-task LoRA adapters, applied
on its own scene(s), success rate under the fixed harness. This is the number every multi-LoRA merging
method must be measured against.

**Progress.**
- 76 adapters → **94 (adapter, scene) eval pairs** (14 adapters each cover 2–3 scenes sharing a
  language; adapter is applied to all its scenes). Each adapter evaluated only on its own scene(s).
- Built a resumable 4-worker orchestrator (`sweep/orchestrator.py`): per adapter, merge base+LoRA in
  `/dev/shm` → serve → delete merge → eval each scene (`--args.task-name`, single scene) → record to
  `results/baseline_results.jsonl`. 2 servers per P6000 (ports 8000–8003).
- Fixes made mid-run: EGL rendering pinned to the P6000s (clients were about to OOM the 2 GB K620);
  trials **50 → 25** (a 50-trial pair took ~62 min under 2-servers/GPU contention → ~24 h total; 25 ≈
  half); salvaged 3 interrupted evals by keeping their verified first-25 rollouts; **per-scene video
  subfolders** (`results/videos/task<idx>/<scene>/`) so multi-scene adapters don't overwrite each other.
- Health check: `bash sweep/health.sh`. Live view: `tmux attach -t sweep`.
- **Status: 94 / 94 pairs done, 0 errors.**

**Result.** Single-task baselines are strong across the board. Over all 94 pairs: **mean 94.4%,
median 100%** success. Distribution: 50 pairs at 100%, 37 at 80–99%, 7 at 50–79%, none below 50%.
Weakest (all ≥60%, mostly book-in-caddy and drawer/shelf placement): task8 STUDY_SCENE3 (60%),
task19 STUDY_SCENE4 (60%), task56 KITCHEN_SCENE5 ketchup-in-drawer (60%), task6 KITCHEN_SCENE9
frying-pan-on-shelf (64%). Files: `results/baseline_summary.md`, `results/baseline_summary.csv`,
raw `results/baseline_results.jsonl`, videos `results/videos/task<idx>/<scene>/`. This is the
reference the multi-LoRA merging methods (EXP-3) must match or beat.

---

## EXP-3 · Multi-LoRA merging methods 🔵 running (Stage 2 passed → full set next)

**Motivation.** The actual research direction — methods to *merge* multiple task-specific LoRA
adapters into a single policy that performs multiple LIBERO tasks, tested on the same harness and
compared against the EXP-2 single-task baseline (**mean 94.4%** — the bar a merged multi-task policy
must approach without per-task adapters).

**Approach (scoping).**
- **Enabler:** all 76 adapters are homogeneous — r=16, α=16, no rslora (scale = α/r = 1.0), same 17
  target modules, same base — so each contributes a per-module delta ΔWᵢ = BᵢAᵢ over the *same* 458
  modules. Merging = a per-module *combine* over the stack of compatible deltas.
- **`scripts/merge_loras.py`:** N adapters + a method → one **applied checkpoint** (base + combined
  ΔW). Reuses `apply_lora.py`'s verified delta math. Methods: `average`, `task_arithmetic` (λ·Σ);
  `ties`, `dare` to follow.
- **No new serving/eval code:** the merged checkpoint is served + evaluated by the existing pipeline
  (`serve_policy.py` + `main.py --args.task-name`), exactly like a single-task one.
- **Storage:** merged checkpoints are ephemeral (built in `/dev/shm`, deleted after eval); only the
  *recipe* (tasks+method+params) and *results* are persisted. Optional save of the small merged delta.
- **Task selection decoupled:** a merge config lists `merge_adapters` (which LoRAs to combine) and
  `eval_scenes` (what to test on) separately → measure in-distribution *retention* (merged tasks'
  own scenes) vs held-out *transfer*.
- **Driver:** `sweep/merge_orchestrator.py` (+ `sweep/merge_health.sh`), a fork of the baseline sweep.

**Iteration policy.** Pairwise merges only, **10 rollouts/eval**, staged: **1 pair → 5 pairs →
full set (38 pairs)**. Advance a method only if retention stays high at each stage. One-command
toolkit (see `GUIDE.md`): `bash sweep/run.sh [--pairs 1:2 --method ties | --config configs/merge.yaml]`
→ `bash sweep/merge_health.sh` (check) → `python sweep/merge_summary.py` (merged-vs-baseline
retention). Results saved per pair+method in `results/merge/task_<a>_<b>_<method>.jsonl`.

**Progress.**
- Merge core (`merge_loras.py`) verified bf16-exact; orchestrator/health/summary/generator built.
  Incremental (module-by-module) combine: TIES build 11 min -> ~2.5 min (cold), 34 -> 21 GB RAM.
  Scheduling = job queue: <=4 merges live (2/GPU), rest queue; builds serialized (RAM). One-command
  driver `bash sweep/run.sh` (defaults to `configs/merge.yaml`); cross-product over pairs x methods.
- **First comparison - pair 1:2 (mug+microwave), 3 methods, 10 rollouts** (baseline 100%/100%):

  | method | microwave | mug | pair mean |
  |---|---|---|---|
  | average | 100% | 90% | **95%** |
  | ties (d=0.2) | 90% | 70% | **80%** |
  | task_arithmetic (lambda=1) | 0% | 30% | **15%** |

**Result (2026-07-23).** Pairwise merging works; **merge scale lambda is the dominant knob**.

- lambda-sweep (task_arithmetic, pair 1:2): lambda 0.3->40%, 0.5->95%, **0.7->100%**, 1.0->15%.
  Non-monotonic sweet spot ~0.7 (too small under-fits, too large interferes). average = ta(lambda=0.5).
- Stage 2 (5 pairs x {average,ties}, baseline mean 93.6%): average **88.0%**, ties **86.6%** mean
  retention; no collapses, task_2_3/task_4_5 perfect, harder pairs task_6_7/task_8_9 drop most.
- Methods validated bit-exact vs PEFT's canonical `ties`/`task_arithmetic` (0.0 err).
- **Passes Stage 2 -> next: full set (38 pairs)**, and pin the lambda optimum (~0.6-0.8).

**Result (2026-07-23) — winner election, 6 pairs x 10 rollouts.** Ran every method that spans all
6 pairs and elected a winner. Upper bound (both parent LoRAs on their own task) = **93.8%**.

| method     | mean | vs upper bound |
|------------|------|----------------|
| **average**  | **89.1%** | **95%** (winner) |
| ties lambda1.0 | 86.2% | 92% |
| ta lambda0.7   | 84.9% | 91% |
| ties lambda0.7 | 50.9% | 54% (density0.2 trim + low lambda destroys signal) |

- ta lambda 0.3/0.5/1.0 stayed single-pair (task_1_2) by choice — cost/time, not needed once 0.7 known.
- One eval (task_0_1 ties lambda0.7 / KITCHEN_SCENE3) hit a `parse_fail`; recovered by re-running that
  single scene (0.3) — didn't change any ranking.
- Table is regenerable any time: `python3 sweep/full_table.py` (add `--all` for partial configs).
- **Winner = weight `average`.** Note it *is* task_arithmetic(lambda=0.5), so "winner + weight average"
  collapse to one method -> triplet sweep runs **average only** (user decision).

**Next: EXP-3b · 3-way (triplet) merging.** 5 triplets x {average}, `eval_tasks: auto` (evaluates on the
union of the merged tasks' scenes, any group size). Launch:
`bash sweep/run.sh --scope 5 --group-size 3 --method average`.

**Result (2026-07-23) — triplets, 5 groups x 10 rollouts, average.** 19/19 scenes ok, 0 errors (4 workers).
Upper bound (3 parents on own tasks) = **92.3%**; merged average = **88.0%** -> **95% retention** — i.e.
3-way merging is about as good as 2-way (pairwise average was also 95%). Merging degrades *gracefully*
with group size.

| triplet | tasks | base↑ | average | retention |
|---|---|---|---|---|
| task_0_1_2   | 0,1,2    | 100% | 73%  | 73% (one member interferes — only real drop) |
| task_3_4_5   | 3,4,5    | 99%  | 100% | 101% (beats parents) |
| task_6_7_8   | 6,7,8    | 80%  | 77%  | 96% |
| task_9_10_11 | 9,10,11  | 91%  | 100% | 110% (beats parents) |
| task_12_13_14| 12,13,14 | 92%  | 90%  | 98% |

- 2 of 5 triplets *exceed* their parent baseline -> positive transfer / regularization, not just interference.
- Regenerable: triplet-aware table via the inline script (parses N-way job names); pairwise via `sweep/full_table.py`.

**Interference diagnosis (2026-07-23).** Broke every `average` merge down per-scene (scene -> owning task).

*Behavioral.* Interference is **asymmetric and task-specific**, not a uniform blur: a few **fragile** tasks
collapse while partners are untouched or improve. Fragile: **t0** (moka pot, -20 in pair / -60 in triplet),
**t6** (frying pan on shelf, -14 / -54), t12 (-16), t1 (~-13). Robust/positive: t9 (+11.5%), t14 (+10%), t8
(net +1.7%, even recovers in triplet), t2/t4/t5 unchanged. Winner-take-all: in 0_1_2, t0->40% while t2 stays
100%. **Fragility compounds with group size** (t0: -20 -> -60; t6: -14 -> -54).

*Weight-space (null result).* `sweep/weight_geometry.py` (Gram-trace identity, no out x in materialization):
all 16 task delta norms are ~uniform (59.5-60.4; fragile t0=60.0 sits mid-pack, largest is robust t13=60.4),
and cosines are uniform across fragile AND robust — cos(self,merged)~0.93, cos(self,Sigma-others)~0.84 for
everyone. **Weight geometry does NOT predict which task collapses.** All task LoRAs share a large common
direction (~0.84 mutual cosine); averaging keeps it and attenuates each task's small orthogonal residual by
~1/N -> collapse depends on whether a task's *behavior* hinges on that residual (functional, not geometric).
=> norm-reweighting can't help; mitigation needs functional importance (Fisher/RegMean) or per-task lambda.

**EXP-3c · 4-way merging (running).** 5 subsets of 4 x average, auto-eval. Tests the compounding prediction
(t0 in group 0_1_2_3, t6 in 4_5_6_7 should collapse further). Launch:
`bash sweep/run.sh --scope 5 --group-size 4 --method average --workers 4`.

**Result (2026-07-23) — EXP-3c 4-way, 5 subsets x 10 rollouts, average.** 25/25 ok, 0 errors.
**Retention holds flat through 3-way then drops at 4-way** — the scaling story:

| size | #groups | base↑ | merged | retention |
|---|---|---|---|---|
| 2-way | 6 | 93.8% | 89.1% | 95% |
| 3-way | 5 | 92.3% | 88.0% | 95% |
| 4-way | 5 | 90.6% | 78.3% | **86%** |

- Uneven, not uniform: 2/5 four-way groups still ~100% (task_8_9_10_11 99%, task_16_17_18_19 100%); mean
  dragged down by severe collapses — task_4_5_6_7 (t6 -> 0%), task_12_13_14_15 71% (3 of 4 tasks drop; t14 was
  +10% robust at 3-way, adding t15 made it fragile -30).
- Fragile trajectory not monotonic: t6 50->10->0% (dies); t0 80->40->60% (bounces at 4-way w/ different partners).
- Positive transfer weakens: 3-way had 2 groups beat parents (101/110%); 4-way best only *matches* (100%).
- Caveats: subsets differ across sizes (4-way adds harder tasks t15/t19 -> confound with size); not a controlled
  per-task ablation. Aggregate trend real; the 95%->95%->86% is the headline.

**PLAN CHANGE (2026-07-23).** The "full sweep" is now **all consecutive (disjoint) triplets**, NOT 38 pairs.
`--scope all --group-size 3` = 25 triplets ([0,1,2]..[72,73,74]), 93 scene-evals, ~6h @ 4 workers, average only.
Rationale: triplets are the target unit of the multi-LoRA story; pairs were the calibration stage.
Run after EXP-3c + the semantic analysis land:
`bash sweep/run.sh --scope all --group-size 3 --method average --workers 4`.

**RESULT (2026-07-24) — FULL TRIPLET SWEEP, 25 consecutive triplets (tasks 0-74), average, 10 rollouts.**
93/93 scenes ok, 0 errors. **Mean retention = 92%** (merged 87.1% vs upper bound 94.7%). Holds close to the
n=5 pilot (95%); the honest full-set number is 92%.
- Distribution: 13/25 triplets >=95% retention, 4 in 85-95%, 8 in 70-85%, **0 below 70%** — no triplet collapses.
  Worst: task_48_49_50 76%, task_0_1_2 / 69_70_71 73%, task_72_73_74 75%. Best: 5 triplets BEAT parents
  (task_18_19_20 112%, task_9_10_11 110%, task_42_43_44 103%) -> positive transfer common at 3-way.
- ANCHOR holds at scale: every one of 25 triplets keeps a well-preserved task (best-per-triplet mean 99%,
  min 90%; 22/25 have a 100% task, 25/25 >=90%). Zero all-fragile triplets. So "each merge keeps an anchor"
  is not an artifact of the first 20 tasks — it's true across the whole set.
- Rank-position chart refreshed to n=25 (results/figs): 3-way r1=99% r2=93% r3=68% (pilot n=5 was 100/91/66 —
  stable). Scaling picture: mean retention 95%(2-way) / 92%(3-way) / 86%(4-way).
- Regenerate: `sweep/full-triplet analysis inline` + `sweep/rank_plots.py`.
CONCLUSION for the project: **3-way LoRA merging works across the full LIBERO task set** — 92% retention, no
group collapses, always an anchor; degradation is confined to a single fragile task per group and only becomes
severe at 4-way.

---

## EXP-9 · Week 2: spectral structure of adapters & conflict directions 🔵 running

**Motivation.** Project has two candidate goals (not choosing yet): (1) improve a single task's in-domain
performance by merging partners INTO its LoRA (positive transfer exists: t19 +30/40pp, t6 +26pp); (2) soup
curation — decide whether a NEW LoRA should be admitted to the soup. Both demand **eval-free signals from
the weights of an unseen adapter** — fragility-history/soup-probe need evals of everything, so they're
preliminaries, not the method. This week: look inside the adapters. Slides for next meeting accumulate in
`results/content_for_slides_10_08.md` (only new-week content).

**Plan (all CPU, no GPU needed):**
- T1 · Per-task singular-value spectra: per-module sv of scale*B@A via the r x r trick (sv^2 of BA = eigs of
  (B^T B)(A A^T)); per-task variance-explained vs rank curve + effective rank; compare fragile vs anchor.
- T2 · Dimensionality of residual dissimilarity: eigenspectrum of the double-centered 76-Gram — is
  task-distinguishing structure concentrated in few dimensions?
- T3 · Twin-conflict directions: for spatial-twin pairs (butter back/front 24:69, book compartments
  8/17/26/34, pan on/under 6:64, book middle/right 19:49, tray/basket 3:27 & 10:29, mirror bowls 44:52),
  form difference vectors d = delta_i - delta_j; (a) are they low-rank per module? (b) do SAME spatial
  contrasts share direction across object families (a "back-vs-front direction")? Inner products via Gram
  identity <da-db, dc-dd> = G[a,c]-G[a,d]-G[b,c]+G[b,d].

Then discuss next steps with these results in hand.

**Results (2026-07-29, all three todos done same day — CPU only):**
- **T1: spectra low-rank but UNIFORM.** Per-module rank-1 = 43% of variance, top-4 = 68%, eff. rank ~8.3/16 —
  identical for all 76 tasks (range 8.21-8.40); fragile vs anchor indistinguishable. No signal.
- **T2: residual structure high-dimensional.** Top PC 2.1%, 34 dims for 50%, 65 for 90% (of 73). No compact
  task-identity subspace.
- **T3: no shared spatial-word direction.** cos(butter-b/f, book-b/f) = -0.001, cos(ketchup-t/b, soup-t/b) =
  +0.009 vs random baseline |cos|~0.009 -> spatial words are encoded contextually, not compositionally.
  Caddy-compartment family forms a NEAR-REGULAR TETRAHEDRON (shared-endpoint cos ±0.5, opposite edges 0.00):
  four compartments mutually equidistant, no front-back/left-right axes. Twin diffs also less rank-
  concentrated than adapters (top-1 33% of rank-32) — conflicts distributed, not localized.
- **Reconciliation of T1-43% vs T2-2.1% (user question, verified on 6 sampled modules):** T1 = within-module
  SVD of the FULL delta (backbone in), T2 = across-task PCA of residuals. Cross-task |cos| of top-1 module
  directions = 0.62-0.94 (~0 for random) -> the top singular component of each module IS the shared backbone
  (hence uniform T1 spectra). Residualized per-module updates: top-1 drops (60->36%, 35->22%) but stay
  low-rank WITHIN a module (top-8 ~85-94%) while the directions differ ACROSS tasks (the 34-dim world).
  Picture: adapter = shared dominant backbone directions + private low-rank residual directions.
- **Week-2 takeaway:** adapters are spectrally/geometrically homogeneous in every linear view (norms, cosines,
  spectra, PCs, word-directions). Cheap linear weight signals cannot detect mergeability or conflicts —
  consistent with week 1 (functional soup-probe works, geometry fails). Candidate next: activation-space
  probing, per-layer directions, or Goal-1 merge-into-task experiments.
  Figs: week2_spectra.png, week2_twin_directions.png; data: task_spectra.npy, residual_eigs.npy,
  twin_dir_cos.npy, twin_diff_spectra.json (results/analysis/).

## EXP-4 · Similarity vs transfer — what's going on under the hood 🔵 running

**Motivation.** Some merges BEAT their parents (5 of 25 triplets). Q1: is task similarity detectable in LoRA
weight space? Q2: do similar tasks merge better (positive transfer) or worse? Understand the mechanism.

**Analysis (2026-07-24, `sweep/similarity_analysis.py`).** Full 75x75 adapter Gram (Gram-trace, cached
`results/analysis/lora_gram.npz`); raw cosine + RESIDUAL cosine (double-centered — removes the dominant
~0.795 shared component). 107 transfer instances from all 36 completed average merges.

*Discovery:* **t7≡t18 and t29≡t62 are byte-identical adapter files** (same md5) — leftover duplicates from
the 90->76 consolidation. Validates the pipeline (residual cos exactly 1.0) ; exclude t18/t62 from pair stats.

*Q1 — YES, similarity is detectable (weak but real):* same-physical-scene pairs have higher residual cos
(+0.022 vs -0.016, perm p<1e-4); language Jaccard correlates with residual cos (pearson +0.42). Top residual
pairs are semantic twins (butter/butter +0.10, book/book, ketchup/ketchup, pan/pan). BUT the scale is tiny:
residual cos spans just [-0.043, +0.103] on top of a 0.795 shared direction. t6 (most fragile task) is the
most isolated adapter in residual space (bottom-4 pairs all involve t6).

*Q2 — NO, partner similarity does NOT predict transfer:* transfer vs residual-cos-to-partners r=+0.06
(spearman -0.01); group retention vs within-group similarity r=+0.05. What DOES structure transfer:
**headroom** — positive-transfer instances (n=11) have mean own-baseline 78%, vs 98% for neutral / 92% for
negative. Positive transfer only happens below ceiling; at base=100% you can only lose. Figure:
`results/figs/transfer_vs_similarity.png`.

**Cluster analysis (2026-07-27, `sweep/cluster_analysis.py`).** Average-linkage clustering + classical MDS on
the residual Gram (duplicates t18/t62 removed, n=73). Findings:
- Residual space is HIGH-DIMENSIONAL / near-isotropic: top eigen-direction only 2.1% of variance; 33 dims for
  50%, 56 for 80%. No few dominant axes — structure is local (neighborhoods), not global.
- Yet clusters are STRONGLY SEMANTIC: k=5 -> {STUDY book tasks (n=7)}, {KITCHEN cabinet/drawer (n=24)},
  {LIVING_ROOM pick->tray/basket (n=17)}, {black-bowl tasks (n=13)}, {mug/plate (n=12)}. k=8 refines into
  open/close-drawer, frying-pan, moka-pot, mug-caddy vs mug-plate clusters. Weight residuals encode
  object + receptacle + scene + verb.
- Nearest neighbor in residual space = the task's semantic twin: t0->t23 (both "moka pot on stove"),
  t6->t64 (frying pan/cabinet), t12->t30 (middle black bowl), t24<->t69 (butter/basket). NN retrieval in
  weight space does task identification.
- Fragility does NOT concentrate in any cluster (every cluster has both fragile and robust members) — again:
  semantics predicts SIMILARITY, not MERGE-SURVIVAL.
- Fig `results/figs/lora_residual_mds.png`. Suggests a future test: merge a whole coherent cluster (e.g. the
  7 book tasks) vs a random same-size group. EXP-4b HIGH pairs are within-cluster, LOW are cross-cluster.

**Fragility history is the first working predictor (2026-07-27).** Leave-one-out over the 52 instances of
tasks with >=2 merges: a task's mean transfer in its OTHER merges predicts its transfer in a held-out merge
with **pearson +0.71** (spearman +0.47); LOO |error| 10.2pp vs 15.3pp for predict-the-mean. Meanwhile ALL
a-priori weight-geometry features fail on per-task avg transfer (73 tasks): headroom r=+0.05, residual
isolation r=+0.05, NN cos r=+0.06. Per-task transfer is strikingly stable & bimodal: fragile tail
(t15 -56pp, t6 -44, t0 -40, t12 -21, t1 -15) vs exact-0 anchors (t2,t4,t5,t13,t16) vs consistent positives
(t19 +35pp: +30 and +40 in its two merges; t9 +11). => "WHAT to merge" is already answerable empirically:
probe a task once, its behavior generalizes across partner sets. Open question -> can we get the signal
WITHOUT any merge: EXP-5 solo scale-probe (below).

**EXP-5 (planned) · solo λ-scale probe as a merge-free fragility predictor.** Mechanistic hypothesis: N-way
averaging scales each task's residual by 1/N; a task collapses iff its OWN adapter, applied alone at λ=1/N,
already fails. Probe: apply each history task's adapter solo at λ=0.333 (3-way equivalent), eval on its own
scenes (10 rollouts). If solo-λ performance correlates with observed 3-way transfer, we have a cheap,
partner-free, a-priori mergeability measure. ~20 jobs / ~30 scene-evals ≈ 2h. Runs when EXP-4b frees the
GPUs (tooling supports it: `--pairs "0,1,2,..." --method task_arithmetic --params '{"lambda":0.333}'`).

**EXP-4b · designed similarity test 🟢 done (2026-07-27).** 24 pairs stratified by residual cos (8 LOW ~-0.04 /
8 MID ~-0.02 / 8 HIGH +0.06..+0.10 = semantic twins), 64/64 scenes ok. **THE HYPOTHESIS INVERTS: similar
tasks merge WORSE, and with far higher variance.**

| stratum | mean retention | std | mean per-task Δ |
|---|---|---|---|
| LOW (dissimilar) | **100.8%** | 10.5 | +0.5pp |
| MID | 96.4% | 2.8 | -4.8pp |
| HIGH (twins) | **86.4%** | 21.8 | -12.5pp |

- Continuous (30 pairs incl. 6 orig): retention vs residual cos **pearson -0.41** (spearman -0.14 — driven by
  the HIGH tail). Fig `results/figs/retention_vs_similarity_pairs.png`.
- **Dissimilar tasks merge for FREE** (LOW ~101% retention — near-independent updates).
- HIGH failures are catastrophic goal-confusions between twins differing in one spatial word:
  24:69 butter back/front -> 60% ret (both drop); **26:34 book front/right compartment -> t26 -93pp**;
  17:26 book back/front -> 69%. Benign HIGH: 3:27 ketchup (diff scenes) 106%, 44:52 mirror-stack 102%.
- **Similarity amplifies interaction in BOTH directions** (within-task contrasts): fragile t6 with dissimilar
  partners -14/-24pp, but with its twin t64 **+26pp (rescued — t6's best result ever)**; t19: +40 with
  dissimilar t30, -10 with similar t49. t11: -6 (MID) vs +4 (HIGH).
=> Mergeability rule v1: **merging dissimilar tasks is safe; merging near-twins is high-variance — they
   either rescue each other or confuse goals** (worst when twins differ only in target position within the
   same receptacle/scene). Similarity is a VARIANCE knob, not a quality knob.

**EXP-5 · solo λ=0.333 probe 🟢 done (2026-07-27) — NULL as predictor, but reveals the MECHANISM.**
23/23 scenes ok. Solo 1/3-scaling **destroys every task** (17/19 -> 0%, incl. perfect anchors t2/t4/t13;
only t12 50%, t16 20%) — yet the same tasks retain 90-100% in real 3-way merges. Correlation with merge
transfer: r=-0.05 (3-way) / -0.12 (avg) — useless as a predictor.
=> **The dilution model of merging is WRONG.** Average of N adapters = full-strength SHARED component
(all adapters share the 0.795 direction; averaging preserves it) + 1/N task residuals. Solo λ=1/3 scales the
shared backbone too -> collapse. **Partners are load-bearing: they replenish the common competence; only the
task-specific residual is diluted.** Explains: LOW-similarity merges free (residuals don't interact), anchors
at 100% (their tasks run on the shared backbone), fragile tasks = those whose success depends on their own
residual, positive transfer = partners' backbone lifting under-trained tasks.

**EXP-5b · base-model probe (λ=0) 🟢 done (2026-07-27) — base = 0% on ALL 23 scenes.** Bare pi0.5 has zero
LIBERO competence. Uninformative as a per-task predictor (no variance), but it pins down the mechanism: the
load-bearing "backbone" is NOT base-model skill — it IS the shared LoRA direction (present only when adapters
are merged in). λ-ladder per task: λ=0 -> 0%, solo λ=1/3 -> ~0%, solo λ=1 -> 60-100%, N-way merge -> 90-100%.

**EXP-5c · grand soup (75-way average) 🟢 done (2026-07-28)** — THE definitive backbone probe: average ALL 75
adapters into one policy, eval on the 23 probe scenes. 23/23 ok.
- First attempt OOM-killed silently (naive merge loads all 75 adapters ~14GB + per-module lists of 75 fp32
  deltas). FIX: streaming merge in `scripts/merge_loras.py` (lazy safe_open handles, running-sum combine for
  average/ta; bit-exact validated vs old code on a pair, 812/812 tensors) + build-cache in orchestrator
  (`job_sig`-keyed, refcounted eviction — 4 probe jobs shared 1 build).
- **Retention curve completes: 95% (2-way) / 92% (3-way) / 86% (4-way) / 81% (75-way)** — sub-linear
  degradation; ONE policy does 75 tasks at 72.2% mean vs 89.0% baseline (probe scenes; hand-picked subset,
  baseline 89% vs 94.7% full-set — caveat).
- **All 6 anchors at exactly 100% at 75-way** — backbone theory confirmed: anchor tasks run on the shared
  component, indestructible under averaging.
- **Fragility not monotonic in N**: t6 40% in soup vs 10% at 3-way; t12 100% vs 80%; t0 40% = 3-way. A grand
  soup can BEAT a bad triplet — dilution averages out pairwise conflicts.
- New 75-way victims: STUDY-book cluster crowds out (t8 30/10/30 vs 70-100 at 3-way; t17 60%), t15 -> 0%,
  t14 50%. Cluster-level residual competition (7 book tasks share one diluted direction).
- Backbone-probe reading: soup-solves = merge-safe list; soup-fails = needs-own-residual list (t8, t14, t15,
  t17 + known t0/t6) -> candidate a-priori "what not to merge" measure for the week's goal.

**EXP-5d · soup-probe validated as mergeability predictor 🟢 done (2026-07-28).** On the 20 tasks with both
soup75 and real-merge data (soup fully held out): soup-transfer vs mean real-merge transfer **pearson +0.83,
spearman +0.70** — matches fragility-history (r=+0.71) at 1/20th the cost (ONE build for all tasks vs one
probe-merge per task). Binary screen (soup drop >10pp = don't merge): 16/20; errors safe-asymmetric — 3 false
alarms (t8/t17/t11 = book-cluster crowding, only a 75-way artifact) vs 1 miss (t12, partner-specific).
**Practical recipe: one grand soup + one eval per task -> merge-safe list.**

**EXP-5e · full 76-soup consistency rerun ⬜ planned.** Rerun the 23 probe scenes under the TRUE 76-adapter
soup (replacing soup75 numbers) + t75's scene -> clean "all 76 tasks, one policy" claim for the slides.
Run when GPUs are free and analysis is done (~2.5h at 4 workers). Expected statistically identical (per-task
weight shifts by 1/76).

**EXP-6 · cluster-vs-random 7-way merge 🟢 done (2026-07-28).** 22/22 ok. **Crowding confirmed head-to-head:**
BOOK cluster (max-similar, base 86%) retains **56%** vs max-dissimilar CONTROL (base 97%) **79%** — coherent-
cluster merging costs ~23pp at matched size 7. Inside the book merge: winner-take-all over confusable goals
(caddy compartments: t34 left 97%, t17 back 80%, t26 front 17%, t8 right 3%); t19 rescued AGAIN (+30pp,
3rd time — undertrained + partners). Surprises: dissimilar is NOT free at 7-way (79% vs ~101% at pairs;
new fragile task found: t22 mug-by-caddy -80pp), and 7-way dissimilar (79%) ≈ 75-way soup (81%) -> retention
roughly plateaus after the first few partners. Plan sweep/cluster7_plan.json.

**EXP-8 · shifted triplet partition ⬜ chained (after grand eval).** [1,2,3],[4,5,6],...,[73,74,75] — 25 new
triplets, every task 1-75 gets a SECOND real-merge instance with different partners (t75's first ever).
Purpose: non-circular ground truth (>=2 instances/task) for the full-set soup-probe validation + doubles the
fragility-history dataset. ~6h; auto-launches when the grand eval finishes.

**RESULTS (2026-07-29) — the endgame runs all landed clean:**
- **EXP-7 grand eval 🟢 (90/90):** one 76-adapter soup on ALL scenes: **77.2% vs 94.2% baseline = 82%
  retention**. Final curve 95/92-93/86/82% at 2/3/4/76-way. t75 (book): 90% (-6pp) — crowding is
  goal-specific, not cluster-uniform.
- **EXP-8 shifted partition 🟢 (93/93, 0 err):** 25 new triplets -> **93% retention**, replicating the
  original partition's 92% under different groupings (n=50 triplets total).
- **Full soup-probe validation (n=76, ground truth = mean over >=2 independent merges/task):**
  **pearson +0.67 / spearman +0.68** (the n=20 preview +0.83 was optimistic). Binary screen: 61/75;
  **recall on true-fragile 18/20 (90%), 2 misses (t12, t32)**; 12 false alarms, nearly all cluster-crowded
  tasks (book/caddy/plate families) fine in small merges -> probe = conservative screen. Fig
  results/figs/soup_probe_validation.png; data results/analysis/final_validation.json.
- **Cluster-discount refinement REJECTED (2026-07-29, honest negative):** centering soup drops on weight-
  space cluster peers hurts (r +0.67 -> +0.54; recall 18/20 -> 11/20) — intrinsic fragility itself clusters
  (wine/moka, drawer), so the discount removes true signal; some false alarms are in small clusters anyway
  (t22, mug n=4). Raw probe = final screen. Disentangling crowding needs a follow-up small-merge probe on
  flagged tasks, not a cluster correction.
- 2D map of all 76 adapters: results/figs/residual_map_76.png (classical MDS of residual space, k=8 weights-
  only clusters, all semantically coherent; duplicates folded into t7/t29; Gram results/analysis/lora_gram76.npz).

**EXP-5e/EXP-7 · GRAND EVAL: 76-soup on ALL 90 unique scenes (design notes) 🔵 running.** One policy, all 76 tasks — the
clean headline number AND the complete soup-probe (predictor score for every task, extends validation n=20 ->
~75). Supersedes the queued 23-scene consistency rerun. Plan sweep/soup76_grand_plan.json (4 jobs share one
cached build), chained after the soup76-t75 single-scene eval. ~5.5h.

(EXP-6 design, for reference: book cluster t8,17,19,26,34,49,60 — most coherent, mean res-cos +0.024, base
86% — vs greedy max-dissimilar control t7,22,33,51,52,59,73 — mean res-cos -0.020, base 97%. Prediction was
control ~100% / book degrades; observed 79% / 56% — crowding right, "dissimilar = free" only partly held.)

**Semantic structure of fragility (2026-07-23) — is a fragile task the "odd one out" in its group?** NO.
- Odd-one-out test (`sweep/weight_geometry.py` pairwise cos): within group (0,1,2) all pairwise cos ~0.79 and
  the fragile t0 sits mid-pack (avg cos to mates 0.789) — robust t2 is actually *least* similar (0.787). In
  (6,7,8) fragile t6 is marginally lowest (0.799 vs 0.803/0.805) but negligibly. No consistent signal.
- Twin refutation (decisive): near-identical tasks have OPPOSITE fragility. t0 "moka pot on the stove" (fragile)
  vs t9 "...put the frying pan on it [stove]" (robust), cos=0.80. t12 "bowl on top of the cabinet" (fragile) vs
  t13 "pan on top of the cabinet" (robust), cos=0.79 — same target, deltas as similar as any group-mates.
=> Fragility is NOT semantic/relational; it's idiosyncratic to each adapter's learned solution. Partner-dependent
  too (t0: 80% pair -> 40% triplet -> 60% 4-way). Partial ceiling confound (fragile tasks ~100% baseline) but
  not the cause (t6 collapsed from 64%; t2 held at 100%).
SCOPE NOTE (superseded 2026-07-24): user redirected — the goal of this week IS mergeability prediction for
  VLAs: find a measure/method that predicts when to merge and what to merge (fragile tasks -> don't merge;
  no-harm / positive-transfer tasks -> merge). Differs from prior general-merging paper by being VLA-specific.

**Result (superseded).** On this pair, average >> ties >> task_arithmetic(lambda=1). Mechanistic
read: average = task_arithmetic at lambda=0.5; summing both full task deltas (lambda=1) over-shoots and
tasks interfere (microwave collapses to 0%). **The merge scale lambda is the dominant knob.** Next: sweep
task_arithmetic lambda (e.g. 0.3/0.5/0.7) to confirm, then advance average/ties to 5 pairs.

## EXP-10 · Localized subspace collision as the mergeability signal 🔵 running

**Motivation.** Week-2 verdict: GLOBAL linear geometry is homogeneous, but each task's residual is low-rank
per module in PRIVATE directions. Hypothesis: interference = per-module subspace collision — a merge hurts
task X exactly where partners occupy the module-level directions X depends on. Global cosines (458 modules
averaged) would hide this; module-level alignment would reveal it. Davis-Kahan framing: merge = perturbation;
survival depends on ||E|| vs local gap.

**Plan.** Phase 1 (CPU): per-stem 76x76 Grams -> per-module residual alignments; instance-level test on the
~131-instance ground truth (transfer of task-in-group, sizes 2-4): mass-weighted / max-module collision
scores vs transfer; per-task module-mass concentration vs fragility. Phase 2 (if signal): principal angles
on top modules. Phase 3 (if signal): Goal-1 GPU pilot — low-baseline tasks x {min,max}-collision partners.

**Result (2026-07-29) — EXP-10 phases 1+2: NULL. Weight space is now exhaustively falsified.**
- Phase 1 (per-module residual matrix alignment, 230 instances / all 76 tasks): every collision feature
  null (|r|<=0.11, perm p>=0.10). Partner-free: residual share r=-0.19 (p~0.10, weak), eff-#modules null.
- Phase 2 (Davis-Kahan quantities, all 458 modules x 76 tasks, dense pass): top-8 subspace projection of
  partner residuals onto task subspaces — null (p>=0.26); per-task singular gaps — null (p>=0.51);
  combined perturbation/gap ratio — null (p=0.43).
- META-FINDING (the real result): every weight-space quantity is UNIFORM across tasks — norms 59.5-60.4,
  raw cos 0.795+-0.01, eff rank 8.21-8.40, eff #modules 165-188, rel gaps 0.022-0.028, proj ratios
  0.009-0.024. Homogeneous training recipe -> statistically indistinguishable adapters in every
  linear/spectral view. Behavioral differences (fragility, transfer) are NOT encoded in weight geometry
  at any granularity we can measure -> they live in the weight->behavior map (function).
- Decision per plan: STOP, report, propose pivot to ACTIVATION-space probing (solo vs merged forward
  passes on twin/fragile tasks; find the layer where behavior diverges).

## EXP-11 · Activation-space mergeability signals ⚫🔵 running

**Motivation.** EXP-10 falsified weight geometry exhaustively; week-1 showed functional probes work
(soup-probe r=+0.67 but needs sim evals). Middle ground: ACTIVATION signatures — forward passes only,
no simulator, ~100x cheaper than evals. Two questions: (a) does activation distance between solo policies
predict merge outcomes (cheap mergeability signal for an unseen LoRA)? (b) mechanism: where (which layer,
which input) do merged policies diverge from solo on twin tasks?

**Infra (built 2026-07-31).**
- `sweep/collect_obs_bank.py`: fixed bank of 180 observations (90 scenes x 2 timesteps, exact eval
  preprocessing, policy-independent) -> results/analysis/obs_bank.npz.
- `sweep/activation_probe.py`: loads base PI0.5 ONCE; per adapter/merge applies LoRA delta in-place
  (pristine CPU copies for exact restore; paligemma lm_head stem skipped — tied to embed_tokens & unused
  at inference), runs the bank with FIXED flow noise + 3 denoise steps (deterministic, comparable),
  captures mean-pooled activations at 9 depths (vision tower, LLM 0/5/11/17, expert 0/8/17, action head)
  -> results/analysis/act_sigs/<name>.npz.
- Gotcha re-learned the hard way: pkill pattern + literal path in the SAME command = self-kill. Also
  long runs -> tmux (Bash background tasks are timeout-bound).

**Running:** 76 solo signatures + 8 mechanism merges (26:34, 24:69, 17:26 confusion; 6:64 rescue;
19:49 mild; 44:52, 3:27, 10:29 free controls) in tmux 'probe', ~10h.
**Analysis plan:** per-layer activation distances (task vs partners on task's own scenes + full bank) vs
the 230-instance ground truth; solo-vs-merged divergence localization (layer x scene) for twins.

**Results (2026-07-31/08-01, 84/84 signatures, analysis `sweep/act_analysis.py` ->
`results/analysis/act_analysis.json` + `act_analysis_report.txt`, fig `results/figs/week2_act_probe.png`).**

*Q(a) — activation distance as a cheap mergeability signal: WEAK, not usable alone.* Per-layer solo-vs-
partner distance vs transfer over all 230 instances: vision/early-LLM ~0; only the late action stack is
even directionally consistent (exp_00 spearman -0.20, exp_17 -0.15, act_out full-bank -0.19; pearson
|r|<=0.16). Combined late-stack feature: instance-level r=-0.06 (n.s.), per-task (distance to ALL 75
others vs mean transfer) r=-0.14 / spearman -0.15 (n.s., n=76). Ranking of predictors stands:
fragility-history +0.71 ~ soup-probe +0.67 >> activation distance ~0.15-0.20 (late stack only) >> weight
geometry ~0.05. The DEPTH GRADIENT (signal only in expert/action layers) is mechanistically meaningful
but the magnitude is not deployable.

*Q(b) — divergence localization on twins: two solid mechanism findings.*
1. **Merging barely changes per-step computation — even in catastrophic pairs.** Merged-vs-parent
   divergence on the parent's own scenes is 0.001-0.005 cos-distance at act_out, vs ~0.02 between two
   different solo task policies: merging moves a policy 10-20x LESS than the gap between tasks. The -93pp
   collapse (t26 in 26:34) shows act divergence 0.0011 — indistinguishable from free pairs on a static
   bank. => **Catastrophic merge failure is NOT a large static computation change; it is tiny per-step
   drift amplified by the closed loop.** Explains why static probes (weights, activations) keep losing to
   functional evals: fragility is a rollout-dynamics property.
2. **Divergence is concentrated in the late action stack.** Early LLM layers ~0.0001 (language processing
   untouched); divergence rises monotonically through expert layers to exp_17/act_out (10-30x larger).
   One outlier: t69 (24:69 butter-front) llm_17=0.0197 — the only case where merging visibly perturbs the
   language stack, in the worst mutual-confusion pair.
3. Within-pair, the parent with larger act_out divergence is the LOSING twin in **5/7** pairs (chance 3.5)
   — suggestive, n too small.

---

## EXP-12 · Goal-1 pivot: improve a WEAK task by merging partners into it 🔵 running

**Motivation.** Pivot (user, 2026-08-01): explore Goal-1 — given a weak target task, find merge recipes that
IMPROVE the target's own performance (partners are ingredients, only the target is scored). Goal-2 (soup
curation) paused, not abandoned. Evidence base: positive transfer is real (t19 +40, t56 +30, t6 +26) and
headroom-gated; rescue anatomy known (twin without placement conflict = augmentation).

**Design (24 jobs, 32 scene-evals, 10 rollouts, target's own scenes only).**
6 targets spanning the spectrum: lifted-before t19 (60%), t56 (60%), t6 (64%); mid t8 (73%), t14 (80%);
never-lifted-hard t15 (76%, best observed -36pp). Per target, 4 recipes:
- R1 `hist`: best historical partner group, uniform average (replication/stability of the lift).
- R2 `soup50`: NEW **weighted** merge — 0.5*target + 0.5*(mean of all 75 others). "Personal soup":
  universal target-protective recipe, no partner selection needed.
- R3 `heavy`: best partners but target weight 0.75 (partners split 0.25) — does protecting the target's
  residual keep the transfer gain while removing merge risk?
- R4 `ctrl`: target + fixed robust dissimilar partners (t2, t51), uniform — generic-regularization control.

**Infra.** `weighted` method added to `scripts/merge_loras.py` (streaming path, params.weights normalized
to convex combination; validated bit-exact vs `average` at uniform weights). Plan `sweep/goal1_plan.json`,
results `results/merge/g1_t<T>_<recipe>.jsonl`.

**Pilot result (2026-08-01, 32/32 ok).** Δ vs own baseline (pp), 6 targets:

| target | base | hist | soup50 | heavy | ctrl |
|---|---|---|---|---|---|
| t19 | 60% | +40 | +30 | +40 | +40 |
| t56 | 60% | +10 | -20 | +10 | -30 |
| t6  | 64% | +26 | +6 | **+36** | +6 |
| t8  | 73% | -7 | +3 | 0 | -7 |
| t14 | 80% | -10 | +10 | +10 | -10 |
| t15 | 76% | -26 | -6 | **-6** | -66 |
| mean | | +5.6 | +3.9 | **+15.0** | -11.1 |

- **`heavy` (target 0.75 + partners 0.25) wins**: 4/6 lifted, 1/6 hurt (t15 -6 = its least-bad merge ever);
  t6 64%->100% (+36, beats oracle best +26). Target-protective weighting works as the mechanism predicts.
- **`ctrl` fails (mean -11)**: generic robust partners are not enough -> lift is partner-specific... but
  target-dependently so (t19 is lifted by ANYTHING incl. ctrl +40; t56 needs its neighbors; t15 hurt by all).
- `soup50` safe but dilute (+3.9); self-weight 0.5 too low.
- Noise: hist replication ±10-20pp on single 10-rollout cells (t19/t6 replicate exactly; t56/t8/t14 lower).

**EXP-12b · full-scale (all 76 targets x 2 deployable recipes) 🔵 running.** Per user rule (all-76 reporting):
- `heavyNN`: target 0.75 + 0.25 x weight-space NN (residual-cos NN, duplicates excluded; NN(t6)=t64 = oracle).
- `soup75`: target 0.75 + 0.25 x mean-of-fleet (selection-free control at the better self-weight).
152 jobs, 188 scene-evals, plan `sweep/goal1_full_plan.json`, results `g1b_t<T>_<recipe>.jsonl`, ~12h.
Readout: how often does target-heavy merging lift weak tasks / preserve strong ones; does cheap NN selection
match oracle partners? NN map cached at `results/analysis/nn_partner.json`.

**Result (2026-08-02, 188/188 ok, 0 errors).** Data `results/analysis/goal1_full.json`, fig
`results/figs/goal1_weak_targets.png`.
- WEAK targets (base<90%, n=15): heavyNN **+5.6pp mean, 8/15 lifted** (t6 +36 -> 100%; t15 +24 — the
  never-lifted task finally lifted, via NN partner t64 history never tried); soup75 +2.6pp, 7/15 (t19 +40,
  t60 +16). STRONG (n=61): both -0.5pp mean, 9/61 hurt each (worst -20; t14<->t65 mutual-NN stacking pair
  hurts BOTH sides — EXP-4b twin-conflict signature).
- **Recipes are complementary (per-target corr +0.18)** — different targets need different partners.
- **Menu recipe (build both, 2 evals/task, keep best w/ baseline fallback): +12.1pp mean on weak, 10/15
  lifted, 0 hurt by construction.** The practical Goal-1 deliverable: train-free, ~2 evals per task.
- Caveats: ±10-20pp single-cell noise; menu selection reuses the noisy eval (mild winner's curse — clean
  version re-evals the winner); some strong-task drops are real (t65 -20 in both recipes).
**Verdict: Goal-1 is promising** — target-heavy merging lifts weak tasks with zero training; next options:
per-target recipe chooser (or accept 2-eval menu), winner re-eval for unbiased numbers, weight sweep
(0.6/0.75/0.9), or partner search beyond NN/fleet for the stubborn targets (t56, t50, t14).

**Menu-vs-oracle gap (2026-08-02, analysis).** Oracle = best delta per weak target over ALL known recipes
(230 historical instances + 6 pilot recipes + 2 full-scale recipes). **Menu mean +12.1pp vs oracle +15.2pp
— gap only 3.1pp; 12/15 weak targets have ZERO gap.** Residual gap concentrates in t56 (+30 only via its
specific neighbors t55/t57), t14 (+10 via uniform 12:13:14), t9 (+5). => A smarter selector can buy at most
~3pp on average / 2 real targets — the 2-eval menu is already near-optimal. Partner-search (step 3) is an
afternoon, not a week.

**EXP-12c stage A result (2026-08-02, 16/16 @25 rollouts) — winner's curse was REAL and large.**
Unbiased menu mean on weak targets: **+12.1pp (10-rollout selection) -> +7.2pp (25-rollout re-eval)**;
still 10/15 lifted >+5pp. Per-target: t19 +40->+24 (real, large), t6 +36->+20 (real: 64->84%), t73 +2->+12,
t60 +16->+8, t33/t63/t68/t20 ~+8, t9 +6, t8 +5. **RETRACTION: t15 +24 -> -28** — the "never-lifted task
finally lifted" claim was a 10-rollout fluke; its selected merge actually HURTS it. Deployment lesson:
the menu's selection eval must use >=25 rollouts (a noisy selector can pick a harmful merge).
**Stage B result (2026-08-02, 76/76 ok) — the two recipes want OPPOSITE doses.**
Self-weight means on the 15 weak targets (10 rollouts):

| recipe | w=0.60 | w=0.75 | w=0.85 |
|---|---|---|---|
| heavyNN | -4.1pp | **+5.6pp (peak)** | +0.9pp |
| soup75  | +0.4pp | +2.6pp | **+5.8pp (rising)** |

- **heavyNN peaks sharply at 0.75**: more twin (w60) triggers conflicts (t50 -52, t60 -44, t73 -28);
  less twin (w85) loses the lift (t6 +36->+6, t15 +24->-6). 0.25 twin dose is the sweet spot.
- **soup wants MORE self-weight**: monotone rising to w85 (+5.8pp, best soup variant); fleet dose 0.15
  beats 0.25/0.40. Untested w=0.90 might be better still.
- **Two stubborn targets finally responded to dose**: t50 soup85 **+28pp** (first-ever positive) and
  t14 soup85 +10pp — their problem was dose, not partners. t56 remains unliftable (best +10 noise-level).
- 6-option menu (2 recipes x 3 weights, biased 10-rollout selection): +16.5pp; with the measured ~40%
  winner's-curse shrinkage, honest expectation ~+10pp — the menu grows with more options but every
  selection MUST be confirmed at >=25 rollouts (the t15 lesson).
EXP-12c complete. Goal-1 exploration verdict: real, deployable, honest numbers ~+7-10pp mean on weak
tasks with confirmed individual wins up to +24pp (t19), 64->84% (t6), and dose-tuning rescuing t50/t14.

---

## EXP-13 · Committing to the FLEET recipe: dose, unbiased confirmation, safety 🔵 running

**Motivation.** User decision: pick ONE menu — the fleet (soup) recipe, for operational simplicity (no
selector, no twin-conflict mode). Before broad claims, close the gaps: (a) dose curve unbounded (rising at
0.85, 0.90 untested); (b) w=0.85 numbers are unconfirmed 10-rollout cells; (c) NO safety data on strong
targets at the operating dose. Held-out "new task" split (d) deferred (user).

**Design (3-stage tmux chain `sweep/chain_exp13.sh`, auto w* selection `sweep/exp13_make_bc.py`).**
(a) soup w=0.90 x 15 weak targets @10 rollouts -> completes the dose curve {0.60,0.75,0.85,0.90};
w* = argmax of the 15-target mean (fixed dose, no per-target tuning — method-level recipe).
(b) ALL 15 weak @ w* @ **25 rollouts** -> the unbiased headline number for the committed recipe.
(c) ALL 61 strong @ w* @10 rollouts -> safety screen at the operating dose.
Results: g1w_*_w90 / e13b_* / e13c_*; dose record results/analysis/exp13_dose.json.
Known cost of fleet-only (documented): forfeits twin-only wins (t6 +20 confirmed is heavyNN-only).

**RESULT (2026-08-03) — EXP-13 complete, 15+61 targets, 0 unresolved failures.** Data
`results/analysis/exp13_final.json`, fig `results/figs/exp13_fleet_recipe.png`.
- **Dose is a TRUE PEAK at w*=0.85**: mean pp on weak = +0.4 (w.60) / +2.6 (.75) / **+5.8 (.85)** /
  +1.0 (.90). Not an extrapolation — bracketed on both sides.
- **Weak targets @25 rollouts (unbiased, fixed dose, NO selection): mean +8.7pp, 9/15 lifted >+5pp,
  ZERO hurt.** t19 +32 (60->92%), t50 +16 (first-ever positive), t33/t68/t9 +12, t20 +10, t15/t73/t8 +8,
  t56/t60/t67/t14 +4, t63 0, t6 -4 (the documented twin-only task).
- **Strong targets @10 rollouts (safety): mean +0.8pp, 55/61 flat, 3 lifted, 3 hurt** (t69 -20, t45 -6,
  t17 -6). Higher self-weight helped safety as predicted (w.75 was -0.5pp/9-hurt).
- **Fleet-wide: 94.5% -> 96.8% mean success, +2.3pp/task, applying ONE recipe blindly to all 76 tasks.**
- Operational notes: 1 build failure from a full root disk (freed 7.5GB: uv cache + rollout videos) and
  1 EGL init failure; both fixed by re-running the single job (resumable orchestrator).
**Deliverable: `merged = 0.85*own_LoRA + 0.15*mean(all other LoRAs)` — train-free, no partner selection,
no eval needed to apply. Lifts weak tasks +8.7pp, leaves strong tasks alone (+0.8pp).**

---

## EXP-14 · Validating the recipe honestly: safety recheck + HELD-OUT generalization 🔵 running

**Motivation.** Two gaps in the EXP-13 claim. (1) The 3 "hurt" strong tasks were single 10-rollout cells.
(2) **Every design choice (fleet-vs-NN, dose 0.85, >=25-rollout rule) was made by looking at the 15 weak
tasks** — so those 15 are a DEV set, and the +8.7pp is a dev-set number. A method claim needs tasks that
influenced nothing.

**Design (3 stages, chained in tmux `sweep/chain_exp14.sh`).**
- (i) `e14a` recheck: t69/t45/t17 @25 rollouts (the 3 apparent strong-task casualties).
- (ii) `e14b` **held-out test**: the frozen recipe applied to the **MID band** = 20 tasks with baseline
  90-97% (headroom to move, and NEVER used for any design choice), @25 rollouts. DEV=15 weak / TEST=20 mid.
- (iii) `e14c` **novel-family test**: same 20 tasks, but the fleet EXCLUDES each target's "family"
  (partners sharing a physical scene or >=2 content words; 1-22 partners removed, fleet 53-72 adapters).
  Simulates a genuinely new LoRA whose relatives are NOT already in the fleet — the strongest available
  proxy for "a task the fleet has never seen", short of training a new adapter.

**Preliminary (from EXP-13 stage-c 10-rollout data, same tasks):** MID band +3.3pp (3 lifted / 15 flat /
2 hurt), ceiling band (>=97%, n=41) -0.5pp. Positive sign on held-out tasks; 25-rollout confirmation running.

**RESULT (2026-08-04, all stages done; data `results/analysis/exp14_final.json`).** GPU2 faulted mid-run
(hardware, see below) -> reran everything on GPU1 x2 workers; 47/47 evals recovered.

*(i) Safety recheck @25 rollouts:* t69 **-20 -> -8pp** (real but half as bad), t45 -6 -> **0pp** (was noise),
t17 -6 -> **-8pp** (real, consistent). Honest safety statement: **~2 of 61 strong tasks lose ~8pp**; the
alarming -20 was a 10-rollout artifact.

*(ii) HELD-OUT TEST (20 tasks that influenced NO design choice, @25 rollouts): +1.5pp*, 12 up / 4 flat /
4 down, sign p=0.077, t p=0.072 — positive but only marginally significant. Context: these tasks average
95.1% baseline, so max possible mean gain was +4.9pp -> **realized 30% of available headroom**.
**CONFOUND (must state in any writeup): the dev/test split is confounded with headroom.** All 15 weak tasks
were used for tuning, so there exists NO held-out weak task -> **the +8.7pp weak-task headline is measured
on the same tasks that shaped the recipe and is NOT independently confirmed.**

*(ii-b) Leave-one-out CV of the dose choice (free re-analysis, 10-rollout data):* 14/15 folds still pick
w=0.85, but the LOO estimate is **+2.4pp vs +5.8pp in-sample**. The entire gap traces to ONE influential
task: excluding t50 flips the choice to w=0.75 (t50 = +28pp at 0.85 but -22pp at 0.75). Dose choice is
stable in form yet **materially hinges on a single data point**.

*(iii) NOVEL-FAMILY TEST — the strongest positive result:* rebuilding each fleet with the target's entire
family removed (1-22 partners dropped; fleets of 53-72 adapters) changes **nothing**: +1.87pp (no-family)
vs +1.47pp (full fleet), paired diff **+0.40pp, sign p=0.79, t p=0.66**; corr(family size removed,
degradation) = **+0.08**. => **The benefit does NOT come from having the task's relatives in the fleet** —
it comes from the generic shared backbone (consistent with the EXP-5 mechanism: all adapters share the
~0.795 direction). A genuinely new LoRA, whose task family the fleet has never seen, gets the same gain.

**Net honest claim after EXP-14:** the recipe is (a) SAFE — ~2/61 mild regressions, no catastrophes;
(b) generalizes in KIND — held-out tasks gain +1.5pp (~30% of their available headroom) and the gain is
family-independent, so it transfers to genuinely novel adapters; (c) its LARGE weak-task effect (+8.7pp)
remains dev-set-measured and unconfirmed out-of-sample, because no held-out weak task exists.
**Infrastructure note: GPU2 (0000:81:00.0) faulted — "Unable to determine the device handle", workers on it
failed 31/31 evals while GPU1 workers succeeded 16/16. Needs admin `nvidia-smi -r -i 2` or reboot. Root
disk also at 99% (/opt/miniconda = 183GB).**

---

## EXP-15 · External adapters: held-out validation + joint-training vs merging 🔵 running

**Motivation.** Two gaps EXP-14 could not close with our own assets. (1) All 15 weak tasks were used to
tune the recipe (dose 0.85, fleet-vs-NN) -> no held-out WEAK task exists (only 1 weak *scene* inside a
never-tuned task; not enough). (2) We have never had a **jointly-trained multi-task reference** to compare
merging against. Both are solved by adapters already on the NAS.

**The assets (`/media/pinas/Chris/pi05_libero_lora_experts/`, 8 experts, verified compatible:**
r=16, alpha=16, no rslora, **identical 17 target modules**, same `pi05_base_pytorch` base -> our merge code
runs unmodified). Trained by cbianch1 (Jan 2026), batch 256, peak_lr 5e-5, ~6-10k steps:
- **4 single-suite experts** — spatial / object / goal / libero_10 (10 tasks each). Suites are DISJOINT
  from our libero_90 fleet (LIBERO-90 = pretrain split; spatial/object/goal/10 = eval splits).
  libero_10 = long-horizon two-step tasks, typically hardest -> our best shot at a held-out WEAK adapter.
- **4 combination experts** — jointly trained on 10+goal, goal+object, object+10 (20 tasks) and
  10+goal+object (30 tasks). NOTE: the dir `pi05_libero_baseline-spatial` is MISNAMED — its metadata says
  `libero_10_goal_object` (tell Chris).

**Experiments (E0 -> E1 + E2 selected by user; E3/E4/E5 kept in reserve).**
- **E0 (running):** norm-stats probe (which of 4 candidate norm_stats files is correct for these suites —
  the EXP-1 lesson: wrong normalization = silent 0%), then solo baselines for the 4 single-suite experts.
- **E1:** frozen recipe `0.85*expert + 0.15*mean(our 76 libero_90 adapters)` on each single-suite expert,
  evaluated on its own suite. **Fully out-of-sample**: different trainer, run, and task distribution;
  zero involvement in any design choice.
- **E2:** **joint training vs weight merging** — for a combination suite A+B, compare Chris's jointly
  trained A+B expert against our uniform merge of the two single-suite experts A and B, on the same 20
  tasks. First real jointly-trained upper bound for the project. Caveat: step counts not matched
  (singles 6-10k, combos 9-10k).

**Infra added.** Orchestrator now supports per-job `adapter_paths` (external adapters), `norm_stats`, and
`suite` (evaluates any LIBERO suite, not just libero_90); suite task names cached in `sweep/suite_tasks.json`.
GPU2 (0000:81:00.0) still faulted -> running single-GPU, 2 workers.
**RESULT E0+E1 (2026-08-05, 80/80 evals) — the recipe does NOT transfer to external adapters.**

*Norm-stats correction first (critical).* Initial baselines were 34-45% (implausible; several tasks at 0%).
Root cause = the EXP-1 failure mode again: we used mehdi's libero90-family norm stats for suites they were
not computed on (**state q01 differs by up to 2.1**). Found the correct per-dataset stats at
`/media/pinas/Chris/assets/siamaky1984/<repo_id>/norm_stats.json`; invalid runs archived to
`results/merge/invalid_wrongnorm/`. With correct stats baselines are sane -> diagnosis confirmed.

| suite | expert solo | + frozen recipe | delta |
|---|---|---|---|
| LIBERO-Spatial | 93.0% | 98.0% | +5.0 |
| LIBERO-Object | 98.0% | 98.0% | 0.0 |
| LIBERO-Goal | 62.0% | 62.0% | 0.0 |
| LIBERO-10 (long-horizon) | 87.0% | 79.0% | **-8.0** |
| **all 40 external tasks** | **85.0%** | **84.2%** | **-0.8 (n.s., sign p=0.63)** |

Weak external tasks (solo<90%, n=10): 50.0% -> 51.0% (+1.0pp) — **no lift**. Strong (n=30): -1.3pp.
=> **On genuinely external adapters the recipe gives nothing.** The +8.7pp weak-task effect does not
generalize beyond our own fleet.

*MECHANISM (CPU analysis, Gram-trace cosines) — and it is decisive:*

| adapter pair | mean cosine |
|---|---|
| within OUR fleet (libero_90, 76 adapters) | **0.791** |
| OUR fleet vs Chris's experts | **0.117-0.186** |
| among Chris's own experts | 0.271 (0.12-0.16, except one anomaly) |

**The ~0.79 "shared backbone" is an ARTIFACT OF HOMOGENEOUS TRAINING (same data, recipe, run) — not a
universal property of LoRAs on this base model.** Chris's experts share almost nothing with our fleet
(~0.13), so adding 0.15 of our fleet-mean adds noise rather than reinforcing a common direction. This
retro-explains every earlier result: EXP-5's backbone, EXP-13's recipe, and EXP-14(iii)'s
family-independence were all *within-fleet* phenomena; the true boundary is the training distribution,
not the task family.

**=> Deployment scope, honestly stated: the recipe applies to adapters from the SAME training pipeline.
Pre-flight check (free, eval-free): cos(new adapter, fleet-mean). ~0.8 -> apply; ~0.1 -> do not.**

*Data-hygiene flag for Chris:* `spatial_expert` vs `goal_expert` cosine = **+0.934** (all other expert
pairs 0.12-0.16) — implausibly similar for different datasets; possible resumed-from checkpoint or a
mislabeled save. Second such issue after the misnamed `pi05_libero_baseline-spatial` dir.

**E2 (joint training vs merging) still running** — unaffected by the above, since both its arms are
Chris's own adapters with matched norm stats.
**RESULT E2 (2026-08-06) — joint training vs weight merging, goal+object, 20 tasks. Full 2x2 over
normalization (both arms x {union, per-suite} norm stats), 80 evals.**

| | union norm stats | per-suite norm stats |
|---|---|---|
| jointly-trained (Chris's goal-object expert) | **94.5%** | 28.5% |
| merged (goal expert + object expert, uniform average) | **18.0%** | 9.5% |

specialist reference (each expert on its own suite, own stats) = 80.0%.
**Best case per method: joint 94.5% vs merged 18.0% -> gap -76.5pp.** Zeros: merged 12/20 (union) and
15/20 (per-suite); joint 0/20 (union).

*The normalization confound is RESOLVED — the merge failure is real.* Giving the merged model its parents'
own per-suite stats makes it WORSE (18.0 -> 9.5%), so it is broken under both normalizations; no
normalization choice rescues it. Meanwhile the control is itself informative: the joint expert drops
94.5 -> 28.5% under mismatched stats, empirically confirming **every model must be evaluated with the norm
stats it was trained with** (and retro-validating E0/E1, where each expert used its own training stats).

*Unified mechanism — this is the project's central finding:*

| adapters | mutual cosine | merging outcome |
|---|---|---|
| our libero_90 fleet (76, homogeneous training) | **0.79** | 3-way retains **92%** |
| Chris's suite experts (independent training runs) | **0.15** | collapses to **18%** (vs 94.5% joint) |

**Weight-averaging LoRAs works if and only if the adapters share a dominant common direction, and that
sharing is produced by homogeneous training (same data/recipe/run) — NOT by the shared base model.**
Note the task content overlaps across the two groups (both have ketchup-in-basket, bowls-on-plates, moka
pots) yet the adapters are near-orthogonal: **the backbone is about training homogeneity, not semantics.**
Joint training also beat both specialists (94.5% vs 80.0%; Goal 62 -> 96%), so on independently-trained
adapters, retraining on the union is not just better than merging — it adds real capability.

**Scope statement for the whole project:** every merging result we have (EXP-3..14: retention curves,
soup, recipe, family-independence) is a WITHIN-FLEET phenomenon. Outside a homogeneous fleet, merging
fails and the recipe does nothing. Free pre-flight test: cos(new adapter, fleet-mean) ~0.8 -> merge;
~0.1 -> do not.

---

## EXP-16/17 · Joint-vs-merge replications + the 3-way 🔵 (2 of 3 done)

**Motivation.** The E2 gap (-76.5pp) rested on ONE pairing. Replicate on the other two 2-suite combos and
test whether the gap widens at 3 adapters (external analogue of our within-fleet retention curve).
Rule applied throughout: every model evaluated with its own training norm stats.

**e16 10+goal (2026-08-06, 40/40):** specialist 74.5% | joint **87.0%** | merged **13.0%** (-74.0pp;
merged zeros 15/20, joint 0). Joint lifts Goal 62->97% but COSTS LIBERO-10 87->77% — the union is not
free for every member. LIBERO-10 (highest-cosine expert, 0.19) is no exception: merged -> 3%.

**e17 3-way 10+goal+object (2026-08-07, 60/60):**

| | LIBERO-10 | Goal | Object | ALL 30 |
|---|---|---|---|---|
| specialist | 87.0% | 62.0% | 98.0% | 82.3% |
| joint | 81.0% | 94.0% | 98.0% | **91.0%** |
| merged | 2.0% | 17.0% | 2.0% | **7.0%** |

**Gap -84.0pp; merged zeros 24/30, joint 0/30.** Merged degrades further with a third adapter
(2-way: 18/13% -> 3-way: 7%). **Contrast with within-fleet: our 2/3/4-way retention was 95/92/86% —
graceful; across independent training runs merging fails catastrophically immediately and worsens with N.**

**Operational notes.** e16_object_10 first crashed on a missing norm-stats path — that expert was trained
under a DIFFERENT HF org (`Fascetta/libero_object_10`, all others `siamaky1984/*`); correct stats found at
`Chris/dragdrop-vlas/pi05_libero_lora/Fascetta/libero_object_10/`. Relaunch progressed 16/40, then the
machine was REBOOTED (2026-08-10, which also RESTORED GPU2 — both P6000s healthy again); resumed on both
GPUs, 4 workers. Asset checks: the 3-way expert's `9000` and `10000` checkpoint dirs are byte-identical
(same md5) — a duplicate save, flagged to Chris along with the misnamed dir and the 0.934 spatial~goal
similarity.

---

## EXP-18 · Explicit shared/residual decomposition of LoRAs (LIBERO-90) 🔵 running

**Motivation (user proposal, 2026-08-10).** All evidence for "shared backbone + private residual" is so far
inferred (EXP-5 mechanism, EXP-13 recipe, EXP-15/16/17 collapse without it). Make it explicit and
constructive: leave-one-out soup -> SVD -> top-k dominant directions -> if a held-out LoRA concentrates its
energy in the OTHERS' directions, those directions are truly global -> split any LoRA into shared + residual.
Payoff: uniform souping = shared + mean(residuals) — shared is fine, residuals are diluted 1/N (what kills
fragile tasks). Explicit split unlocks the operator averaging cannot express:
**W = shared (once) + Σ residuals at FULL strength** (plausible: residuals of dissimilar tasks are
near-orthogonal, EXP-4). User question to answer: merge shared or residual, of similar or dissimilar tasks?

**Design.**
- **Phase A (CPU) — globality test.** Held-out t in {t1, t2 anchor, t6 fragile, t19 lifted}: LOO-soup of the
  other 75; per-module SVD (458 modules, factorized QR trick — never materialize out x in); k-sweep
  {1..64} of energy captured in the held-out LoRA by soup top-k (two-sided projection U_k U_k' D V_k V_k').
  Baselines that make it a real test (projection bookkeeping alone is an identity): chance floor
  k^2/(out*in), held-in adapters (what "global" looks like), own top-k (ceiling). Globality verdict =
  held-out curve lands on held-in curve >> chance. Saves top-64 soup factors for t1/t6 (Phase B inputs).
  Script `sweep/loo_decomposition.py` -> `results/analysis/loo_decomposition.json` + report.
- **Phase B (GPU) — causal interchangeability.** Rebuild t1/t6 as `soup-shared(top-k) + own residual`
  (shared taken from the OTHER 75) via synthetic low-rank adapters; eval on own task. Performance preserved
  => shared truly interchangeable.
- **Phase C (GPU) — the payoff operator.** Structured merge `shared + Σ residual_i` (full strength) vs
  plain average: one dissimilar group, one twin pair. Hypothesis: shared is trivially mergeable (identical
  in everyone); dissimilar residuals sum freely (orthogonal); twin residuals collide (= EXP-4b confusion).

**Status.** Phase A chained behind the pending e16_object_10 rerun (user: wait for pending jobs first);
runs in tmux 'phaseA', CPU-only. Prediction from known geometry (0.79 mutual cosine): soup top-k captures
~75-80% of a held-out adapter's energy at small k, residual ~20-25%.
**Phase A RESULT (2026-08-10) — globality CONFIRMED + a structural discovery.** Report
`results/analysis/loo_decomposition_report.txt`, data `loo_decomposition.json`.

Capture of a held-out adapter's energy by its LOO-soup's top-k directions (energy-weighted over 458 modules):

| k | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| held-out (t1/t2/t6/t19 range) | 0.35-0.36 | 0.55-0.56 | **0.79-0.80** | 0.80-0.82 |
| held-in reference | 0.36 | 0.56 | 0.81 | 0.86 |
| chance | ~0 | ~0 | 0.0002 | 0.003 |

- **Held-out curves land exactly on the held-in curve** (gap <=2pp at every k, ~3 orders above chance) for
  all four task types (anchor/fragile/lifted) -> the soup's dominant directions are TRULY GLOBAL; the
  shared/residual split is well-defined: **shared ~80% of any adapter's energy at k=16, residual ~20%** —
  matching the prediction from the 0.79 mutual cosine.
- **DISCOVERY: the soup is EXACTLY rank-16 per module** (soup self-concentration = 1.000 at k=16). The mean
  of 75 rank-16 adapters collapses back to rank 16: the backbone occupies the SAME 16-dim subspace in every
  adapter (matching each LoRA's own rank budget), and the residuals cancel in the mean. k=16 is therefore
  not a choice but the natural, intrinsic dimension of the shared component.
- Held-out ~= held-in also shows membership in the soup contributes ~nothing (1/75 weight) — LOO vs full
  soup is immaterial for the decomposition.

**Phase B launched (2026-08-10):** synthetic adapters built (`sweep/build_decomp_adapters.py`, algebra
verified: interchange = shared+residual to 1e-9; ||P(residual)||/||residual|| = 3.6e-4) for t1/t6 x
{interchange = others'-shared + own-residual, residual-only, shared-only}, k=16; 6 evals @25 rollouts.

**EXP-16 object+10 FINAL (2026-08-10, 40/40 after LIBERO-reinstall recovery):** specialist 92.5% |
joint **80.5%** | merged **33.0%** (gap -47.5pp; merged zeros 6/20, joint 0/20). Completes the set:

| combo | specialist | joint | merged | gap |
|---|---|---|---|---|
| goal+object | 80.0% | 94.5% | 18.0% | -76.5pp |
| 10+goal | 74.5% | 87.0% | 13.0% | -74.0pp |
| object+10 | 92.5% | 80.5% | 33.0% | -47.5pp |
| 3-way 10+goal+object | 82.3% | 91.0% | 7.0% | -84.0pp |

Notes: object+10 is the mildest case (merged 33%, not near-zero) AND the only combo where joint
UNDERPERFORMS the specialists (80.5 vs 92.5 — this joint expert, trained on the Fascetta dataset variant,
is the weakest of the four). Robust conclusion across all four: merged 7-33% vs joint 80-94%.
EXP-16/17 🟢 done.
**All-76 capture (2026-08-10, `sweep/full76_capture.py`) — the rule-compliant version.** Capture@k=16 under
the full-76 soup directions, ALL 76 adapters: **mean 0.803, sd 0.011, range [0.768 (t45), 0.830 (t41)]** —
residual fraction 0.197 +- 0.011, range [0.170, 0.232]. ANOTHER uniformity: every adapter, including all
fragile ones (t0 .203, t6 .199, t15 .207, t50 .212), has ~20% residual. corr(residual fraction, mean merge
transfer) = -0.19 (weak) => residual SIZE doesn't predict fragility either — consistent with EXP-9/10:
what matters is whether the task's SUCCESS depends on its residual, not how big it is. Strict 76-fold LOO
queued for paper-grade purity (6 shards; restarted after server reboot killed the first attempt —
2nd reboot today; LIBERO's new persistent install at /home/lzhou/libero-local survived it, fix validated).

**Phase B RESULT (2026-08-10, 6/6 @25 rollouts) — the shared component is CAUSALLY INTERCHANGEABLE.**

| variant | t1 (base 100%) | t6 (base 64%) |
|---|---|---|
| interchange (OTHERS' shared + own residual) | **100%** | **72%** |
| residual only | 0% | 0% |
| shared only | 76% | 24% |

- **Interchange = full performance** (t6 even +8 over baseline): swapping a task's shared component for one
  built purely from the other 75 adapters costs NOTHING. The decomposition is causally validated.
- Residual alone = 0% everywhere (needs the backbone; consistent with EXP-5 solo-lambda).
- Shared alone: 76% for easy t1 vs 24% for fragile t6 — tasks differ in how much they live on the backbone;
  fragile tasks are exactly those needing their residual (t6: 24% -> 72% when its residual is added).

**Phase C launched (2026-08-10):** structured operator W = S_16 + SUM residual_i (full strength, no dilution)
via `sweep/build_structured_merge.py` (shared from t1's LOO factors — valid: interchangeability + LOO~full).
Groups: twin 24:69 (avg ref: t24 -50, t69 -30), dissimilar 5:33 (avg ref: clean), triplet 12:13:14
(avg ref: 80/100/90%). 7 evals @25.
**Strict LOO-76 RESULT (2026-08-11, 76/76):** capture@16 mean **0.798 sd 0.011**, range [0.762 (t45),
0.825]. Max |diff| vs the full-soup shortcut = **0.006** -> the shortcut was valid; Phase A table is now
paper-grade (all 76 tasks, strict leave-one-out).

**Phase C RESULT (2026-08-11, 7/7 @25 rollouts) — the undiluted-residual operator FAILS; dilution is
load-bearing.** W = S_16 + SUM residual_i (full strength) vs plain average:

| group | task | base | plain avg | STRUCTURED |
|---|---|---|---|---|
| twin 24:69 | t24 | 100% | 50% | **36%** |
| | t69 | 100% | 70% | **64%** |
| dissimilar 5:33 | t5 | 100% | 100% | 100% |
| | t33 | 88% | 90% | 88% |
| triplet 12:13:14 | t12 | 96% | 80% | **0%** |
| | t13 | 100% | 100% | 100% |
| | t14 | 80% | 90% | **52%** |

- Structured NEVER beats averaging; in conflict cases it is far worse (t12: 80% -> 0%). Weight-space
  near-orthogonality of residuals does NOT imply functional non-interference (the project's recurring
  lesson, now at the operator level).
- *Mechanism, and it unifies everything:* from task t's perspective, plain averaging costs it dilution of
  its OWN residual (1/N) but protects it from FOREIGN residuals (also 1/N). The structured operator restores
  the own residual but injects N-1 foreign residuals at FULL strength — and foreign-residual interference
  empirically dominates the own-dilution benefit. **There is an inherent trade-off: one policy cannot give
  every task full own-residual without full foreign interference.**
- This retro-explains EXP-13: the per-task 0.85 recipe = full shared + 0.85 own residual + ~0.002 foreign
  residuals — i.e., it already sits at the sweet spot of this trade-off, which is only reachable with
  PER-TASK policies, not a single merged one. The soup's dilution is not a flaw; it is the protection.

**EXP-18 COMPLETE (A+B+C).** Arc: decomposition validated at weight level (all-76, strict LOO), causally
(shared interchangeable; fragility = residual-dependence), and the operator space mapped: single-policy
merging has an inherent own-vs-foreign residual trade-off; per-task weighted recipes are the optimum.
**Phase C2 launched (2026-08-11) — two new operators in the decomposition's coordinate system.**
Same 3 reference groups (twin 24:69, dissimilar 5:33, triplet 12:13:14); all references already measured
(baseline, plain avg = residual coeff 1/N, structured = coeff 1).
- **O1 sqrt-scaling:** W = S_16 + (1/sqrt(N))*SUM residual_i — the principled midpoint (orthogonal residuals
  sum to a single-residual norm at 1/sqrt(N)); maps the alpha trade-off curve between the two tested extremes.
- **O2 need-weighting:** W = S_16 + SUM c_i*residual_i with c_i = max(1/N, need_i/max-need), where
  **need_i = baseline_i − grand-soup-score_i** (backbone-dependence, ALREADY MEASURED in week 1 for all 76).
  Reallocates the interference budget: backbone tasks contribute little residual, needy tasks keep theirs.
  Group needs: t24 20pp, t69 40pp | t5 0, t33 48pp | t12 6pp, t13 0, t14 30pp -> coeffs [.5,1] / [.5,1] /
  [.33,.33,1].
6 builds (`o1_sqrt_*`, `o2_need_*` in decomp_adapters/), 14 evals @25 (`e18d_*`).
**Residual pairwise cosines (2026-08-11, `results/analysis/residual_cosines.npz`).** Cosines between the
NEW subspace-orthogonal residuals (D_t − P_16(D_t)), all 76 tasks: **mean −0.013, sd 0.030, max +0.101**
(butter twins 24:69; then book/pan twins — the same semantic ranking as the old double-centered measure,
which this closely reproduces). Decisive for Phase C's interpretation: the failed triplet 12:13:14 has
residual cosines of −0.005/−0.006/−0.015 — *exact orthogonality* — and t12 still collapsed to 0% under
full-strength residual summing. **Functional interference with zero weight-space overlap**, the sharpest
statement yet of the geometry≠behavior lesson; conversely the only "high" residual alignments (~+0.1) are
precisely the confusion twins.
**Phase C2 RESULT (2026-08-11, 14/14 @25) — the operator family is mapped: plain averaging is its optimum.**

| task | base | avg (1/N) | O1 (1/sqrtN) | O2 (need) | struct (1) |
|---|---|---|---|---|---|
| t24 | 100 | 50 | 68 | **0** | 36 |
| t69 | 100 | 70 | 72 | **88** | 64 |
| t5 | 100 | 100 | 100 | 96 | 100 |
| t33 | 88 | 90 | **96** | **96** | 88 |
| t12 | 96 | 80 | 12 | 16 | 0 |
| t13 | 100 | 100 | 100 | 100 | 100 |
| t14 | 80 | 90 | 72 | 60 | 52 |
| **mean** | 95 | **83** | 74 | 65 | 63 |

- **The alpha-curve is monotone**: 83 -> 74 -> 63 as the residual coefficient goes 1/N -> 1/sqrtN -> 1.
  No sweet spot between averaging and full strength; within `W = S_16 + sum c_i r_i`, uniform 1/N wins.
- **Need-weighting exposes the zero-sum (worse) nature of residual budget**: it gave t69 its best result of
  ANY operator (88%) — while sending its twin t24 to 0%. Boosting one residual actively poisons neighbors;
  interference is not a budget that can be reallocated, and the mean drops (65 < 83).
- **t12 is exquisitely foreign-residual-sensitive**: cosines to partners ~0, own need only 6pp, yet ANY
  elevated residual load kills it (avg 80% -> 12/16/0% at higher coefficients).
- Dissimilar pairs tolerate everything (t33 even gains: 96% under O1/O2) — interference concentrates in
  shared-context (twin) groups and around sensitive tasks.
**EXP-18 CLOSED.** The decomposition is real, causal, and diagnostic — but within its natural operator
family, the humble soup is provably (empirically) the optimum. Improvements to single-policy merging must
come from outside weight-space recombination (consistent with EXP-11: interference is functional/dynamic).
Remaining unexplored: O3 (TIES/DARE on residuals), O4 (per-module conflict surgery) — expectations now low
given exact-orthogonality interference; noted as options, not scheduled.
**The 2x2 factorial RESULT (2026-08-11, 8/8 @25) — the original week question answered.** Each cell merges
ONE component with ONE partner type, holding the other component intact (own):

| delta vs baseline | SIMILAR (twin 24:69) | DISSIMILAR (5:33) |
|---|---|---|
| **SHARED merged** | -4.0pp (96/96%) | +2.0pp (100/92%) |
| **RESIDUAL merged** | **-12.0pp** (92/84%) | +2.0pp (100/92%) |

- **Dissimilar column: everything is free** (+2pp both cells; t33 even gains, consistent with its 48pp
  backbone-need). **Similar column: shared is ~free (-4), residuals cost (-12)** — twin confusion lives
  specifically in the residuals, as hypothesized on day 1 of the week.
- **Super-additive interaction:** full plain-average of the same twin pair = BOTH cells at once = -40pp
  (measured earlier at 10 rollouts), far worse than the additive -16 of the two isolated merges. Keeping
  EITHER component intact protects substantially; blending both compounds. (Caveat: -40 is a 10-rollout
  number; the factorial is 25.)
- **Answer to the week's question: merge shared components with anyone; merge residuals only across
  dissimilar tasks; the single dangerous cell is residual x similar.**
Builds `m22_*`, results `e18e_*`, fig `results/figs/week3_2x2_matrix.png`.
**FULL 2x2 MATRIX RESULT (2026-08-12, 76/76 evals @25, 16 pairs, n=16 task-instances/cell).**
Driver `sweep/matrix22_driver.py` (self-driving: build -> eval -> DELETE per pair; the 750MB synthetic
adapters filled the disk when built upfront). Report `results/analysis/matrix_2x2_full.txt`.

| delta vs own baseline | SIMILAR (8 twin pairs) | DISSIMILAR (8 pairs) |
|---|---|---|
| **SHARED merged** | **+0.6 +-1.1** | **+1.3 +-1.5** |
| **RESIDUAL merged** | **-8.2 +-6.3** | **-1.0 +-3.3** |

- **The n=1 pilot's -4pp shared-x-similar cell was NOISE, as suspected**: at n=16 it is +0.6 +-1.1.
  **Shared components merge free with ANY partner** — three independent cells all ~0/positive.
- **Residual x similar has a huge SE (+-6.3) because the twin column is BIMODAL**, and splitting it by the
  EXP-4b anatomy is decisive:

| | shared merged | residual merged |
|---|---|---|
| confusable twins (same scene, differ by placement: 24:69, 26:34, 49:75, 68:74) | +1.2 +-1.8 | **-19.3 +-11.2** |
| benign twins (distinct motion/receptacle: 6:64, 3:27, 1:55, 44:52) | +0.0 +-1.3 | **+3.0 +-3.3** |
| dissimilar (8 pairs) | +1.3 +-1.5 | -1.0 +-3.3 |

  Worst cells: 26:34 book-compartments **-43pp**, 24:69 butter back/front -16, 49:75 -14. Best: the RESCUE
  twin 6:64 (pan ON vs UNDER shelf) **+10pp** from residual merging — it *wants* its twin's residual.
  This is the week-1 confusion/rescue anatomy reproduced from first principles in the decomposition.
- Outlier worth noting: dissimilar pair 6:7 at -19pp (t6 again — the residual-dependent fragile task).

**ANSWER TO THE WEEK'S QUESTION (all 76-task-derived pairs, 25 rollouts):**
**merge SHARED components with anyone (free, +1pp); merge RESIDUALS freely across dissimilar tasks and
benign twins (~0/+3pp); NEVER merge residuals of confusable twins (same scene, goal differs only by target
placement) — that single cell costs ~-19pp and is the entire source of twin-merge damage.**
Fig `results/figs/week3_2x2_full.png`. **EXP-18 fully closed.**

---

## EXP-19 · N=3 merge-safety factorial: are the pairwise rules compositional? 🟢 done

**Motivation (user, 2026-08-12).** The merge-safety matrix is pairwise-only (16 pairs, N=2). Two open
compositionality questions: (a) does "residuals of dissimilar tasks merge freely" survive N=3 (each task
now keeps only 1/3 of its residual + carries 2 foreign ones — Phase C hints it may not: t12 died in the
all-dissimilar triplet under elevated load); (b) does the confusable-twin penalty stay LOCALIZED to the
pair or spread to an innocent third member; (c) does all-confusable-family confusion compound.

**Design (user upsized from 1 to 3-4 triplets/type): 15 triplets, 122 evals @25, ~18h.**
Same factorial per member: sh = mean of 3 projections + own residual; re = own projection + mean of 3
residuals. Types (from `sweep/matrix3_triplets.json`):
- all-dissimilar x4: (12,13,14) [has avg+undiluted refs], (5,63,69), (21,7,66), (36,43,50)
- confusable pair + stranger x4: (24,69,+29), (26,34,+10), (49,75,+25), (68,74,+40)
- all-confusable family x3: caddy (17,26,34), (8,17,34); book-shelf (49,60,75)
- benign/rescue twins + stranger x4: (6,64,+32), (3,27,+37), (44,52,+64), (1,55,+46)
Driver `sweep/matrix3_driver.py` (build->eval->delete per triplet, resumable); report auto-written to
`results/analysis/matrix_n3_report.txt`. Caveat to carry: some tasks appear in multiple triplets
(cells not fully independent); conffam families overlap by construction (only 4 caddy tasks exist).
**RESULT (2026-08-16, 122/122 evals, 0 errors) — the pairwise rules are only PARTLY compositional.**
Report `results/analysis/matrix_n3_report.txt`, fig `results/figs/week3_n3_factorial.png`.

| type (Delta vs baseline) | shared merged | residual merged | N=2 reference (sh / re) |
|---|---|---|---|
| all-dissimilar (n=12) | -3.0 +-1.7 | **-12.5 +-6.0** | +1.3 / -1.0 |
| confusable pair + stranger (n=12) | +0.1 +-1.6 | -10.7 +-7.4 | +1.2 / -19.3 (pair) |
| all-confusable family (n=9) | +2.7 +-1.2 | **-38.1 +-13.4** | -- |
| benign/rescue + stranger (n=12) | +0.3 +-1.4 | **+1.5 +-2.1** | +0.0 / +3.0 |

Four findings:
1. **Shared merging stays free at N=3 in every type** (all |mean| <= 3pp) — fully compositional, as the
   ~0.99 mutual alignment of shared parts predicts.
2. **"Dissimilar residuals merge freely" BREAKS at N=3**: -1.0pp pairwise -> -12.5pp at triplets, driven by
   single sensitive tasks in perfectly-orthogonal groups (t21 100->36, t12 96->60, t14 80->56, t50 72->48).
   Confirms Phase C's mechanism at the factorial level: foreign-residual load grows with N and dominates.
3. **Confusable damage is LOCALIZED but COMPOUNDS**: strangers riding with a confusable pair are untouched
   (96-100% across all 4), while the pair's loser collapses harder than pairwise (t26 93->7). All-confusable
   families are catastrophic (-38.1 mean; t26 ->4%, t8 ->5%, t60 ->4%, t75 ->16%) with sharpened
   winner-take-all: exactly one family member survives (t34 / t49 at ~100%).
4. **The rescue is robust to N**: t6 64->88% (+24, its best result ever) with twin t64 AND a stranger
   aboard; benign-twin triplets all ~0/+.
**Revised merge-safety rules (N-aware):** shared merges freely at any N; residual merging degrades with N
even for orthogonal/dissimilar residuals (budget the group size); confusable families must never share
residuals (compounding winner-take-all), though adding dissimilar strangers to a confusable pair costs the
strangers nothing; benign/rescue twins remain safe and beneficial. **EXP-19 done.**

---

## EXP-20 · Week 24-08: improve a target LoRA by merging the pool INTO it 🔵 running

**Project goal (user, restated 2026-08-18).** Given a NEW LoRA and a pool, design a merge policy that
IMPROVES the new LoRA's own task. Unlike traditional model merging, merging here can *exceed* the
task-specific baseline (t19 60->92, t6 64->88, t50 +28pp observed). Pool-consistent setting for now
(homogeneous LIBERO-90 fleet; the EXP-15 external-adapter negative is parked, not violated).

**Strategic reframe.** Week-17's negative ("plain averaging is the optimum; a single policy cannot give
every task its full residual without full cross-talk") was about SYMMETRIC operators serving all members.
With ONE beneficiary the trade-off dissolves: give the target its residual at full strength and admit only
a whisper of the partners'. Note EXP-13's 0.85 recipe was already an approximate instance (and worked,
+8.7pp) — but it needlessly scaled the target's own residual to 0.85; the decomposition-native operator
does not.

**FLAGSHIP operator (running):** `W = S_16 + 1.0*r_target + alpha * mean(partner residuals)`.
S_16 = pool shared component (Phase B: interchangeable and FREE — alone it gave t6 +8pp). Arms: alpha=0
(= Phase-B interchange), twin partner at alpha in {0.25,0.5,1.0}, pool-mean at alpha in {0.25,0.5}.
Targets (weak = room to improve): t6 64%, t19 60%, t50 72%, t15 76%, t56 60%; partners chosen as the top
residual-cosine BENIGN twin (confusable pairs excluded per week-17 rules): t6->t64, t19->t49, t50->t21,
t15->t73, t56->t0. 30 policies, 30 evals @25. Driver `sweep/flagship_driver.py` (build->eval->delete).

**User's 6 questions — disposition after review:**
1. residual rank fragile vs robust -> running on CPU (`sweep/residual_rank.py`, all 76, exact rank<=32
   factorization of the residual; expect a 5th uniformity given norms/cosines/spectra/module-count/
   residual-fraction were all uniform).
2. corr(residual fraction, baseline) -> **ANSWERED: -0.264, p=0.019** (harder tasks carry LARGER
   residuals). Not a confound for the transfer correlation: partial corr controlling for baseline is
   -0.193 vs raw -0.193, because baseline itself does not predict transfer (+0.03). Independent effects.
3. k=8/32 -> **k=32 is a NO-OP** (the soup is exactly rank 16; directions 17-32 have ~zero singular
   values, P_32 == P_16). Reframed to a below-saturation sweep k in {4,8,12,16} = does boundary
   PLACEMENT matter behaviorally (are ranks 9-16 of "shared" secretly task-specific?).
4. capture-derived coefficients -> **capture is too uniform to be informative** (0.803 +-0.011, range
   0.77-0.83 => weights within +-3% of 1/N, i.e. ~plain averaging). Also O2 need-weighting used a far
   more variable signal (0-48pp) and lost badly (65 vs 83). Superseded by the flagship's asymmetric
   weighting, which is the version of "not uniform" that has room to win.
5. TIES/DARE on residuals only -> queued, ASYMMETRIC per user ("protect the target more"): target
   residual untouched, partners' residuals trimmed/dropped. Note TIES resolves sign conflicts and our
   residuals are near-orthogonal (mean cos -0.013), so DARE (drop+rescale) is the better fit.
6. per-rank / spectral chunking -> queued as a 2-parameter coefficient SCHEDULE over rank index (16 free
   coefficients is not searchable); to be informed by the k-sweep, which is its coarse version.
**FLAGSHIP-1 RESULT (2026-08-19, 30/30 @25) — the asymmetric operator works, but partner choice
INVERTS the naive expectation.** Report `results/analysis/flagship_report.txt`, fig
`results/figs/week4_flagship.png`.

| target | base | a=0 (shared swap) | twin .25 | twin .5 | twin 1.0 | rand5 .25 | rand5 .5 |
|---|---|---|---|---|---|---|---|
| t6 | 64 | 56 | **88** | 80 | 20 | 60 | 64 |
| t19 | 60 | **80** | 56 | 32 | 4 | **84** | 80 |
| t50 | 72 | 68 | 64 | 48 | 4 | 68 | 76 |
| t15 | 76 | 68 | 44 | 28 | 12 | 64 | 68 |
| t56 | 60 | 60 | 68 | **72** | 64 | 60 | 64 |
| **mean delta** | | **+0.0** | -2.4 | -14.4 | -45.6 | +0.8 | **+4.0** |

- **RANDOM partners beat the carefully-selected benign twin.** Best fixed policy = rand5 at alpha=0.5:
  **+4.0pp mean, worst case -8pp**. Twin arms degrade monotonically (-2.4 / -14.4 / -45.6).
- **Which partner helps is strongly task-specific**: t6 gets its best-ever result from its twin
  (64->88, +24), while t19 is HURT by its twin (-4/-28) and gains +24 from random partners — and gains
  +20 from the bare shared swap alone (alpha=0).
- Best-arm-per-target mean = +11.2pp, but that is a 6-way selection on 25-rollout cells -> winner's
  curse (the week-2 lesson); being re-checked.
- **CONFOUND FOUND IN OUR OWN DESIGN (flagship-2 launched):** injected residual energy is
  E = alpha/sqrt(k) for k near-orthogonal partners, so twin(k=1) at alpha injects E=alpha while
  rand5 injects 0.45*alpha — the rand5 arms were ~2x GENTLER, which alone could explain the win. At
  roughly matched energy (twin .25 -> E=.25 vs rand5 .5 -> E=.22) the gap narrows to -2.4 vs +4.0,
  so the conclusion survives but weakly. **FLAGSHIP-2** runs the properly matched grid
  (E in {0.22, 0.45} x k in {1,5,20}) + fresh re-runs of the two winning cells.
**FLAGSHIP-2 RESULT (2026-08-19, 22/22 @25) — energy-matched: partner COUNT is a real knob, and the
headline wins SURVIVE re-evaluation.** Report `results/analysis/flagship2_report.txt`, fig
`results/figs/week4_flagship2.png`.

| arm | E=alpha/sqrt(k) | t6 | t19 | t50 | t15 | t56 | mean delta |
|---|---|---|---|---|---|---|---|
| twin k=1 a=0.22 | 0.22 | 88 | 52 | 60 | 56 | 60 | -3.2 |
| **rand5 k=5 a=0.50** | 0.22 | 64 | 80 | 76 | 68 | 64 | **+4.0** |
| rand20 k=20 a=1.0 | 0.22 | 64 | 72 | 64 | 44 | 48 | -8.0 |
| twin k=1 a=0.50 | 0.45 | 80 | 32 | 48 | 28 | 72 | -14.4 |
| rand5 k=5 a=1.00 | 0.45 | 56 | 72 | 84 | 64 | 44 | -2.4 |
| rand20 k=20 a=2.0 | 0.45 | 52 | 48 | 60 | 28 | 48 | -19.2 |

- **The flagship-1 confound is RESOLVED and the conclusion holds**: at MATCHED injected energy, k=5 beats
  k=1 by 7.2pp (E=0.22) and 12.0pp (E=0.45). So it is not merely that rand5 was gentler — **partner COUNT
  is a genuine knob with an interior optimum at ~5**: one partner is too idiosyncratic, twenty wash out
  (k=20 is worst in BOTH energy bands, -8.0 / -19.2).
- **Dose still matters on top of count**: every k is worse at E=0.45 than at E=0.22 (-7 to -11pp),
  so the recipe is "a few partners, gently".
- **Winner's-curse re-checks PASS** (this is the first time in the project a headline survived cleanly):
  t6 twin a=0.25: 88% -> **92%** on fresh rollouts (baseline 64%, so **+28pp**); t19 rand5 a=0.25:
  84% -> **76%** (baseline 60%, **+16pp**). Both remain large; t19 shrank ~8pp as expected from selection.
- **Best deployable fixed policy so far: S_16 + 1.0*r_target + 0.5 * mean(5 random pool residuals)
  -> +4.0pp mean over 5 targets (worst case -8pp), no partner selection needed.** Per-target tuning
  reaches +28pp (t6) but needs a selector we do not yet have.
**Q3 k-sweep — PARTIAL (17/25 cells), rerun chained.** First pass ran CONCURRENTLY with the all-76 screen;
two 4-worker orchestrators over 2 GPUs caused 4 `server_died` + 4 build failures (disk/GPU contention).
OPERATIONAL RULE (new): only ONE orchestrator at a time — stage drivers behind each other, never overlap
(staging the CPU build phase is fine; the eval phase must be exclusive). Failed records purged, 17 valid
cells kept, rerun of the 8 missing cells chained behind all-76 (tmux ks2). Partial trend (do NOT report
yet): 'op' arm at k=4 looked best (+6.7 on 3 targets) vs +4.0 at k=16 — but the cells are non-overlapping
across k, so the means are not comparable until complete.

**All-76 deployment screen: RUNNING** (46/94 at 08:52, ~4h left) — applies
`S_16 + 1.0*r_target + 0.5*mean(5 random)` to every task; gives the weak/strong split and the count harmed.
**ALL-76 DEPLOYMENT SCREEN — PRELIMINARY (64/76 targets; 12 lost to the concurrency contention, retry
chained). THE HEADLINE DOES NOT SURVIVE, and the reason is the important finding.**

| | n | mean delta | lifted(>+5) | flat | hurt(<-5) |
|---|---|---|---|---|---|
| ALL | 64 | **-0.25pp** (SE 0.73) | 6 | 53 | 5 |
| WEAK (base<90%) | 13 | +0.46pp | 4 | 7 | 2 |
| STRONG (base>=90%) | 51 | -0.43pp | 2 | 46 | 3 |

Fleet-wide 94.3% -> 94.1%. Biggest gains t67 +12, t19/t27/t61/t63/t68 +8. Biggest losses **t15 -28,
t50 -20**, then -8s.

**THE KEY DIAGNOSTIC — the recipe is not a fixed policy, it is a LOTTERY.** The same recipe on the SAME
5 pilot targets, differing ONLY in which 5 random partners were drawn (seed 0 vs seed 1000+T):

| target | base | pilot draw | all-76 draw |
|---|---|---|---|
| t6 | 64 | 64 | 68 |
| t19 | 60 | **80** | 68 |
| t50 | 72 | **76** | 52 |
| t15 | 76 | 68 | **48** |
| t56 | 60 | 64 | 60 |
| mean | | **+4.0pp** | **-7.2pp** |

**An 11pp swing from the partner draw alone.** So the flagship's +4.0pp was a property of one lucky draw,
not of the recipe. Partner IDENTITY dominates; "5 random partners" has enormous variance
(sd 5.8pp across 64 targets, with -28pp tails). => **There is no selection-free recipe. A working
per-target partner SELECTOR is now the critical path**, and residual cosine is already known not to be it.
Consistent with the flagship-1 heterogeneity (t6 loves its twin, t19 hates it).

Operational: 12 targets failed (8 server_died + 4 build failures) from the SAME two-orchestrator contention;
records purged, retry chained behind the k-sweep rerun (tmux a76b), strictly solo.
**Q3 k-sweep COMPLETE (25/25 @25 rollouts, rerun solo).** Report `results/analysis/ksweep_report.txt`.

| arm | k=4 (cap .55) | k=8 (.67) | k=12 (.74) | k=16 (.80) |
|---|---|---|---|---|
| swap (S_k + own residual) | +1.6 | +0.0 | -2.4 | +0.0 |
| op (S_k + own + .5*rand5) | **+8.8** | -0.8 | — | +4.0 |

**Answer to Q3: the boundary placement k does NOT matter in any robust way.**
- swap arm is flat across k (+1.6 / 0.0 / -2.4 / 0.0, all within SE 3-7): taking MORE or LESS of the pool's
  shared component changes nothing — the interchangeability found in Phase B is not specific to k=16.
- op arm's k=4 (+8.8) looks best but is NOT a real effect: per-target SE is 3.2, and the paired k=4 minus
  k=16 contrast is **[+16, -12, 0, +24, -4] — mean +4.8 with SE 6.6, INCONSISTENT in sign across targets**.
  It is the same partner-draw lottery as the all-76 screen, not a property of k.
- Practical consequence: **k=16 (the soup's natural rank) stays the default**; the split point is not a
  tuning knob worth spending on. This also DEFLATES Q6 (per-rank schedules): if the coarse 4-way cut of the
  spectrum is behaviourally flat, a finer schedule over the same axis has little room. Q6 downgraded to
  low priority; the selector problem is the critical path.
**ALL-76 DEPLOYMENT SCREEN — COMPLETE (76/76 targets, 94 scene-evals @25).** Report
`results/analysis/all76_report.txt`, data `all76.json`, fig `results/figs/week4_all76.png`.

| | n | mean delta | lifted(>+5) | flat | hurt(<-5) |
|---|---|---|---|---|---|
| **ALL** | 76 | **-0.18pp** (SE 0.62, sd 5.4) | 6 | 65 | 5 |
| WEAK (base<90%) | 15 | +0.40pp | 4 | 9 | 2 |
| STRONG (base>=90%) | 61 | -0.33pp | 2 | 56 | 3 |

Fleet-wide 94.5% -> 94.3%; t = -0.30, indistinguishable from zero. 37/76 tasks unchanged exactly;
11/76 move by >5pp in EITHER direction. Gains t67 +12, t19/t27/t61/t63/t68 +8; losses t15 -28, t50 -20.

**VERDICT: the selection-free recipe does not survive at scale — and the reason is the week's real
finding.** The flagship's +4.0pp was measured on 5 weak targets with ONE partner draw. Rerunning the
IDENTICAL recipe on those same 5 targets with a DIFFERENT random draw gives **-7.2pp** (t19 80->68,
t50 76->52, t15 68->48): an **11pp swing from the partner draw alone**. So:
- there is **no selection-free recipe**: "5 random partners" is a lottery (sd 5.4pp fleet-wide, -28pp tail);
- **partner IDENTITY is the dominant factor**, above dose, count and boundary k (Q3 null, flagship-2 dose
  effects are second-order next to this);
- the confirmed large wins remain real and reachable (t6 64->92 = +28pp, t19 60->76 = +16pp, both re-checked
  on fresh rollouts) — they simply require choosing the RIGHT partners;
- **=> a per-target partner SELECTOR is the critical path.** Residual cosine is already excluded (t6 gains
  most from its twin, t19 is hurt by its). Next candidates: cheap behavioural probes (a few rollouts of
  target-with-partner), or a learned/heuristic score fit to the 230-instance transfer history.
**SELECTOR PROBE #1 (2026-08-21, CPU, free) — "donor quality" does NOT exist as a transferable property.**
Mined all 230 (target, partners, delta) instances from the uniform-average merge history.
- Naive donor score (mean delta of OTHER tasks when partner P is present) looks structured in-sample —
  best donors t21 +8.5, t20 +8.0, t55 +7.0; worst t71 -35.5, t72 -29.5, t74 -26.0 — but predicts held-out
  transfer at **pearson +0.087 (n=230)**, i.e. nothing.
- Cross-validated variance decomposition (LOOCV, ridge; the in-sample R^2 of 0.87 is illusory with 139
  parameters over 230 points):

| model | LOOCV R^2 | MAE |
|---|---|---|
| target identity only | **+0.411** | 9.6pp |
| donor identity only | **-0.075** | 13.7pp |
| target + donor | +0.379 | 10.0pp |
| predict global mean | 0.000 | 13.1pp |

**Conclusion: there is no such thing as a globally "good donor".** All the predictable variance sits in the
TARGET (how much a task suffers when merged at all — the fragility-history result from week 2, r=+0.71,
here +0.41 in a harder pooled setting), and adding donor identity makes prediction WORSE (overfitting).
This is consistent with the all-76 lottery and with flagship-1's heterogeneity (t6 loves its twin, t19 is
hurt by its): usefulness is a **(target, donor) INTERACTION**, not a donor property.
=> A selector cannot be a ranking of pool members. It must score PAIRS, which the current data cannot fit
(only 84 of the ~5700 possible (target,donor) pairs have >=2 observations). Remaining routes: (a) a
similarity-conditioned interaction model (needs a pair feature that works — residual cosine is excluded),
(b) activation-space pair distance (EXP-11 infra, no simulator), or (c) accept a cheap behavioural probe
per candidate partner (a few rollouts) — the honest fallback the evidence currently supports.
**Q5 RESULT (2026-08-21, 15/15 @25) — spectral truncation of partner residuals helps, weakly but
consistently.** Report `results/analysis/dare_report.txt`, fig `results/figs/week4_q5_truncation.png`.

| arm | t6 | t19 | t50 | t15 | t56 | mean | sd |
|---|---|---|---|---|---|---|---|
| dense (ref) | +4 | +8 | -20 | -28 | +0 | -7.2 | 15.8 |
| top-2 | +0 | +16 | -4 | -28 | -16 | -6.4 | 16.6 |
| top-4 | +8 | +4 | -8 | -24 | +8 | **-2.4** | 13.7 |
| top-8 | -8 | +20 | -8 | -12 | +0 | **-1.6** | **12.8** |

- **Paired against the dense reference (same target, same partners, only the partner residual truncated):
  top-4 +4.8pp (SE 2.7, 4/5 targets improve), top-8 +5.6pp (SE 5.2, 3/4).** Directionally consistent and
  the top-4 contrast is ~1.8 SE — suggestive, not significant at n=5.
- **The effect is on the TAIL, which is what we wanted**: the catastrophic cases shrink (t50 -20 -> -8,
  t15 -28 -> -12) and the spread falls monotonically from top-2 to top-8 (sd 16.6 -> 13.7 -> 12.8).
  So a partner's DAMAGE is concentrated in its idiosyncratic low-energy directions, while its useful
  content is in its dominant ones — the first structural handle we have on the lottery variance.
- Still net-negative overall (-1.6pp best arm), so this does NOT rescue the selection-free recipe; it
  reduces the downside rather than creating upside.
- Q5 answered. Note the arms are energy-matched, so this is a STRUCTURE effect, not a dose effect.
**SELECTOR PROBE #2 (2026-08-21, CPU, free) — activation-space pair distance also FAILS.**
Reused the 76 EXP-11 solo signatures (180-obs bank, 9 depths). Feature = mean activation cos-distance
between target and donor, measured on the TARGET's own scenes (deployment-relevant) and on the full bank.
- Pooled over 230 instances: best layer exp_17 at **r = -0.138** (all others |r| <= 0.16); act_out full-bank
  -0.157. Same weak late-stack-only signal EXP-11 found for merge outcomes — nothing usable.
- **Decisive test — WITHIN-TARGET ranking** (does the score order donors FOR a given target? that IS the
  selector's job): **mean r = -0.043 over 32 targets**, with per-target values scattered uniformly across
  [-1.0, +1.0] (e.g. +1.0, +0.98, +0.95 ... -0.98, -0.93, -0.89). That spread is what pure noise looks like
  at n=3-5 observations per target, not a weak signal.
**=> Both free selector routes are exhausted** (weight-space residual cosine: excluded earlier; donor-quality
history: R^2 -0.075; activation-space pair distance: r -0.04 within-target). Every static feature we can
compute fails to rank partners for a target. This is the same wall EXP-11 hit at the operator level and is
consistent with the project's recurring finding: merge outcomes are a closed-loop/dynamical property, not a
static-geometry one. **The honest remaining option is a cheap BEHAVIOURAL probe** (a few rollouts of
target+candidate), which converts the method from "free" to "cheap" but is the only route the evidence
supports.
**BEHAVIOURAL PROBE SELECTOR (2026-08-21, 40 probes @3 + 10 commits @25) — IT WORKS. The method closes.**
Report `results/analysis/probe_selector_report.txt`, fig `results/figs/week4_probe_selector.png`.

| target | base | rand5 | TOP-3 (probed) | BOTTOM-3 (control) | top-bot |
|---|---|---|---|---|---|
| t6 | 64 | 68 | 72 | 44 | **+28** |
| t19 | 60 | 68 | 76 | 72 | +4 |
| t50 | 72 | 52 | 72 | 72 | 0 |
| t15 | 76 | 48 | 68 | 36 | **+32** |
| t56 | 60 | 60 | 72 | 40 | **+32** |
| **mean delta** | | **-7.2** | **+5.6** | **-13.6** | **+19.2** |

- **A 3-rollout probe carries real selection signal**: paired TOP3-BOTTOM3 = **+19.2pp, SE 7.1, t=2.71,
  positive in 4/5 targets** (the 5th is exactly 0). The anti-selection control is what makes this
  interpretable — the ranking itself is causal, not the operator.
- **TOP3 also beats the random-partner recipe by +12.8pp (SE 3.2, 5/5 targets)** and turns the all-76
  screen's net-zero into **+5.6pp** on these targets.
- **Cost: 74 rollouts per target** (8 candidates x 3 + 2 commits x 25) — versus 200 for exhaustive
  probing. In deployment only the top-3 commit is needed: 8x3 + 25 = **49 rollouts**, no training, no data.
- **Caveat that makes this a LOWER BOUND**: at 3 rollouts the probe score can only take 4 values
  (0/.33/.67/1.0), so the top-3 cut was TIE-BROKEN by sort order in all 5 targets (2-6 candidates tied at
  the cutoff). The selector we tested is therefore partly random; a sharper probe (5-7 rollouts, or
  scoring on the target's hardest scene) should do better. +19.2pp is what a partly-random selector buys.
**=> The week's arc closes: asymmetric operator (right shape) + behavioural probe (the only selector the
evidence supports) = a train-free method that improves a target LoRA using a pool, at the price of ~50
rollouts.** Static features cannot substitute (3 families tested, all null) — selection is irreducibly
behavioural, which is itself the paper-worthy claim.

### EXP-20.11 — Probe selector v2 (10-rollout probes + robust-task safety) — **done**
**Motivation.** v1's +19.2pp selection effect was a lower bound: at 3 rollouts the probe score takes
only 4 values and the top-3 cut was tie-broken by sort order in all 5 targets. Also unanswered:
is probe-selected merging *safe* on targets with no headroom?

**Setup.** `sweep/probe10_driver.py` (e26). 10 weak+robust targets: WEAK [6,19,50,15,56],
ROBUST [0,1,2,4,13] (all 100% baseline, different families, single-scene). 8 candidates × 10-rollout
probes → top-3 and bottom-3 committed @25. Same seed (rng 2000+T) and operator as v1.

**Result — both questions answered positively.**
- WEAK: TOP3 **+14.4pp** (SE 4.3, t=3.34, p=0.029, 5/5 positive), up from +5.6pp at 3 rollouts.
- Selection effect **+29.6pp** (SE 2.7, t=10.9, p=0.0004), up from +19.2pp. **Probe resolution, not
  the operator, was the binding constraint** — the tie-break diagnosis is confirmed.
- ROBUST: all five end at **100%** under top-3 AND bottom-3. Merging cannot hurt them at this dose.
- **Safety is self-enforcing:** probe-score spread across 8 candidates is 0.5–0.8 on weak targets but
  0.0–0.1 on robust ones (6–8 of 8 tied at the top). Where selection is arbitrary, it is also
  harmless — no separate headroom gate is needed.
- Figure `results/analysis/fig_probe_selector_v2.png`; report `results/analysis/probe_selector10_report.txt`.

### EXP-20.12 — All-76 probe-selector deployment screen — **running** (launched 2026-08-22 11:43)
**Motivation.** User-approved gate cleared (option 2, deployment-faithful): the sharper probe beat
its own lower bound and robust targets were unharmed, so the method goes to the full fleet under the
reporting rule (all 76 LoRAs in every reported result).

**Setup.** `sweep/probe76_driver.py` (e27), tmux `probe76`, watcher `brf85f6k7`.
All 76 targets × 8 candidates @10-rollout probes → top-3 committed @25; bottom-3 control on the
`T % 5 == 0` subset (16 targets) for cost. ~8,400 rollouts.
**Reuse:** e26 shares the seed, operator and rollout counts, so its 10 targets were ported to e27
job names before launch (100 result files) — ~13% of the run pre-completed.

**Reads.** (a) fleet-wide Δ and the weak/strong split, versus the −0.18pp selection-free screen
(EXP-20.9); (b) does the +14.4pp weak-target lift survive on all 15 weak targets; (c) does the
robust half stay flat across 61 strong targets, not just 5.

**Cost decision (2026-08-22, user):** measured throughput is **74 min per fresh target** (4 workers,
both GPUs saturated, no bottleneck) → **~78 h / 3.2 days**, finishing ~Tue 25 Aug. Alternatives offered
(cut candidate pool 8→5; drop the 1-in-5 bottom-3 control) were **declined — run as designed**.
Implication: the GPUs are committed until Tue 25 Aug; no other orchestrator may run (one-at-a-time rule),
so Q-follow-ups queue behind this.

**RESULT (landed 2026-08-25 22:4x, 3.5 days, 0 errors).** The method holds at fleet scale.
- **ALL 76: +2.22pp** (SE 0.65, t=3.40, **p=0.001**). Fleet-wide success **94.5% → 96.7%**.
- **WEAK n=15: +9.78pp** (SE 1.77, t=5.51, p=7.7e-05); **9 lifted >5pp, 0 harmed**.
- **STRONG n=61: +0.36pp** (n.s.); 3 harmed >5pp (t21 −12, t69 −8, t26 −5).
- **Paired vs the selection-free screen** (same targets, same operator, only partner choice differs):
  selection adds **+2.40pp overall** (t=3.20, p=0.002) and **+9.38pp on weak** (t=3.42, p=0.004).
- **Generalization check:** on the 10 weak targets never used in any pilot, **+7.47pp, 10/10 positive**
  (pilot claimed +14.4pp on 5 hand-picked fragile targets — the effect shrinks but stays unanimous).
- **Safety self-enforcing at n=76:** corr(headroom, probe spread) **r=+0.73, p=1.2e-13**; base-100%
  targets have spread 0.085 vs 0.458 for base<90%. Anti-selection control: TOP3−BOT3 = **+24.3pp on
  weak (n=7)** vs **+1.0pp on strong (n=16)**.
- Figure `results/analysis/fig_all76_probe_selector.png`; report `results/analysis/probe_selector76_report.txt`.

**Known cost, and the guard that fixes it (settled OFFLINE, no GPU).** The 3 harmed strong targets had
near-flat probes. I proposed gating on probe *spread*; re-scoring the existing e27 data **refutes that**
(spread<0.15 → ALL +1.91, WEAK +8.58, still 3 harmed — it costs weak-target gain and doesn't reduce harm).
The rule that works is **abstention on the probe's own verdict — deploy the merge only if the best probe
score exceeds the target's baseline**:

| rule | ALL | WEAK | STRONG | harmed>5 | abstains |
|---|---|---|---|---|---|
| merge always (as run) | +2.22 | +9.78 | +0.36 | 3 | 0 |
| **best probe > baseline** | **+2.77** | **+9.78** | **+1.05** | **1** | 37 |
| spread < 0.15 (refuted) | +1.91 | +8.58 | +0.27 | 3 | 42 |

Strictly dominates the ungated method, costs zero extra rollouts, and the decision signal (10-rollout
probes) is independent of the measured outcome (25-rollout commits). **This is now the recommended
deployment rule.** Caveat: it is an offline re-scoring of one run, not a fresh confirmation.

### EXP-21 — Guard confirmation + externally-trained adapters (planned)
**Motivation.** Two open threads. (a) The abstention guard was validated by re-scoring, not by a fresh
run — a confirmation would remove the last in-sample worry, though it costs a full fleet sweep. (b) The
whole result lives in the pool-consistent (homogeneous fleet) setting the user approved for now;
externally-trained adapters share no backbone (cos ~0.13), where S_16 has no meaning. Extending there is
the real generality question.

## EXP-22 — Week of 31-08: dosing, symmetry, and replacing the probe with a network
**Motivation.** Last week's method works fleet-wide (+2.22pp all-76, +9.78pp weak) but costs 80 rollouts
per scene and has only ever been run at ONE dose. Two attacks: (a) is the operator's dosing right?
(b) can the behavioural probe be distilled into a network so selection becomes free?

### EXP-22.1 — E1+E2 dosing: W = S_16 + beta*r_target + alpha*r_donor — **running** (launched 2026-08-26 12:34)
`sweep/dosing_driver.py` (e28), tmux `dosing`, watcher `b3lgg00pr`. 8 targets (5 weak: t6,19,56,15,50;
3 strong: t21,69,25 — t21/t69 are the two the ungated method HARMED), single top-1 probe donor from e27,
10 rollouts. Arms chosen so contrasts are PAIRED, not confounded:
- E1 donor dose: (beta=1, alpha=0.25 / **0.50** / 1.00) — the 0.50 cell is exactly the e27 probe operator
  at the same 10 rollouts, so it is **reused free** (8 of 40 cells).
- E2 interpolation: (0.75, 0.25) and (0.50, 0.50).
- Contrasts: (0.75,0.25) vs (1.00,0.25) = price of un-protecting the target, donor dose held fixed;
  (0.50,0.50) vs (1.00,1.00) = same ratio, half the scale = magnitude vs balance.
Build math validated against a dense reference before launch (rel. err 3e-7).
**Reads.** Is 0.5 the right donor dose? Does protecting the target at 1.0 actually matter, or is it a
choice we inherited? Does a smaller dose remove the harm on t21/t69?

### EXP-22.2 — E3 symmetry: is B a good donor for A when A is good for B? — **partially answered OFFLINE (free)**
The 604 e27 probes already contain **25 reciprocal pairs**. Affinity(T,p) = probe score − baseline(T).
- Raw: pearson **r = −0.524** (p = 0.007) — looks strongly ANTI-symmetric.
- **This is a ceiling artifact.** A 100%-baseline task can only lose; a 64% task can only gain. On the
  17 pairs whose baselines are within 10pp, **r = −0.005** — no reciprocity in either direction.
  (A linear headroom control does NOT remove it — the truncation is a bound, not a slope — which is why
  the matched-baseline restriction is the right test.)
- **Provisional answer: affinity is NOT symmetric, and not anti-symmetric — it is simply not reciprocal.**
  Sign agreement 11/25 = 44% (chance ~50%).
- ⇒ Per the decision rule, this routes to **a learned network**, not a latent-space cosine. A cosine is
  symmetric by construction, so it *cannot* represent this relation regardless of the space it lives in.
- **Confirmation still needed** (n=17 is thin): ~30 designed matched-headroom pairs probed in BOTH
  directions @10 rollouts (~600 rollouts). Queued behind E1+E2.

### EXP-22.3 — E4 affinity hypernetwork (T2L-style) — **design settled, not yet built**
Given a target LoRA + k candidates, emit k affinity scores; merge top-n. Hold out 10 tasks (5 weak, 5 strong).
**Feasibility correction (important).** Training with rollouts *in the loop* is not affordable: all 5,700
ordered pairs at 10 rollouts = 57,000 rollouts ~= 28 days of exclusive GPU. Instead the behavioural loss is
**amortised**: we already own **604 measured (target, donor) affinity labels** from e27 (plus 230 older
merge instances), collected at 10 rollouts each. That IS behavioural supervision — just precomputed. If
more labels are needed, a second probe sweep adds 608 labels for ~3 days.
**Difference from T2L, stated honestly.** T2L maps a *task description* to LoRA *weights*, and never takes
LoRA weights as input. Ours maps (target LoRA, candidate LoRA) to a *scalar affinity* — a pair scorer, not
a weight generator. What transfers is the shape: a shared trunk with per-module embeddings, and T2L's
finding that task-performance supervision beats weight reconstruction. We also have per-task *language*,
so the encoder can take both weight-space and text features — the text half is the part closest to T2L.

**EXP-22.3 RESULT — the hypernetwork runs, fits training targets, and does not transfer.**
Pipeline: `sweep/hypernet_features.py` (76x67: W residual-Gram / A activation sig / L language / S scalars)
+ `sweep/hypernet.py` (asymmetric pair scorer, shared trunk, within-target listwise loss, holdout = 10
unseen TARGETS, baselines random/oracle/ridge every run).
- **Two bugs found and fixed.** (i) *Label leak*: the probe report stores candidates sorted by affinity,
  so list position IS the label — a constant-score model "recovered the oracle" via argsort ties (first
  run reported hypernet == oracle in 3/3 seeds). Now shuffled deterministically. (ii) *Score collapse*
  (sd = 0.000): 61 of 76 targets are saturated with near-flat candidate scores, so ~80% of groups carry
  a near-uniform soft target and vote for constant output. Fixed by weighting each group by aff.std().
- **Fit vs transfer:** train spearman **+0.83** (train top-3 +2.28 vs train oracle +2.46) but holdout
  spearman **+0.02**. It memorises the training targets and transfers nothing.
- **Ablation (4 seeds) was MISLEADING and is retracted.** It suggested W alone recovered ~25% of the
  oracle gap. A 12-seed **paired** re-run (same holdouts for model and baseline) gives W **+0.01pp,
  p=0.99, 6/12 seeds positive, 0% of gap**; WALS +0.84pp, p=0.20. Lesson: 4 unpaired seeds on a
  10-target holdout is not enough to separate anything here.
- **Learning curve settles WHY (CPU-only, no GPU spent).** Gain over random vs number of training
  targets: 15 -> +0.11, 25 -> +0.31, 40 -> +0.45, 55 -> +0.48, 66 -> +0.74. The **trend is real**
  (r = +0.96, p = 0.010) — so the signal is not absent — but the slope is **0.0106 pp per target**.
  Recovering 25% of the 8.63pp oracle gap needs **~204 training targets**; 50% needs **~407**.
  **Our fleet has 76 tasks -> 66 training targets. That is the ceiling, and it is 3-6x short.**
- **Decision: do NOT spend the 3 GPU-days on a second probe sweep.** That sweep would add more
  *candidates per target* (8 -> 16), but the learning curve shows the binding axis is the number of
  *targets*, which is fixed by fleet size. It would not address the bottleneck.
- Caveat stated plainly: extrapolating a 15-66 curve out to 204-407 is an order-of-magnitude estimate,
  not a precise requirement, and curves like this usually *saturate* rather than stay linear — which
  would make the true requirement larger, not smaller.
- Figure `results/analysis/fig_hypernet_learning_curve.png`; reports `hypernet_confirm.txt`,
  `hypernet_ablation.txt`, `hypernet_learning_curve.txt`.

**Where this leaves the project's central claim.** Four families of selector have now failed on held-out
targets: weight-space cosine, donor-quality history, activation distance, and now a *learned* nonlinear
asymmetric model over all three feature views at once. The first three were correlational failures; this
one is a capacity/sample-size failure with a measured bound. Together they make
**"partner selection is irreducibly behavioural"** a much stronger statement than it was last week.
The behavioural probe (80 rollouts/scene) remains the method.

**EXP-22.2 RESULT — E3 symmetry CONFIRMED by design (n=30 pairs, both directions, 10 rollouts).**
30 pairs among the 35 tasks with headroom (baseline <= 0.96, so affinity is measurable both ways).
- pearson **r = -0.115** (p = 0.54); spearman -0.175 (p = 0.35); **sign agreement 15/30 = 50% = chance**.
- **mean |A->B minus B->A| = 17.7pp**, extremes 60pp (t54->t15 -56 vs t15->t54 +4; t20->t50 -62 vs
  t50->t20 -5). The fleet-wide effect of the whole method is 2.2pp, so the asymmetry dwarfs it.
- Reproduces the free offline estimate (r = -0.005 on 17 matched pairs). **Affinity is not reciprocal.**
- ⇒ The task-list branch "if there IS symmetry, use cosine in latent space to compute alphas" is closed:
  a cosine is symmetric by construction, so no embedding can represent this relation. Confirms the
  asymmetric pair encoder (signed difference) used in the hypernetwork.
- Figure `results/analysis/fig_symmetry.png`; report `results/analysis/symmetry_report.txt`.

**Operational gotcha discovered (will recur, now fixed in drivers).** The orchestrator's build cache is
**tmpfs** (`/dev/shm/mergesweep/cache`, 32G) and each merged PI0.5 policy is **6.8G**. Four workers hold
four distinct policies = 27G resident; the next build writes a `.build` temp dir first, needing 34G ->
guaranteed `ENOSPC` ("No space left on device", sometimes surfacing as "No such file or directory").
**Fix: n_workers=3** (20.4G resident + 6.8G building = 27.2G). Costs ~25% throughput. Zero failures since.
Note this may explain some previously-unexplained `server_died`/build failures blamed on concurrent runs.

**EXP-22.1 RESULT — E1+E2 dosing complete (8 targets, 10 rollouts, all contrasts paired).**
- **Q1: alpha = 0.5 is a REAL interior optimum.** Paired, all-fresh cells: 0.50 beats 0.25 by **+8.8pp**
  and beats 1.00 by **+8.8pp** (ALL: +3.5 / +12.2 / +3.5; WEAK: +9.6 / +19.6 / +13.6). Too little donor
  transmits nothing, too much overwrites the target. The dose used all project was right by luck; now verified.
- **WINNER'S CURSE MEASURED: +5.0pp.** The e27 probe cell (a max-of-8 at alpha=0.5) reads +17.2 where the
  independent re-measure gives +12.2. **Reusable calibration: any probe-selected maximum overstates by
  ~5pp at 10 rollouts.** Last week's headline numbers should be read with this in mind. (Caught before
  reporting: the +27.6pp weak-target spike at exactly the selection dose was the tell.)
- **Q2: keep the target at 1.0.** Un-protecting it costs -2.5pp (0.75 vs 1.00 at fixed donor);
  halving the whole operator at fixed RATIO costs **-8.8pp** — magnitude matters, not balance. The
  0.50/0.50 arm is the worst in the grid (ALL -5.2, STRONG -13.3). Interpolation is a step back toward
  symmetric merging and behaves like one. Target-specific though: at 0.75/0.25 t19 +40 vs t15 -26.
- Figure `results/analysis/fig_dosing.png`; report `results/analysis/dosing_report.txt`.

### EXP-22.4 — Probe reliability (test-retest) — **done 2026-08-27**
**Reframed from the intended weight-sensitivity test:** the dnd_libero90_dataset "checkpoints" (steps
9500-9990) are functionally identical (cos 1.0000, max elementwise diff 1.4e-5), so they cannot answer
task-level vs weight-level. They DO give four independent measurements of the same adapter = test-retest.
- **Solo success of the SAME adapter, 4 repeats:** t6 70/80/100/50, t19 80/70/70/70, t50 70/40/70/60.
  **Mean SD 13.3pp** on a quantity we treat as ground truth (binomial expectation at 10 rollouts ~14pp).
- **Donor-ranking reliability rho = +0.409** (t6 +0.627, t19 +0.532, **t50 +0.066**).
- **Ceiling on any predictor = sqrt(rho) = 0.64.** The hypernetwork was chasing a target that was never 1.0.
- **Reported results are NOT affected:** baselines and commits are 25-rollout measurements independent of
  the 10-rollout probes (verified in the jsonl: baseline n_trials 25-50, commit n_trials 25, probe 10).
  So +2.22pp / +9.78pp are honest measurements of deployed policies.
- **But selection was half-blind, so those are LOWER bounds.** Empirically consistent: 3 -> 10 rollout
  probes took weak gain +5.6 -> +14.4pp. Spearman-Brown: 10 -> 25 rollouts lifts reliability 0.41 -> 0.63.
- ⇒ **Highest-value next step for the METHOD: a less noisy probe, not a smarter one.**
- Figure `results/analysis/fig_probe_reliability.png`; report `results/analysis/ckpt_stability_report.txt`.

### EXP-22.5 — Phase 1: action-error proxy labels (T2L SFT branch) — **FAILS**
Borrowed T2L's SFT idea: score a merged adapter by applying it and taking the model's error on task data,
instead of rolling it out. Label = mean L1 between predicted and demonstrated action over 24 demo
timesteps from the target's own scenes (3 demos x 8 steps), operator base + D_target + 0.5*D_donor.
- **Pipeline validated first:** the task's own LoRA cuts action error 0.2797 -> 0.0439 (**-84%**) vs the
  un-adapted base, and restore is bit-exact. So preprocessing (image 180-rotation, 8-dim state, prompt)
  is correct -- a wrong pipeline would have shown no adapter advantage.
- **Result: no relationship.** Within-target spearman vs measured rollout affinity =
  **-0.082 (SE 0.138, p = 0.57)**, 5/12 targets point the WRONG way, per-target values scatter over
  [-0.58, +0.88] -- the signature of noise at n=8 per target. Pooled pearson r = -0.010 (n=96).
- **Label noise does not explain it.** With label reliability rho = 0.41 a perfect proxy would still
  reach ~sqrt(0.41) = 0.64. We measured -0.08. Attenuation-corrected it is still ~ -0.13.
- **Why this failure is the most informative one yet.** Action error is DETERMINISTIC (zero measurement
  noise) and TASK-GROUNDED -- the two properties I argued would distinguish it from activation distance.
  Neither helped. A merged policy can predict near-identical actions on demonstration states and still
  fail at rollout, because failure comes from compounding drift into states the demos never visit.
- ⇒ **Fifth selector family to fail**, and the cheap-label route to breaking the 66-target ceiling is
  closed. Figure `results/analysis/fig_action_error_proxy.png`.

### EXP-22.6 — Phase 2: 25-rollout probes — **done 2026-08-29. NULL: 10 rollouts is enough.**
10 weakest targets, same 8 candidates, same operator, same 25-rollout commit budget; only probe quality
differs (where selections agree, e27's commit serves both arms — 3/10 agreed).
- **Gain from the sharper probe: −1.10pp (SE 1.42, p = 0.46).** Deltas −4,+4,+8,−4,−7,0,0,0,−4,−4.
- 7/10 selections CHANGED but outcomes are equivalent → the ρ=0.41 noise shuffles among near-equivalent
  good donors; it rarely swaps good for bad. Top-3 overlap is higher on easier targets (r=+0.38).
- Reconciles with 3→10 gain: 3 rollouts = 4-level scores = ranking by ties (too coarse); 10 resolves
  good-vs-bad; 25 only re-orders within-noise donors.
- **Method verdict: the 10-rollout probe is at the practical optimum.** Bracketed from both sides now —
  cheaper selection fails (5 families), sharper selection buys nothing.
- Fixed during run: driver batching barrier idled GPU2 (groups of 3 with waits → single-plan submission;
  both GPUs 100% after). Figure `results/analysis/fig_probe25.png`; report `probe25_report.txt`.

### EXP-23 — Hypernet v2: binary objective + consolidated labels — **FIRST SIGNIFICANT TRANSFER**
**Diagnosis addressed.** Signal was limited three ways: (1) 66 targets vs ~204 needed; (2) label
reliability 0.41 caps any predictor at 0.64; (3) the LISTWISE objective demanded an exact ranking that
is half noise and — per EXP-22.6 — unnecessary (each target has several near-equivalent good donors;
deployment needs ANY good one + the abstention guard).
**Changes (CPU-only, free):** consolidated ALL owned pairwise measurements into 634 labels
(`consolidate_labels.py`: e27 + e29 + e30 + e31, repeats rollout-weight-averaged — 80 pairs now at
effective reliability ~0.6); switched the loss to weighted BCE on "donor good (aff>0)".
**Result (12 seeds, paired vs random on identical holdouts):**
- binary:   **+1.31pp over random (SE 0.39, t=+3.35, p=0.0065), 17% of oracle gap**;
            good-donor hit rate 0.66 vs 0.62 random (p=0.073)
- listwise, same data: +0.61pp (p=0.29) — so the OBJECTIVE is the driver, not the extra labels.
First positive holdout transfer in the hypernetwork line. Still modest vs the probe; the open question
is the learning-curve slope under this objective (running: hypernet_lc2.txt) — it decides whether
synthetic targets are worth GPU.

**EXP-23.2 RESULT — NO-GO at the 25-synthetic checkpoint; run aborted, ~3 GPU-days saved.**
Mixed real+synthetic training, holdout always real, 10 seeds paired:
+0 syn: +1.16pp (p=0.001) | +12: +0.75 | +25: +0.74 → adding 25 synthetics = **−0.42pp (p=0.36)**
vs **+0.45pp predicted** by the real-target slope. Perturbed adapters (r_t + 0.5*r_p) are NOT new task
diversity — the learner already knows the 76 tasks' geometry; recombinations add rows, not information.
The 27 banked synthetics' labels are kept (results/merge/e32_*). Side finding: the +0 baseline here
used WLS features (no activation block) and scored +1.16pp (p=0.001) — dropping the dead A-block helps.
**Every signal axis is now measured: targets (real ones only), candidates (flat), labels (consolidated),
objective (binary = the fix), synthetics (refuted).** The v2 network (~+1pp label-space transfer) is
what 76 tasks support.

**EXP-23.3 RESULT — deployment verdict: the free selector is NOT competitive.**
15 weak targets, LOO ensemble ranking all 75 candidates, top-3 committed @25 (same operator/budget as e27):
- vs baseline: **+2.80pp (SE 2.96, p = 0.36)** — positive point estimate, not significant.
- vs random-5: +2.40pp (p = 0.34).  - vs probe: **−7.00pp (p = 0.030)** — significantly worse.
- **The split explains the mechanism.** Weakest 6 targets (base ≤76%): **−1.33pp**; mid-tier 9 (80–90%):
  **+5.56pp**. Picks are dominated by two "universal donors" (t42 in 7/15, t32 in 7/15): what the network
  learned is a **donor PRIOR in good/bad space** — enough for safe small gains where headroom is small,
  useless-to-harmful on fragile targets, which need the (target,donor) INTERACTION.
- **This completes the picture across all levels:** label space +1pp (real, p<0.01) = the learnable donor
  prior; deployment on fragile targets ≈ 0 = the interaction stays behavioural-only at 76 tasks. The old
  donor-history regression missed the prior because it predicted MAGNITUDE (R²=−0.075); in binary
  good/bad space the prior exists. The probe remains the method.
**HYPERNETWORK LINE CLOSED (2026-08-30), fully quantified at every level:** objective fixed (binary),
labels consolidated, candidates axis flat, synthetic targets refuted, deployment tested. What 76 tasks
support is a donor prior worth ~+2-3pp on mid-tier targets at zero cost — not a replacement for the probe.

### EXP-24 — Does probe-selected merging COMPOSE? — **NO, and the guard is why**
Round-1 winners as new targets (r' = r_T + (1/6)*sum r_top3), 8 fresh random candidates probed @10,
commit top-3 @25 only if the best round-2 probe beat the round-1 committed score.
- **4/4 targets passed the guard; 0/4 actually improved.** round2 - round1 = **-9.00pp** (SE 6.61, p=0.27);
  t19 84->84, t56 80->80, t6 84->76, **t50 76->48**. (t15 abstained — no probe beat its round-1 score.)
- **The mechanism is a selection-for-noise trap.** Probe said 90/90/100/90; commits gave 84/80/76/48 —
  **mean overstatement 20.5pp, 4.1x the 5.0pp winner's curse measured in round 1.**
  Reason: when the incumbent is ALREADY GOOD (76-84%), the only way a probe beats it is an upward noise
  draw, so the guard "commit if best probe > incumbent" **selects precisely for measurement error**. The
  higher the incumbent, the more biased the guard becomes.
- **Bears directly on the deployment guard** (EXP-22 recommended "merge only if best probe > baseline"):
  it is safe against a WEAK baseline (real headroom, modest bias) but degenerates as the incumbent rises.
  A second round would need an unbiased accept test — e.g. re-measure the candidate at commit budget
  before accepting, which costs exactly what it saves.
- ⇒ **The method is single-shot.** One round extracts what the pool offers for a target; there is no free
  hill-climbing loop. Report `results/analysis/iterate_report.txt`.

### EXP-25 — 5-rollout probe: can selection cost halve? — **NO. 10 rollouts is the knee.**
Same 10 weak targets / 8 candidates as Phase 2, probed @5 (40 r/scene vs 80).
- **-3.70pp vs the 10-rollout selection** (SE 3.81, p = 0.357); selection agreement **1/10** (vs 3/10 for
  the 25-rollout arm). t56 collapses 80 -> 48; excluding it the mean is -0.56pp (n=9), so the loss is
  driven by occasional catastrophic mis-picks rather than uniform degradation — exactly the failure mode
  a coarse score (6 levels at 5 rollouts) predicts.
- **Probe-budget curve now complete** (weak-target gain): 3 -> +5.6 | **5 -> +6.8** | **10 -> +10.5** |
  25 -> +9.4. A clean interior optimum at 10 rollouts / 80 per scene, bracketed by measurement on both
  sides. Figure `results/analysis/fig_probe_budget.png`; report `results/analysis/probe5_report.txt`.

### EXP-30 — Hypernetwork v1: LoRA2Coef trained by REINFORCE on real rollouts — **DROPPED (donor-independent)**
Motivation: reverse the DnD hypernetwork — two LoRAs in (target, donor), merging coefficient out — and
train it on the real objective (rollout success of the merged policy) instead of proxy labels.
- Setup: raw B/A factors of all 458 modules, JL-projected to (128 modules x 32 x 64) tokens; HyperConv
  contracting encoder -> 128-d embeddings; asymmetric head [e_T, e_p, e_T-e_p, e_T*e_p] -> c in [0.10, 1.20],
  warm start exactly 0.5. REINFORCE: reward = success(merged) - success(target), antithetic pairs, per-target
  baseline, advantage standardisation. 66 train / 10 held-out targets (5 weak + 5 strong). 640 rollouts/step.
- Iterations: (1) random donors -> constant 0.385 (reward landscape flat) -> stratified donors + spread
  diagnostic. (2) c -> 0.000 saturation: c=0 scores exactly 0 with zero variance, a genuine optimum for a
  non-discriminating learner -> bounded sigmoid output, lr 3e-3 -> 5e-4, prior to 0.5. (3) coefficient
  drifted 0.50 -> 0.33 over 7 steps while "spread" grew 0.0002 -> 0.027.
- **Decisive diagnostic (step 7, ~4,500 rollouts): coefficient for a held-out target's known-GOOD donors
  0.3145, known-BAD donors 0.3145 — identical to 4 decimals.** The spread was variation across TARGETS,
  not donors. Same "do not merge" attractor as iterations 1-2, bounded and slower. Stopped.
- The encoder is NOT the problem: identical adapters (t7/t18) -> embedding distance 0.000000; distinct
  (t6/t19) -> 0.181. The head cannot learn a donor-dependent map from this reward signal.
⇒ Confounded with EXP-34 unless representation and training signal are separated — hence EXP-34.

### EXP-34 — Raw weights vs hand features, SAME labels and objective — **raw weights are WORSE**
Motivation: v1 changed two things at once vs the feature hypernetwork (representation AND training
signal). Isolate representation: same 634 rollout labels, same binary objective, same leave-out-10 protocol.
- **Raw weights (HyperConv encoder): +0.46pp over random (SE 0.37, p=0.245); hand features: +1.31pp
  (p=0.0065).** Per-seed raw: 0.86, 0.68, 1.75, 1.49, 0.15, 0.58, -0.31, -1.50 — a larger model
  overfitting 634 labels, not extracting more signal.
⇒ **Representation was never the bottleneck, and REINFORCE's failure was not the reward signal either.**
Fourth independent route (hand features; donor history; action-error proxy; raw weights under supervised
AND RL) to the same conclusion: the (target, donor) interaction is not recoverable from 76 adapters.
The only lever the learning curve endorses (~140 tasks, 0.018pp/task) is more tasks -> EXP-36.

### EXP-33 — Layerwise dosing probe: does WHERE the donor lands matter? (5 fixed patterns) — done
Motivation: before REINFORCE on a 5-d layerwise coefficient (v2), ask whether ANY non-uniform pattern
beats uniform. Bands vision / llm-early / llm-late / expert / action; every pattern rescaled to the
uniform alpha=0.5 injected Frobenius norm. 5 targets x top-1 donor x 25 rollouts.
- n=5: uniform +10.4, front +10.4, **back +18.4**, expert_only +12.0, vision_only +1.6pp.
  back vs uniform **+8.0pp (SE 6.1, p=0.258)**. Read as a go signal for a fixed back-weighted pattern.
⇒ Validated at fleet scale in EXP-35 — where it did not hold. Pilot was best-of-5 patterns on 5 targets.

### EXP-35 — Back-weighted dosing on ALL 76 targets, paired vs uniform — **NO: uniform stays**
Motivation: if a fixed back-weighted pattern beats uniform, the operator improves for free (no learning,
no extra rollouts). Strictly paired: same targets, same e27 top-3 donors, same 25-rollout commit, same
injected energy per donor (verified, e.g. 4.58 vs 4.58); only the band distribution differs.
- **n=76: uniform +2.22pp, back +1.54pp; paired diff -0.68pp (SE 0.47, p=0.157); back better on 12/76.**
  WEAK n=15: uniform +9.78, back +9.51 (diff -0.27). STRONG n=61: +0.36 vs -0.42 (diff -0.78).
- The pilot's +8pp (n=5) was selection inflation: the best of 5 patterns on 5 targets with ~10pp
  per-cell noise. Third time this week a small-n maximum evaporated at full N.
⇒ **Uniform alpha=0.5 is optimal along the layer axis too.** A layerwise hypernetwork output (v2) has
nothing to find; that branch is closed. The operator is unchanged.

### EXP-36 — Enlarge the fleet: train new LoRA experts (76 -> ~106) — **running (stage 1 passed 2/3 gates)**
Motivation: every learned-selector route fails at 76 tasks and the learning curve says ~140. The only
axis that moves it is more real tasks. Material on the NAS: libero_goal / object / spatial demos, 10 each.
- Trainer `sweep/train_lora.py`: the fleet's exact recipe (r=16, alpha=16, dropout 0, gaussian init, the
  17 target modules from the fleet's adapter_config.json), the model's own flow-matching forward as the
  loss, LoRA params only, byte-identical norm stats to the fleet's (md5 match), same base model.
- Stage 1, ONE adapter (put_the_bowl_on_the_stove, 2,297 samples, 1,500 steps, batch 8, 46.1M trainable):
  (a) trains: loss 0.0755 -> 0.011 — PASS. (b) structure: 458 stems, identical stem set and shapes to the
  fleet, rank 16 — PASS. Delta norm 34.8 vs 59.9 for a fleet adapter (1,500 steps undertrains vs the
  fleet's ~10k; batch run should train longer). (c) 25-rollout test on its own task, with a fleet
  adapter never trained on that task as the control — running (e47_*).
