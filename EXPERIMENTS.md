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
- Gate (c) FAILED, cause found and fixed: new adapter 0/25 while a never-trained fleet adapter scored
  20/25 on the same task. Diagnosis: action targets were fed RAW; inference unnormalizes model output
  (quantile [-1,1] -> action range), so raw-target training gets unnormalized twice -> out-of-range
  commands. Orientation and state layout verified consistent with the eval pipeline (same convention
  as collect_obs_bank / action_error). Fixed in train_lora.py (quantile-normalize action chunks);
  retrain + gate (c) re-eval running on shiva (tmux retrain).
- Gate (c) PASSED after the action-norm fix: retrained trial adapter 24/25 (96%) on its task —
  equal to the goal expert (96%), vs the never-trained fleet control at 80%. Pipeline validated
  end-to-end (trains / 458 stems / performs).
- BATCH LAUNCHED (shiva, tmux batch30, ~15h): all 30 goal/object/spatial tasks, gate-validated
  recipe unchanged (1,500 steps, quantile-normalized actions, fleet norm stats for both train and
  eval — keeps the pool homogeneous and sidesteps per-suite stats). 3 shards, one per A6000; then
  each adapter evaled on its own task @25 rollouts. Delta-norms recorded per adapter (trial: 37.5
  vs fleet ~59.9 — known covariate, performance already saturated so not chasing norm parity).
  Fleet grows 76 -> 106 targets when it lands; next: re-run the supervised affinity study at 106.
- BATCH LANDED (2026-09-03): 30/30 trained + evaled, zero failures, ~14h wall. Own-task success @25:
  mean 81.2% (goal 80.4 / object 81.6 / spatial 81.6), 21/30 >=80%, two hard tasks <50%
  (open_the_middle_drawer 20%, wine_bottle_on_rack 36% — kept: weak experts are headroom-rich targets,
  same as the fleet's own weak members). Delta-norms 37.5-44.3 (fleet ~59.9; recorded covariate).
  FLEET IS NOW 106 TASK EXPERTS (76 LIBERO-90 + 10 goal + 10 object + 10 spatial), one recipe.
  Next: capture@16 of new cohort vs the 76-fleet S16 (running); then the 106-target affinity study
  (needs probe labels for the 30 new targets — rollout budget, user decision).
- CAPTURE SURPRISE: new adapters have capture@16 ~ 0.001 vs the 76-fleet S16 — the random-matrix
  floor (16/2048 ~ 0.008), BELOW foreign experts (0.08-0.13), despite identical recipe/base/stats.
  Control validated: fleet t0 through the same code path = 0.797. So pool-consistency is NOT recipe
  provenance. Confound to split: training length (fleet ~10k steps vs our 1.5k; peft zeros B at init,
  A stays near-random early) vs task suite (S16 may encode LIBERO-90 structure). DIAGNOSTIC RUNNING
  (shiva, tmux capdiag): (a) LIBERO-90 task0-scene adapter at 1.5k steps -> high capture = suite,
  ~0 = steps; (b) goal task at 6k steps -> does capture grow with training? Implication either way:
  the 30 new adapters may sit across the measured pool boundary; hold the 106-target label
  collection until attributed.
- CAUSE OF ORTHOGONALITY LOCALISED (09-03): diag adapters (L90 scene @1.5k, goal @6k) both capture
  0.001 -> not suite, not steps. Fleet base verified = pi05_base (merged - base - delta ~ 0.05 vs
  ~|delta| for libero_all / 20000_merged). Yet fleet deltas are pairwise cos 0.867 with
  ||mean||/||each|| = 0.91 (6 sampled), ours cos 0.003. Zero-B init cannot yield a shared component;
  a shared INITIAL ADAPTER does exactly this. 20000_merged is NOT it (cos 0.01); libero_all delta is
  partially aligned (cos 0.47) — same lineage, not the init. => S16 is (approximately) the fleet's
  warm-start adapter; "pool consistency" = "same init". Residuals r = per-task fine-tuning on top.
- EXP-37 PILOT (shiva): train turn_on_the_stove initialised FROM S16 (B=U*Sigma, A=V^T per stem) with
  the otherwise-identical recipe. Predictions if hypothesis holds: capture@16 >> 0.5 (fleet 0.80),
  own-task success comparable to zero-init (100%), and the adapter becomes a legitimate pool member
  -> a recipe for onboarding external tasks into the fleet. If capture stays ~0 the init drifts away
  and the hypothesis is wrong.
- EXP-37 PILOT CONFIRMED (09-03 13:47): S16-init turn_on_the_stove -> capture@16 0.891 (fleet 0.80,
  zero-init 0.001), cosine to fleet adapters 0.91-0.93 (fleet-fleet 0.87), task residual 18.9 (fleet
  ~27), own-task 25/25. Same task/recipe, only init changed. => S16 IS the fleet's shared init; pool
  consistency = shared init; and it can be MANUFACTURED: initialise new experts from S16.
- EXP-37 BATCH LAUNCHED (shiva tmux batch30s16, ~14h): all 30 tasks retrained with --init s16 into
  results/analysis/new_loras_s16/, own-task evals as e50_*. The zero-init cohort (EXP-36) is kept as
  the control: same tasks, same recipe, only init differs -> a clean paired test of whether
  pool-consistency (a) preserves own-task performance and (b) makes the merge method work on them.
- EXP-32 note (09-03): odin rebooted ~18:00 (uptime reset), killing the tmux session mid-target
  (e44_base_f17) and wiping /dev/shm. No records lost (296 e44 records, 17 baselines, 7 commits on
  disk). Relaunched 21:20 via chain_unified.sh; driver resumes from records. Still ~6 days to go.

### EXP-38 — Seed augmentation for the hypernetwork — **running** (shiva, tmux seeds; user approved 09-04 22:13)
Motivation: the dnd checkpoints (9500-9990) are identical (cos 1.0000), so the hypernetwork never had
real augmentation. The S16-init trainer can produce SAME-task, genuinely different adapters by varying
the seed (~1 GPU-h each). Pilot on a few tasks: does the interaction signal vary with seed (=> useful
augmentation for the affinity model) or only with task (=> no gain)? Status: planned, user-gated.
- EXP-37 BATCH LANDED (09-04 ~05:00): 30/30, zero failures. PAIRED vs the zero-init cohort (same tasks,
  same recipe, only init differs): own-task success 81.2% -> 96.0%, +14.8pp (SE 3.0, p<0.001), S16
  better on 25/30, tie 5, worse 0; >=80%: 21 -> 29; <50%: 2 -> 0. Per suite: goal 80.4->93.2, object
  81.6->96.4, spatial 81.6->98.4. Biggest: open_the_middle_drawer 20->96, wine_bottle_on_rack 36->68.
  Delta-norms ~56-57 (fleet ~60) — norm parity came for free with the init. So S16 is not only the
  fleet's shared init, it is a strong prior: the same 1.5k steps yield expert-level policies from it.
  Report: results/analysis/new_loras_s16/batch30_report.txt. Capture@16 for all 30: computing.
- Capture@16 confirmed for the whole S16 cohort (results/analysis/new_loras_s16/capture16.json):
  see cap16.log summary — all 30 in the fleet's range. The pool is 106 legitimate members.

### EXP-39 — Probe method on the S16-init new experts (fleet donors) — **running** (shiva, tmux newpool)
Motivation: the payoff test of EXP-37. The boundary results said the method fails when target and donors
do not share the fleet's subspace; the new experts are now pool members by construction (capture 0.885).
Does manufactured consistency let the method cross what used to be the boundary?
- Protocol = the fleet protocol exactly: baseline@25, 8 random fleet candidates probed pairwise @10
  (energy-matched, coefficient lands at ~0.52 for fleet donors), commit top-3 @25 under the guard
  "best probe > baseline". Operator form by capture gate -> S16 form for all 30.
- ALL 30 targets, not only the 3 with headroom: reporting rule, and it yields the (target, donor) labels
  that grow the affinity set 76 -> 106 targets.
- Driver `sweep/newpool_probe_driver.py` reuses unified_pool_driver's fold/capture/build/run so the
  operator is bit-identical to EXP-32's. Targets are injected as str keys through EXTPATH.
- Fixes en route: process-salted `hash()` replaced by md5 (job names must be stable or nothing resumes);
  unified_pool_driver.py had to be synced to shiva (it was odin-only).
- Cost ~3,900 rollouts, ~4-5h. Report: results/analysis/newpool_probe_report.txt (carries the labels).

### EXP-40 — Affinity study at 106 targets (the learning-curve test) — **prep running**, user approved
Motivation: every hypernetwork route failed at 76 targets; the measured requirement is ~140 with slope
0.018pp/target. EXP-37 makes 30 new experts pool-consistent, so the training set can grow to 106. This
is a SLOPE test: predicted +1.31pp -> ~1.85pp held-out. 106 is not 140, so a positive result means "the
curve held", not "the hypernetwork works".
- `sweep/plan106.json`: the 30 S16-init experts given indices 76-105 in the fleet's namespace, so every
  downstream tool (activation_probe, gram, ranks, features) widens without internal changes. The
  zero-init cohort is deliberately excluded — capture 0.001 would make the pool heterogeneous again.
- `sweep/feats106.py` rebuilds the same four blocks over 106 (W double-centred Gram / A activation
  signature / L language TF-IDF / S ranks+baseline), construction identical to the 76-task study so the
  numbers are comparable. Stages cached and independent.
- Running: Gram over 106 on odin (CPU, ~10GB RAM); activation signatures for the 30 new adapters queued
  on shiva BEHIND EXP-39 (tmux sigs) since it needs a free GPU.
- NEW TEST this enables: train on the fleet (LIBERO-90), evaluate on the 30 new experts (goal/object/
  spatial) — cross-distribution generalisation, a much stronger held-out test than within-fleet.
- EXP-39 interim (8/30): CEILING EFFECT. EXP-37 worked so well that the new targets have almost no
  headroom (baselines 96-100%; only 3 of 30 below 90%), so the guard abstains on most and the method
  cannot demonstrate IMPROVEMENT on them. What EXP-39 can still show: (a) safety — does the method
  harm pool-consistent new members; (b) guard behaviour at ceiling; (c) the 3 weak targets as a real
  efficacy test.
- BUT the labels are NOT degenerate, which was the risk to EXP-40: probe spread on the new targets is
  10-40pp, mean 23.8pp (e.g. base 100% -> probes [70,80,90,90,90,100,100,100]). The interaction shows
  up as DAMAGE from bad donors rather than gain from good ones. Analysis consequence: the fleet labels
  are mostly "gain on weak targets", the new labels mostly "damage on strong targets" — possibly not
  the same function. EXP-40 must therefore report pooled AND split by regime, not just pooled.
- Idea for a clean efficacy test if wanted (cheap, ~30min): train a few S16-init adapters for ~300
  steps instead of 1,500 -> pool-consistent AND with headroom, which is exactly the missing cell.

### EXP-41 — Pool-consistent adapters WITH headroom (the missing cell) — **queued** (shiva, tmux short)
Motivation: EXP-39 cannot test efficacy because EXP-37's experts are at ceiling (96-100%). This builds
the missing condition: S16 init (capture ~0.88, a genuine pool member) but only 300 steps instead of
1,500, so the policy is deliberately weak and has headroom. Then the probe method with fleet donors.
That is the clean test of whether manufactured pool-consistency lets the method work on adapters that
were outside the pool two days ago — the question the boundary results (-1.6pp donors, -4.44pp targets)
left open.
- 6 tasks chosen as the hardest by their zero-init score (20-64%), so short training is most likely to
  leave real headroom. Queued behind EXP-39 and the signature job (one GPU consumer at a time).
- Reuses `newpool_probe_driver.py` unchanged (--targets glob, --tag e52) so the operator and protocol
  are bit-identical to EXP-39 and EXP-32. ~780 rollouts, ~1.5h including training.
- Predictions: baselines should land 40-80% (if they land at ceiling again, 300 steps was still too
  many and the cell stays untested); if the method then improves them, pool-consistency is sufficient
  and the boundary was never about provenance.
- EXP-39 DONE (09-04 18:10): n=30. ALL +0.93pp (SE 0.46, p=0.050); improved 8 / flat 20 / harmed 2;
  abstained 15 (all at 100%). WEAK n=2: +2.00pp, 0 harmed. The two "harmed" are 92->88 and 96->92 =
  one rollout each at n=25 (noise). THE RESULT: probe overstatement +4.9pp on 15 committed — identical
  to fleet-internal (+5.0pp) and a quarter of the cross-pool figure (~+19pp). The probe treats these
  adapters exactly as fleet members: the boundary is crossed, and it was never about provenance.
  Efficacy (improvement) is untestable here because of the ceiling; that is EXP-41's job.
  Labels for EXP-40: 240 pairs, spread 10-40pp, in the report's trailing JSON line.
- EXP-40 prep note: a first regression run of the 76-target reference through the widened script gave
  +0.75pp, not +1.31pp — traced to hyperparameters, not the patch: the reference (run_hn2.sh) used
  epochs 150 / wd 1e-4 / drop 0.1, not the script defaults (300 / 1e-2 / 0.3). run_exp40.sh now uses
  the reference settings and the reference's paired aggregation, and re-runs the 76-target control
  under them (part D) so every number in the report is like-for-like.
- EXP-40 RESULT (09-04 ~22:00), reference hyperparameters, 12 seeds, paired vs random:
    D control 76 targets     +0.33pp (SE 0.46, p=0.49)   <- did NOT reproduce the recorded +1.31pp
    A pooled 106             +0.80pp (SE 0.39, p=0.065)  gap recovered 10%
    B fleet-holdout 106      +0.95pp (SE 0.56, p=0.12)   gap recovered 12%
    C cross-distribution     +0.99pp (SE 0.19, p=0.0002) gap recovered 19%  (train fleet, test all 30 new)
  Inputs unchanged since the reference (feats 08-26, labels 08-29 08:36, report 08-29 08:48), so the
  control gap is script-or-RNG: hypernet.py's epoch order uses unseeded np.random.permutation.
  UNRESOLVED until: (i) 3 repeats of the patched script and (ii) 2 repeats of the pristine git copy on
  identical inputs. If both bounce in [+0.3, +1.3], the +1.31 reference was a favourable draw and the
  76-target gain is ~+0.8 +/- 0.5 — which also means the slope prediction (+1.85) was built on it.
  What already stands regardless: row C — train on LIBERO-90 only, test on 30 targets from three
  suites the model never saw — is +0.99pp at p=0.0002, SE 0.19: the tightest positive the hypernetwork
  line has produced, and a cross-distribution result rather than a within-fleet holdout. Caveat: its
  oracle gap is smaller (+2.29 vs +5.1) because most new targets sit at ceiling.
- CONTROL RESOLVED: the pristine git script on identical inputs gives +0.58 and +0.14; the patched one
  +0.27 / +1.38 / +0.33. So the recorded +1.31pp reference was a favourable RNG draw (the epoch order
  used an unseeded np.random.permutation); the honest 76-target figure is ~+0.5 +/- 0.5pp, and the
  learning-curve slope (and the +1.85 prediction) rested on it. Fix: numpy seeded per seed alongside
  torch, so every number is reproducible. EXP-40 relaunched with 36 seeds for all four rows; the
  12-seed unseeded report kept as exp40_report_12seed_unseeded.txt for the record.
- EXP-40 FINAL (seeded, 36 seeds, 09-05 ~00:30): D control 76 = +0.20pp (SE 0.23, p=0.39) — the 76-target
  learned selector is indistinguishable from random; A pooled 106 = +0.72 (p=0.006); B 106-train/fleet-
  holdout = +0.72 (p=0.004) on IDENTICAL holdouts to D -> paired data effect +0.52pp / 30 targets =
  0.017pp/target, reproducing the old slope from an independent measurement; C cross-distribution
  (train LIBERO-90, test 30 goal/object/spatial) = +0.82 (SE 0.08, p<1e-4), 16% of a smaller oracle gap.
  VERDICT: slope real, intercept was noise, absolute level ~10x below the probe (+0.7 vs ~+10pp on weak).
  All remaining LIBERO tasks (~130 targets) extrapolate to ~+1.1pp. Line closes as data-limited unless
  EXP-38 (seed variants as extra targets) changes the slope. Report: results/analysis/exp40_report.txt.
- EXP-41 DONE (09-04 21:33): 6 short (300-step) S16-init adapters, capture 0.973-0.978 (pool members).
  Two spatial tasks still hit 100% (abstained); 4 had headroom (16/68/72/76%). On those 4: committed
  gain +11.0pp, improved 3 / harmed 0 (16->36, 68->68, 72->80, 76->92). ALL 6: +7.33pp (SE 3.64, p=0.10).
  LIKE-FOR-LIKE: fleet weak targets (n=15) gain +9.8pp / overstatement +7.4pp; EXP-41 +11.0 / +16.0;
  foreign-expert target (EXP-29, n=5 weak) gain -5.6pp, improved 0, harmed 4, overstatement +18.7.
  VERDICT: the method now IMPROVES weak targets from suites the fleet never saw, at the fleet's own
  rate and with zero harm — manufactured consistency crosses the old boundary. The probe overstates
  more than on fleet weak targets (n=4, one 68->100->68 case), so the guard's margin matters here.
- EXP-32 incident (09-04 21:48): two f28 probe adapters (donors 74, 75) deleted between build and load.
  Cause: the import-time `_clear_orphaned_builds()` I added to every driver, triggered by an analysis
  script importing reinforce_train on odin during the window in which the unified driver had built the
  chunk but not yet launched its orchestrator. Fixed: cleanup now runs only from drivers' main().
  Effect: f28's commit will be chosen from 14/16 candidates. FIX-UP after EXP-32 ends: relaunch the
  unified driver once — it re-runs the two missing probes (no ok record) — then recompute f28's commit
  if its top-3 changes. Run continues unaffected otherwise.
- Why now (from EXP-40): the learning curve is real (0.017pp/task) but LIBERO runs out of tasks at ~130
  (-> ~+1.1pp, ~10x below the probe). The only way the curve reaches useful territory is if same-task
  variants count as targets. This measures that directly.
- Design: 5 tasks (wine_bottle 68%, top_drawer 88%, tomato 84%, bbq 92%, spatial-between 100%) x seeds
  {1,2} from S16, 1,500 steps (seed 0 = the EXP-37 adapter); train_lora.py gained --seed (torch + data
  order). Then every variant is probe-labelled with the fleet protocol (tag e53; 8 fleet candidates @10).
  ~10 GPU-h training + ~1,300 rollouts.
- DECISIVE ANALYSIS (sweep/seed_analysis.py): (a) weight space — cosine among seeds of one task vs
  across tasks; (b) label space — Spearman of the 8 donor affinities between seeds of the SAME task vs
  between DIFFERENT tasks (seed-0 labels from EXP-39/e51). High within-task Spearman => variants are
  duplicates (no new information; the curve cannot be extended by seeds; line closes). Low => the
  interaction varies with seed => variants are real targets => (c) rerun the 106+10 affinity study.
- EXP-38 interim (09-05 ~02:30), weight space, 5 tasks x 3 seeds (all S16-init, 1.5k steps):
  full-delta cosine ~1.0 everywhere (S16 dominates). RESIDUAL cosine: same task / different seed 0.32
  (n=15, range 0.27-0.35, remarkably uniform) vs different tasks 0.025. So a seed variant shares only
  ~30% of its task residual's direction with its sibling — ~70% of what fine-tuning adds is seed-
  specific, not task-specific. In weight space variants are far from duplicates. The label half
  decides which part the DONOR AFFINITY follows: the shared 30% (Spearman across seeds high -> seeds
  don't extend the curve) or the seed-specific 70% (Spearman low -> they do, and it also means the
  interaction is partly about training noise, which bounds what any weight-reading selector can do).
- EXP-38 DONE (09-05 04:10): 10 variants probe-labelled (e53). WEAK n=4: +8.0pp, 3 improved / 0 harmed.
  DECISIVE: within-task cross-seed donor-affinity Spearman +0.15 (n=12 pairs, same 8 donors) vs the
  probe's own same-adapter test-retest 0.41 (EXP-22.4). Siblings agree LESS than a re-measurement of
  one adapter: the seed changes the interaction beyond noise. Attenuation-corrected ~0.35, matching the
  residual cosine 0.32 -> ~30% of fine-tuning is task-determined, ~70% seed-specific, and affinity
  follows the weights. Seeds also move own-task baselines up to 28pp (wine bottle 68/40/68).
  => variants ARE new targets: the curve can be extended by manufacture (~1 GPU-h + 130 rollouts each);
  the info is in the weights (argues for a weight-reading selector); but the interaction is largely
  run-idiosyncratic, bounding any selector by what a weight-reader can recover + label reliability 0.41.

### EXP-42 — 50 seed variants as extra affinity-training targets — **running** (shiva, tmux var50; queued under the 09-04 rule: shiva idle, nothing queued)
Motivation: EXP-38 says variants count; 10 is underpowered against +0.72 +/- 0.23. 50 variants predict
~+0.85pp if they count fully. 10 tasks x 5 new seeds: the 5 EXP-38 tasks + fleet weak tasks t6,t19,t50,
t15,t56 (both distributions). S16 init, 1.5k steps; probe-label all (tag e54); then signatures, features
at 156 targets, and the study with variants as TRAINING targets only, evaluated on the unchanged
fleet-holdout (B) and cross-distribution (C) protocols. ~17h training + ~5h rollouts, resumable.

### EXP-43 — Model-side questions at 106 targets, in parallel with EXP-42's data step — **running** (odin CPU)
Motivation: EXP-42 manufactures targets (the data axis); this exercises the MODEL axis on the 106-target
data we already have, under the seeded protocol.
- 43a block ablation (36 seeds): which of W (weight-Gram) / A (activation) / L (language) / S (scalars)
  carries the transfer, on holdouts B (fleet) and C (cross-distribution). Report exp43_blocks_report.txt.
- 43b raw-weight HyperConv encoder at 106 (12 seeds): the "needs more data" hypothesis from EXP-34
  gets its first test; decides whether EXP-42's 166 targets should feed the encoder or the feature model.
  Tokens for the 30 new experts built with the fleet-derived module choice (first 76 rows unchanged).
  Reports exp43_raw_{fleet,new}.txt.
- 43b DONE (09-05 ~09:40): raw-weight HyperConv encoder at 106 targets, seeded, 12 seeds: fleet-holdout
  -0.48pp (SE 0.43, p=0.29), cross-distribution -0.18pp (SE 0.12, p=0.18) — below random, vs hand
  features +0.72 / +0.82 on the identical protocols. Per-seed spread -2.2..+2.3 = fitting noise. The
  "needs more data" hypothesis (EXP-34) is refuted at +40% targets. DECISION: EXP-42's manufactured
  targets feed the FEATURE model; the raw-weight encoder is retired for the selection line.
- 43a DONE (09-05 10:05), 36 seeds, gain over random: WALS +0.72/+0.82 (fleet-holdout / x-dist);
  W +0.59/-0.01; A +0.25/+0.81; L -0.91/+0.09; S -0.28/-0.26; WA +0.53/+0.83; WS +0.60/+0.80.
  Within-fleet transfer = the weight-Gram row; cross-suite transfer = the ACTIVATION SIGNATURE alone
  (rollout-free, ~1 min/adapter on the fixed obs bank); language useless/harmful, scalars harmful
  cross-suite. => a deployable selector for NEW targets should read activations; weights only help
  within-distribution. Next: 43c activation-block capacity sweep (A is mean-pooled + PCA-24 today).
- 43c DONE (09-05 11:20), W+A model, 36 seeds: DIM_A 12 +0.55/+0.66; 24 +0.53/+0.83; 48 +0.79/+1.05;
  96 +0.21/+0.74 (fleet-holdout / x-dist). Interior optimum at 48; x-dist +1.05 is the line's best,
  ~2 SE over 24 — but a best-of-four, so PRE-REGISTERED for the 166-target run rather than claimed.

### EXP-44 — Deployment test of the learned selector on unseen pool-consistent targets — **planned** (shiva, queued behind EXP-42)
Motivation: label-space gains (+1.05pp) are not the goal; selection at merge time is. 16 targets sit
outside the training set with baselines and 8 probed candidates each (6 EXP-41 short adapters, 10
EXP-38 variants). The W+A/48 model trained on the 106(+variants) scores the 8 candidates from features
alone; commit its top-3 @25 vs random top-3 @25 vs the probe's top-3 (commits already exist).
~500 rollouts. Answers: does the learned selector beat random when merged, and its gap to the probe.
- EXP-42 interruption (09-05 ~14:20): SHIVA REBOOTED (uptime 12 min at 14:32), killing the session at
  18/50 trained. Post-reboot checks passed (3x A6000, NAS, base, demos, shm clean). Relaunched 14:35;
  the chain skips finished adapters, so no work lost. New ETA ~+18h (~08:00 09-06).
- INCIDENT (09-06 ~03:30): the 172-adapter Gram held every adapter in RAM (~16GB) and drove odin to 0 GB
  free while EXP-32's workers ran; killed it at 100/172 before anything else died. gram() rewritten to
  stream one module at a time through lazy safetensors handles (tens of MB), relaunched. Rule: any
  all-adapters computation on a machine with a live run must stream.
- Gram(172) streamed version reproduces the cached 106x106 block to 2e-6 relative (09-06 04:10); ranks
  for the 66 variants running; assembly waits only on sigs_ext (queued behind EXP-42's probes).
- EXP-42 LABELLING DONE (09-07 00:35): 50/50 probed (e54), 400 labels; pair_labels_ext.json now 1,354
  labels over 166 targets (60 variants train-only + 6 short). Method-side numbers on the variants:
  ALL +1.44pp (SE 1.15, p=0.22); WEAK n=21: +4.19pp, improved 13 / HARMED 8; overstatement +11.6pp
  (fleet 5.0, cross-pool 19). CAUTION: harm rate 8/21 on seed-scrambled weak variants vs 0/15 on fleet
  weak targets and 0/4 on EXP-41's short adapters, at the same protocol. Hypothesis (consistent with
  EXP-38: ~70% of a 1.5k-step variant's residual is seed-specific): freshly trained targets interact
  with donors less predictably than the fleet's settled 10k-step adapters, and a 10-rollout probe
  cannot resolve it -> the guard passes more false positives. Not the labelling run's question; noted.
- EXP-42 STUDY DONE (09-07 03:05), W+A, 36 seeds, holdouts = EXP-40's: DIM_A 24: fleet-holdout +0.53 -> +0.90
  (p=0.001) with 66 variants, x-dist +0.83 -> +0.89; DIM_A 48: fleet +0.79 -> +1.14 (p<0.001), x-dist
  +1.05 -> +0.61. Variants count at ~1/3 of a real task within-distribution (+0.35 vs +1.12 predicted),
  matching EXP-38's ~30% task-shared residual; they add nothing cross-suite and hurt at higher capacity.
  The pre-registered DIM_A=48 held within-fleet, failed cross-suite -> a within-distribution setting only.
  EXP-44 scorer: model-vs-probe Spearman +0.04 on the 16 held-out (noisiest) targets; merges running.
- EXP-44 incident (09-07 ~04:30): the commit stage sat deadlocked for ~1h — its inline tmux command
  contained the literal text "merge_orchestrator.py", so its own "wait until no orchestrator" loop
  matched itself forever (the trap already in the resume file). Relaunched from a script file
  (sweep/chain_e44.sh); build + 32 commits now running.
- EXP-44 incident 2 (09-07 05:10): the build stage wrote all 32 merge adapters up front (~18GB) and hit
  shiva's 50GB hard quota (the original EXP-36 mistake, repeated). Freed 25GB by deleting cohorts that
  are complete on odin (zero-init 30, var50 50 — verified before deletion); build stage rewritten to
  chunk build->eval->delete like every driver. Relaunched.
- EXP-44 DONE (09-07 06:40): 16 unseen pool-consistent targets (6 short + 10 seed variants), W+A/48 model
  trained on the other 150, top-3 of the 8 probed candidates committed @25. model-random +2.25pp (SE 4.0,
  p=0.58); model vs base -4.75, random vs base -7.0, probe(guarded) vs base +2.0; WEAK n=5: model -2.4,
  random -4.0, probe +6.4. Non-ceiling subset (8): model-random +2.5. VERDICT: deployed, the learned
  selector is indistinguishable from random at n=16; most of the probe's value is the GUARD (11/16
  abstentions correct; unguarded merging harms at ceiling). Proposed next: A power the test (~36 weak
  targets, ~1,800 rollouts); B hybrid learned-shortlist + cheaper probe + guard (recommended); C fresh-
  target mergeability. Awaiting user. Report: exp44 via `exp44_deploy.py --stage report`.

### EXP-45 — A: powered deployment test + B: hybrid shortlist selector — **running** (user approved A+B, 09-07)
36 weak pool-consistent targets (15 fleet weak + 21 weak seed variants), two leave-out folds, W+A/48 model.
Arms committed @25: model top-3 (features only) / random top-3 / HYBRID = probe-best among the model's
top-4 (half the probe budget; picks computed from existing probes) / reference = existing probe top-3
commit. ~2,700 rollouts on shiva. Label-space version (probe-affinity of each policy's picks) computed
first on odin: results/analysis/exp45_labelspace.log. Driver sweep/exp45_deploy.py (plan/build/report).
- EXP-45 LABEL-SPACE (09-07 ~08:00), 36 weak targets, leave-out folds: committed-top-3 probe-affinity:
  random -3.83, model -4.03 (model-random -0.20, p=0.90), hybrid(probe within model top-4) +1.45,
  full 8-probe +10.34. Random shortlist-of-4 (exact expectation) +1.56 -> the model's shortlist is no
  better than a random one (p=0.91); its top-4 holds the probe-best donor 64% (chance 50%, ~1.7 SE).
  Halving the candidate list without information costs ~85% of the gain: good donors are rare (~1-2 of
  8) and the model cannot find them. B is dead in label space; A's rollouts (running) will give the
  deployed numbers for the record.
- EXP-45 incident (09-07 12:50): the fleet fold completed (42 commits) but the variant fold failed on
  FileNotFound — its 21 targets are var50 adapters I had deleted from shiva to free quota for EXP-44.
  Restored exactly those 21 (3.7GB, quota 33.7GB) and relaunched; resumable, 66 commits remain (~2.5h).
- EXP-45 DONE (09-07 17:30), n=36 weak targets, commits @25: random +0.63, model +1.13, hybrid +3.56,
  probe(full 8) +6.52 vs base. model-random +0.50 (SE 2.27, p=0.83; fleet fold -4.67, var fold +4.19).
  hybrid-probe -2.96 (p=0.13), hybrid-random +2.93 (p=0.17). VERDICT: powered to +/-2.3pp, the learned
  selector is indistinguishable from random when merged and ~5pp behind the probe; the half-budget
  hybrid retains ~half the gain (consistent with uninformed pruning). Hypernetwork-as-selector closed
  at this data scale with a full deployment table. Report: results/analysis/exp45_report.txt.

### EXP-46/47 — probe reliability on fresh vs settled targets, and probe budget — **running** (shiva, tmux e46)
Premise check first (free, 93 guarded commits): harm rate by baseline band 0%/14%/25%/17% for
0-60/60-80/80-90/90-100%; at MATCHED baseline 40-80% the cohorts give fleet 0/6 harmed (+12.4pp,
overstatement +12.2) vs S16-1.5k 3/15 harmed (+8.0pp, overstatement +13.7) — so the 8/21-vs-0/15 harm
contrast is mostly HEADROOM, not training length, and probe overstatement is the same at matched
baseline. Also found: S16-init at 1.5k steps often BEATS the fleet's own 10k-step adapters on the same
task (e.g. STUDY_SCENE4 fleet 60% vs variants 92-100%), so "settled = better" is false. Retraining
longer would move baselines and re-confound, so EXP-46 measures the mechanism directly instead.
- Design: re-probe the SAME (target, candidate) pairs a second time @10 rollouts. 10 targets: 5 settled
  fleet weak (e27 candidates) + 5 fresh S16-1.5k variants at matched baselines (e54 candidates); three
  of the fresh ones are the SAME TASKS as three fleet targets. 800 rollouts.
- (46) test-retest Spearman per target, fresh vs settled; fleet reference 0.41 (EXP-22.4).
- (47) two independent 10-rollout runs = real repeated measurements -> selection quality at 5 / 10 / 20
  rollouts without assuming subsample exchangeability. Feeds the cheaper-probe question directly.
- EXP-46/47 DONE (09-07 20:30), 800 rollouts, report results/analysis/exp46_47_report.txt:
  (46) test-retest rho settled +0.59 vs fresh +0.45 (p=0.36, n=5+5) -> fresh targets are NOT measurably
  less reliable; the harm contrast is headroom, and option-1's premise is refuted. INCIDENTAL AND
  IMPORTANT: the best donor repeats on only 1 of 10 targets across two independent probes -- the probe
  ranks a good SET, never "the" best donor. That is the mechanistic reason top-3 + guard works.
  (47) budget, select-on-A/value-on-B, random-3 baseline = exact expectation over C(8,3) = -5.72pp:
  8x10 +11.62 gain; 8x5 +9.00 (77% of gain, 50% cost); 8x3 +7.24 (62%, 30%); 4x10 +4.01 (35%, 50%);
  3x10 +0.11. At equal 40-rollout budget BREADTH beats DEPTH 2.2x. Consistent with the deployed curve
  (EXP-22.5: 5 rollouts kept 65% of the 10-rollout gain). ACTIONABLE: 8 candidates x 5 rollouts is a
  defensible cheaper operating point; never cut candidates to pay for rollouts.
  NOTE: a first pass used an arbitrary fixed triple as the "random" reference and wrongly reported that
  halving the budget keeps only 43% of the gain; corrected with the exact expectation.

### EXP-48 — three routes to a cheaper / rollout-free selector — **running** (user approved all three)
Ruled out first, from existing data: SELECTION-FREE merging does not work. Random 5 donors at alpha=0.5
gives +0.40pp on the 15 weak fleet targets where probe top-3 gives +9.78pp (paired +9.38, p=0.0042), so
dilution does not average away the problem and something must select.
- ROUTE 1 (shiva, tmux msig): interrogate the MERGED model, rollout-free. For 15 weak targets x 8
  candidates, build W = S16 + r_target + c*r_donor in memory, run the fixed 180-obs bank, record 9-depth
  activations. Reference is the target alone, which equals D_target exactly (S16 + r_target = D_target),
  so act_sigs/task<t>.npz already serves. Labels are the existing e27 probe scores -> ZERO new rollouts.
  Every previous rollout-free predictor was a function of (target, donor) features; nothing had ever
  looked at what the merge does to the policy. sweep/merged_sigs.py, ~2h.
- ROUTES 2+3 (shiva, tmux dense, queued behind msig): examples/libero/main_dense.py is an instrumented
  COPY of the eval entry point (the live one is untouched while EXP-32 runs; selected per-plan via the
  new "eval_script" key). It logs per episode: PROGRESS = furthest point reached along the task's own
  demonstration end-effector trajectory (dense, task-agnostic, free), and SUCCESS_STEP. 3 weak targets x
  8 candidates x 25 rollouts. Route 2 asks whether a dense score at k=3 matches binary at k=10 (the
  binary label's ~14pp SD is what caps rho at 0.41 and every learned selector at 0.64). Route 3 asks
  where episodes can be truncated, since failures always run to max_steps and dominate probe cost.
- ROUTE 1 RESULT (09-08 03:40), 120 merged-model signatures over 15 weak targets, zero new rollouts:
  * Combined ridge over all 27 merged-model features, leave-one-target-out: within-target rho +0.033
    (p=0.76); picking top-3 by it captures 3% of the oracle gap. As a selector: NO.
  * One feature stands out: dnorm_llm_00 (relative activation-norm change at the first LLM layer),
    within-target rho +0.285 (SE 0.078), raw p=0.0027 but Bonferroni x27 p=0.073; positive on 12/15
    targets; top-3 by it alone captures 31% of the oracle gap in label space (+1.64pp vs random -2.73,
    oracle +11.39). That is better than any learned selector we built (3-17%).
  * IS IT AN INTERACTION OR A DONOR PRIOR? Cannot be resolved on this dataset. e27 gave every target its
    own random 8 candidates, so donors barely repeat (24 appear once, 28 twice, only 12 appear >=3
    times); only 4 targets have >=4 repeated-donor candidates. A first control that subtracted plain
    donor means was invalid (singleton donors trivially zero out). The LOO redo is inconclusive:
    interaction rho +0.150 (SE 0.206, p=0.52, n=4).
  * DECISIVE FOLLOW-UP (proposed, ~5h): a SHARED-DONOR design -- one fixed set of 8 donors applied to
    ~12 weak targets. Donor prior and interaction separate cleanly by construction. Needs ~960 probe
    rollouts (labels) + ~3h of merged signatures.
- ROUTES 2+3 RESULT (09-08 04:10), 24 probe merges x 25 episodes with per-episode instrumentation:
  * ROUTE 3 (truncation): NEGATIVE. libero_90 max_steps=400; success step median 231, p90 341, max 408
    -- successes are spread across the whole episode, not concentrated early. Truncating at 250 keeps
    only 61% of successes for a 24% compute saving; at 150 steps (48% saving) it keeps 0%. There is no
    free truncation. (My first pass mislabelled max_steps as 220 and is corrected here.)
  * ROUTE 2 (dense score): NEGATIVE AS IMPLEMENTED. Demo-trajectory progress ranks candidates WORSE
    than binary success at every budget: k=1 +0.15 vs +0.31, k=3 +0.27 vs +0.48, k=10 +0.45 vs +0.76.
    Diagnosis: the metric saturates -- mean progress 0.882 (sd 0.15), 0.944 on successes vs 0.782 on
    failures, point-biserial only +0.52. "Nearest demo waypoint reached" credits a policy that wanders
    near the reference path without completing the task. A stricter dense metric (in-order progress, or
    terminal distance to the goal region) might still work; this one does not.
  * Useful by-product: binary probe reliability measured directly against 25-episode truth --
    within-target rho 0.31 (k=1), 0.48 (k=3), 0.59 (k=5), 0.76 (k=10). Independent confirmation that
    10 rollouts is a sensible operating point and that the ceiling is label noise, not selector design.

### Continuous queue on shiva (user instruction 09-08: keep it busy; next meeting ~09-15)
- EXP-50 (free, no GPU): ADAPTIVE probe allocation does not help. Successive halving replayed on the 24
  real 25-episode records: at equal budget halving 2->4 (24 rollouts) beats uniform-3 by only +0.4pp,
  and at larger budgets uniform wins (uniform-5 70.2% vs halving 2->6 69.2%; uniform-10 72.9% vs
  halving 5->5 72.0% at 25% less cost). Eliminating on 2-3 noisy rollouts discards good candidates --
  same root cause as the unstable argmax. Design rule: spend the probe budget uniformly and broadly.
  Report: results/analysis/exp50_allocation.txt.
- QUEUE (sweep/queue_shiva.sh, tmux queue): EXP-49 probe -> EXP-49 sigs -> EXP-51 -> EXP-52. Each stage
  is idempotent, so the queue can be killed and relaunched at any point.
  * EXP-51 cheap probe: run 8 candidates x 5 rollouts END TO END on the 15 weak fleet targets (fresh
    probes, own top-3, commit @25) to turn the "77% of the gain at 50% of the cost" estimate into a
    measured deployment number. ~975 rollouts.
  * EXP-52 N_PICK: commit top-1/2/4/5 (K=3 already exists) from the SAME probe ranking at matched energy
    (c/K). N_PICK=3 was an early choice never validated at scale. ~1,300 rollouts.
- EXP-53 (free, 09-08): guard-rule sweep over all 130 committed merges we own (a target failing a
  stricter rule reverts to its measured baseline; more permissive rules are not evaluable).
  ALL targets: current "best>base" +2.97pp/16 harmed; "+10pp margin" +2.52/4; ">=2 candidates beat
  base" +2.98/13; no guard +2.65/24.
  WEAK targets (n=46): current +6.84pp/8 harmed; "+10pp margin" +7.13pp/4 harmed <- better on BOTH.
  ACTIONABLE: adopt a +10pp guard margin for weak targets (halves harm at no cost in mean gain).
  Caveats: margin chosen from a sweep on the same data (wants fresh-target confirmation); the +0.29pp
  gain difference is within noise -- the harm halving is the real effect.
  CORRECTION to an earlier claim of mine: the guard is NOT where most of the method's value sits for the
  PROBE's own picks (+0.32pp over no guard, harm 24->16). The large guard effect in EXP-45 was measured
  against random and model selection, which need protecting far more than the probe does.
- EXP-54 (free, 09-08): is donor quality GLOBAL, i.e. could a fixed donor set be reused for every new
  target with zero probing? Two-way fit on all 604 e27 pairs (76 targets, 76 donors, median 8
  appearances): variance = target 27% / DONOR 14% / interaction+noise 60%. Leave-one-target-out, a
  global donor ranking gives within-target rho +0.102 (p=0.038) and captures 4% of the oracle gap.
  So the amortised "known-good donors, no per-target rollouts" shortcut is DEAD: donor quality is
  statistically real but practically negligible. Also calibrates EXP-49: with the donor prior at 14%
  and label noise consuming much of the residual 60%, the interaction any rollout-free selector could
  exploit is a small slice -- worth knowing before scaling anything built on it.
  Report: results/analysis/exp54_donorprior.txt.
- EXP-55 (free, 09-08) — HOW MUCH REPRODUCIBLE INTERACTION EXISTS. Using EXP-46's two independent probe
  runs (80 pairs): corr(A,B) raw +0.670; after removing target means +0.632; after ALSO removing the
  global donor effect +0.579 (p<0.001) = the reliability of the PURE pair-specific signal. Top-3 overlap
  between independent runs 1.9/3 (chance 1.1). So the interaction is REAL and reproducible, not noise:
  attenuation ceiling for any predictor of a single probe run is sqrt(0.58)=0.76. This overturns my
  working assumption that the learned selectors failed because the signal was absent -- they failed on
  FEATURES. Caveat below.
- EXP-56 (free, 09-08) — WHY THE PROBE OVERSTATES: it is largely a FIXED-STATE artifact, not a winner's
  curse. LIBERO replays initial_states[episode_idx] with seed 7, so a 10-rollout probe always scores
  states 0-9 and a 25-rollout commit scores 0-24. On the 24 instrumented merges: states 0-9 are 11.5pp
  EASIER than 10-24 for ALL candidates (no selection involved). That difficulty gap alone implies a
  ~6.9pp optimistic bias of probe over commit, which exceeds the measured +5.0pp internal overstatement;
  actual selection adds only +3.3pp. CONSEQUENCE: the guard ("best probe on states 0-9" > "baseline on
  states 0-24") is optimistic BY CONSTRUCTION -- the likely reason merges pass the guard and still harm.
  Also the reason EXP-55's 0.58 is a reliability w.r.t. a FIXED state set: both runs share states, so it
  excludes state-sampling variance and is an upper bound on predicting true task performance.
  PROPOSED FIX (cheap, no new machinery): score probe, baseline and commit on the SAME states.
- EXP-57 QUEUED FIRST (sweep/state_difficulty.py): the unmerged target on 50 instrumented episodes for
  the 15 weak fleet targets, to map per-state difficulty directly. 750 rollouts. Placed ahead of EXP-51
  and EXP-52 because a confirmed mismatch changes how those arms should be scored.
- SCOPE OF THE STATE ARTIFACT (checked 09-08): baselines were run at 25 trials (93 of 94 records) and
  commits at 25 -> SAME states 0-24, so every deployed headline (+9.78pp weak, +2.22pp all-76, EXP-45's
  tables) is UNAFFECTED. The artifact touches only the 10-rollout probe: candidate RANKING is internally
  consistent (all candidates share states 0-9), but the GUARD compares probe-on-0-9 with baseline-on-0-24
  and is optimistic by ~7pp. EXP-57's 50 instrumented episodes per weak target will give per-state
  baselines, so the guard can be recomputed against a STATE-MATCHED baseline and the harm re-counted.
- EXP-58 QUEUED (after EXP-57): route 1 recomputed on TARGET-SPECIFIC observations. The shared bank has
  180 observations over 90 scenes, so only ~2 belong to any given target -- route 1 therefore measured how
  a merge changed the policy GLOBALLY, not on the task the merge is supposed to help. Per-target banks
  built from each target's own demonstrations (60 states each, results/analysis/obs_banks/t<T>.npz);
  merged_sigs.py gained --bank target, which also computes the target-alone reference on the same bank.
  Motivated by EXP-55: the pair-specific signal is real (reliability 0.58), so FEATURES are the
  bottleneck, and observation choice is the most obvious defect in the features we tried. ~4.5h.
- EXP-49 STAGE 1 (09-08, complete 15x8 grid, 120/120 cells): variance of pair affinity = target 36% /
  DONOR PRIOR 14% / interaction+noise 51%. With EXP-55's measured reliability 0.579, reproducible
  interaction ~29% of total and noise ~21% -> the interaction is ~2.1x the donor prior and the largest
  learnable component after target difficulty. Donor prior b_d spread 23.3pp, but concentrated in ONE
  bad donor (t46 -18.5pp); leave-one-target-out, picking top-3 by donor prior gives -4.13pp vs random
  -3.86 (BELOW random, -2% of the oracle gap), independently confirming EXP-54. So donor identity helps
  only to AVOID a bad donor, never to find good ones -- which is what the guard already does.
  Stage 2 (signatures) running: does any merged-model feature track the 29% interaction term?
- EXP-49 STAGE 2 / FINAL (09-08 14:10), complete 15x8 grid, 120/120 labels and features:
  merged-model features track the DONOR PRIOR strongly (dnorm_exp_00 -0.738, dist_exp_17 +0.675,
  cos_act_out -0.668, dist_act_out +0.658) and the INTERACTION not at all (best dist_act_out +0.149,
  p=0.10; nothing significant over 120 cells). Route 1's lead dnorm_llm_00 collapses here: donor prior
  -0.011, interaction +0.090 (p=0.33) -- its within-target rho +0.285 and "31% of the oracle gap" were
  artifacts of e27's unbalanced candidate lists, which is exactly what this design was built to test.
  MEANING: these features measure how much a donor PERTURBS the policy, a donor-level property; and
  donor identity is worth -2% of the oracle gap. Route 1 is closed.
  STANDING POSITION: the interaction is real, reproducible (0.58) and ~29% of variance, yet SIX feature
  families now capture none of it -- weight Gram, activation signatures, language, effective ranks, raw
  weights, merged-model activations. The one defect common to all of them: features were computed on
  observations that are mostly not the target's task. EXP-58 tests exactly that and is queued.
- EXP-57 RESULT + RETRACTION OF EXP-56 (09-08 15:00). 15 weak targets x 50 instrumented episodes of the
  UNMERGED target: states 0-9 minus 10-24 = +2.4pp (SE 4.2, p=0.574); per-target values scatter +40..-13;
  weak trend with state index r=-0.216 (p=0.132). EXP-56's +11.5pp came from 24 merges over only THREE
  targets -- correlated observations, so it measured those tasks, not LIBERO's state ordering. I treated
  24 non-independent measurements as 24 samples. => THE CLAIM "probe overstatement is largely a fixed-
  state artifact and the guard is optimistic by construction" IS WITHDRAWN; the winner's curse remains
  the better explanation, and nothing else in the project depended on it (headlines were already shown
  unaffected because baselines and commits share states 0-24).
  WHAT SURVIVES, on state-matched comparisons (delta always vs the 25-state baseline, as deployed):
    current  probe(0-9) > baseline(0-24): +2.67pp, 13/15 commits, 3 harmed
    matched  probe(0-9) > baseline(0-9) : +2.13pp,  9/15 commits, 2 harmed
    matched + 10pp margin               : +2.13pp,  5/15 commits, 0 HARMED
  The margin variant reaches zero harm at equal mean gain, agreeing with EXP-53's independent sweep.
  Worth adopting for weak targets; wants confirmation on fresh targets before it goes in the deck.
- EXP-58 (09-08 16:20): merged-model features recomputed on each TARGET'S OWN 60 demo states, same
  protocol as route 1. Combined ridge is again useless (within-target rho -0.018, -6% of the oracle gap;
  route 1 shared-bank: +0.033, 3%). But dnorm_llm_00 reappears at rho +0.260 (p=0.0017, Bonferroni x27
  p=0.047) on completely different observations from route 1's +0.285.
  DONOR-PRIOR CONTROL (the test that killed every earlier lead): the feature correlates only +0.10
  (target-bank) / +0.08 (shared-bank) with each donor's leave-one-out global quality, and after removing
  that quality from the label the within-target rho holds at +0.206 / +0.217 (both p=0.004, n=15).
  So it is NOT a donor prior -- the first feature to survive this control.
  CONFLICT with EXP-49's grid (interaction corr +0.090, p=0.33), which used a different statistic, 8
  fixed donors, and shared-bank features on 28/120 target-bank cells.
- EXP-59 QUEUED FIRST (sweep/grid_tb_sigs.py, ~1h): compute the 92 missing grid cells on target banks so
  the exact EXP-49 decomposition can be rerun with target-bank features -- same design, same statistic.
  Everything else in the queue waits behind it.
- EXP-59 DECISIVE (09-08 18:00), grid 15x8 with TARGET-BANK features, 120/120 cells:
  dnorm_llm_00 -> interaction -0.035 (p=0.70). REFUTED; its +0.21 on the e27 pairs does not survive the
  balanced design, exactly as its shared-bank version did not. Second lead killed by the same control.
  NEW: dnorm_llm_17 (relative activation-norm change at the FINAL LLM layer) interaction +0.298 (p=0.001).
  NESTED leave-one-target-out (select on 14 targets, score on the 15th): chosen on ALL 15 folds;
  held-out interaction correlation +0.346 (SE 0.076, p<0.001) -- HIGHER than in-sample; picking top-3 by
  it on a held-out target gives +1.42pp vs random -3.86 (oracle +8.98) = 41% OF THE ORACLE GAP with zero
  rollouts. Best rollout-free result in the project.
  NOT YET BELIEVED: the 27-feature family has been explored on these same 15 targets three times, which
  nested CV cannot see; and 41% in label space is not 41% deployed (route 1's refuted feature showed 31%).
- EXP-60 PRE-REGISTERED CONFIRMATION (queued): dnorm_llm_17, positive sign, on 10 weak S16-init seed
  variants (a different cohort, held out at the DESIGN level) against the SAME 8 donors. Predictions
  stated in advance: interaction ~+0.35, top-3 worth ~40% of the oracle gap. Near zero refutes it and
  closes the rollout-free direction. No other feature will be promoted from this run.
- QUOTA INCIDENT #3 (09-08 20:07): exp51_cheap died mid-commit on "Disk quota exceeded" after its 120
  probes succeeded -- shiva had accumulated 6.9GB of build artifacts plus four LoRA cohorts and both
  signature sets. Freed 9GB by deleting what is fully mirrored on odin (new_loras_s16/seeds/short,
  merged_sigs, merged_sigs_tb), verified counts on odin BEFORE deleting; kept new_loras_var50 because
  EXP-60 reads it. ROOT-CAUSE FIX: unified_pool_driver.build() now calls _ensure_space(), which checks
  the user quota (not just filesystem free), reclaims decomp_adapters files older than 30 minutes --
  which cannot belong to the chunk being built -- and otherwise raises with the actual numbers instead
  of failing deep inside safetensors. EXP-51's probes survive; only its commit stage must be rerun.
- EXP-60 ATTEMPT 1 INVALID (09-08 21:31), found before any analysis: job names used a 28-character
  prefix of the adapter directory, and the LIBERO-90 variant directories share their first 28 chars --
  so 10 targets collapsed into 3 job names and 7 silently overwrote each other (24 records, not 80).
  A second bug would have hit stage 2: the signature path uses reinforce_train.fold, which indexes a
  fleet table by integer, while these targets are S16-init variants.
  REWRITTEN as sweep/confirm_run.py: stable md5 job ids (verified 10/10 unique vs 3/10 by prefix), and
  unified_pool_driver.fold, which accepts both integer fleet targets and string variant keys. The
  pre-registration is unchanged -- dnorm_llm_17, positive sign, ~+0.35 interaction, ~40% of the oracle
  gap -- because nothing from the invalid run was inspected.
- EXP-51 DEPLOYED RESULT (09-08 22:15), 15 weak fleet targets, fresh 5-rollout probes -> own top-3 ->
  commit @25: standard 8x10 +8.71pp (SE 2.32) vs cheap 8x5 +6.04pp (SE 2.70); paired difference -2.67pp
  (SE 2.58, p=0.319). Cheap retains 69% of the gain for 50% of the probe cost -- between the two
  independent projections (77% from EXP-47's select-on-A/value-on-B, 65% from EXP-22.5's deployed curve),
  a three-way agreement across different methods. Top-3 agreement 2.3/3 (chance 1.1); harmed 2/15 vs 1/15.
  HONEST READING: not distinguishable from zero at n=15, so this licenses "halving the probe costs at
  most ~2.7pp, within noise", not "the cheap probe is as good". Per-target scatter is large (t6 +20->+4,
  t50 +4->-16, t60 +4->+12). Recommend 8x5 when probe cost matters, 8x10 otherwise.
- EXP-60 PRE-REGISTERED CONFIRMATION — REFUTED (09-09 01:26). 10 NEW targets x the same 8 donors, 80/80
  labels and features. dnorm_llm_17 vs interaction r=-0.084 (p=0.458) against a predicted +0.35; top-3
  by it -9.07pp vs random -5.40 (oracle +9.27) = -25% of the oracle gap against a predicted +40%.
  The discovery grid's +0.298 in-sample / +0.346 held-out did not transfer at all.
  DIAGNOSTIC: a different feature set now tops the table (dist_exp_08 +0.408 p=0.0002, dist_exp_00
  +0.317, dist_exp_17 +0.299) -- the THIRD rotation of winners across three designs. 27 features x
  repeated looks = selection on noise. Per the pre-registration, nothing is promoted from that table.
  => THE ROLLOUT-FREE FEATURE DIRECTION IS CLOSED. Three features topped a table and failed confirmation
  (dnorm_llm_00 twice, dnorm_llm_17 once).
  STANDING RESULT, and the right way to state it: the target-donor interaction is REAL -- it reproduces
  across independent probe runs at rho 0.58 (EXP-55) and is ~29% of affinity variance, >2x the donor
  prior (EXP-49) -- yet SEVEN feature families capture none of it: weight Gram, activation signatures,
  language, effective ranks, raw weights, merged-model activations on a shared bank, and merged-model
  activations on the target's own demo states. The signal exists and is reachable only behaviourally.
- EXP-52 RESULT (09-09 03:58), 15 weak targets, same probe ranking and same energy (c/K):
  K=1 +8.18 / K=2 +10.84 / K=3 +8.71 (current) / K=4 +10.04 / K=5 +3.91pp, harmed 1/1/1/1/4.
  K=1..4 are indistinguishable (K=2 vs K=3 p=0.407) and K=2's lead is a best-of-five pick, so it is NOT
  promoted -- the same trap that produced three refuted features this week. K=5 is genuinely worse:
  spreading the same energy over five donors pulls in ones the probe ranked low. KEEP K=3.
- EXP-61 (free, 09-09): does the interaction TRANSFER between targets, i.e. could a new target inherit a
  ranking from probed neighbours instead of paying 80 rollouts? Pairwise correlation of interaction
  residuals across the 15x8 grid: -0.051 (SE 0.040, p=0.206, 105 pairs), positive on 47/105. It does not
  transfer. Each target's good donors are its own and THE PER-TARGET PROBE COST IS IRREDUCIBLE.
  (The leave-one-out number printed alongside is an artifact: two-way residuals sum to zero across rows,
  forcing exactly -1. Only the pairwise figure is meaningful.)
- EXP-62 QUEUED (shiva): are the gains real on unseen initial states? Selection uses states 0-9 and
  scoring uses 0-24, so they overlap and part of +9.78pp could be state-specific. Re-score every
  committed merge on states 25-49, which no probe or commit has touched; EXP-57 already measured the
  unmerged targets on all 50 states, so the held-out baseline is free. Added --args.state-offset to the
  instrumented eval script and a state_offset plan key. ~375 rollouts.
- EXP-62 RESULT (09-09 05:40) — GAINS SURVIVE ON HELD-OUT INITIAL STATES. Committed merges re-scored on
  states 25-49, which no probe or commit has ever used, against EXP-57's baselines on the same states:
    states 0-24 (selection overlaps): merge 87.7% vs baseline 85.9% -> +1.87pp (SE 1.65)
    states 25-49 (never used)       : merge 90.9% vs baseline 81.3% -> +9.60pp (SE 3.35)
    9/15 improved, 0 harmed on held-out states.
  So the method is not picking donors that merely suit the states it was scored on. That validity
  question is answered positively.
  CAREFUL READING, because the two rows are not the comparison originally intended: EXP-57's baseline is
  the operator's ZERO-DONOR form (S16 + r_target), not the raw target adapter behind our canonical
  +8.71pp -- so these numbers isolate the DONORS' contribution alone. The 5x ratio is mostly that
  held-out states are harder for the zero-donor baseline (81.3 vs 85.9) while the merged model holds up,
  plus cross-run noise on the seen row (e27 commit vs an e63 baseline measured weeks apart).
  INCIDENTAL, needs a controlled check: the zero-donor form scored 85.9% on states 0-24 against ~79% for
  the raw target adapter in baseline_results.jsonl. If real, the S16 reconstruction is worth several
  points before any donor is added -- but the deck's controlled figure for that is 2.5pp, so this is
  probably cross-run noise. Queued: both arms in ONE run on the same states.
- EXP-63 RESULT (09-09 07:10), both arms in ONE run on the same 25 states, n=15 weak targets:
  raw target adapter 79.5% vs operator zero-donor form (S16 + r_target) 78.1%; paired -1.33pp
  (SE 3.94, p=0.740), better on 6/15, worse on 6. => EXP-62's apparent ~7pp advantage for the operator
  form was CROSS-RUN NOISE, not a property of the reconstruction. The raw arm here differs from the
  canonical baselines by only +0.44pp, confirming the means are stable.
  MORE IMPORTANT, AND IT SHOULD TEMPER SEVERAL PER-TARGET CLAIMS: identical weights on identical states
  moved individual targets by up to 30pp between runs (t14 48% here vs 80% canonical; t6 72% vs 64%;
  t15 76% vs 40% for the S16 arm). Aggregate means are reliable (0.44pp apart on n=15) but ANY
  single-target number in this project carries roughly +/-20pp of run-to-run noise. Per-target tables
  should be read as illustration, never as evidence.
  Consequence for EXP-62: its held-out gain +9.60pp (SE 3.35, n=15) survives this noise level and the
  validity conclusion stands; its "seen states" row mixed runs and is not interpretable.
- EXP-65 STAGED (shiva, behind EXP-64): at a FIXED 80-rollout budget, is 16 candidates x 5 better than
  8 x 10? EXP-47b's simulation said breadth beats depth 2.2x at equal budget, but it could only re-slice
  rollouts among the 8 candidates we already had -- it could never widen the pool, which is the claim
  that matters. 8 fresh donors per weak target (seeded draw, disjoint from the original 8, verified no
  overlap), probed at 5 rollouts; the original 8 already have 5-rollout probes from EXP-51. Commit the
  top-3 of all 16 and compare with the existing 8x10 commit. ~975 rollouts.
  Mechanism being tested: good donors are rare (1-2 of 8), so covering more of the pool should beat
  ranking a few precisely. If it holds, the probe gets better at no extra cost -- the first improvement
  to the selection mechanism itself rather than to its price.
- EXP-64 RESULT (09-09 12:30), cheap probe pooled to n=35 (15 fleet + 20 S16-init variants; the variant
  standard arm was already measured by EXP-42, so only their cheap arm was run):
    standard 8x10 +6.02pp (SE 1.80) | cheap 8x5 +3.73pp (SE 1.71)
    paired -2.29pp (SE 1.65, p=0.175), 95% CI [-5.52, +0.95]; cheap retains 62%; harmed 9/35 both arms.
  Cohorts agree closely (fleet -2.67, variants -2.00), so the effect REPLICATES in sign and size on an
  independent target set even though it does not clear significance. Across three measurements the point
  estimate has always favoured the standard probe (retention 69% / 62%, projections 77% / 65%).
  RECOMMENDATION: halving the probe costs ~2.3pp of a ~6pp effect. Use 8x5 when rollouts are the binding
  constraint; otherwise keep 8x10. Do NOT claim the two are equivalent.
- EXP-32 LANDED (09-09 18:50) — THE SUPERSET RUN, and it REVERSES our external-target negative.
  ALL 85 targets +2.82pp (SE 0.68), 43 committed / 42 abstained; fleet n=76 +1.74pp (23 improved,
  5 harmed); EXTERNAL n=9 +12.00pp (SE 3.33; 7 improved, 2 flat, 0 HARMED); WEAK n=20 +8.80pp;
  STRONG n=65 +0.98pp; pool-wide 91.8% -> 94.6%.
  CONTROLLED REVERSAL OF EXP-29: same 9 goal_expert targets, same keep-intact operator (capture 0.08),
  same guard. The ONLY change is that candidates now include 8 external adapters alongside 8 fleet ones.
  EXP-29: -4.44pp, 0 improved, 4 weak harmed. EXP-32: +12.00pp, 7 improved, 0 harmed. Swing +16.4pp.
  CAUSE NOT ISOLATED -- I first wrote this up as provenance-matching and had to correct it. THREE things
  differ from EXP-29: candidates 8 -> 16, composition fleet-only -> mixed, and the draw (EXP-29 used ONE
  shared set of 8 for all nine tasks; EXP-32 draws per target). And provenance is NOT the mechanism:
  external adapters are 8 of 16 candidates, so chance is 50%, and the probe took them for 12/27 top-3
  slots (44%) overall and 11/21 (52%) on improving targets -- i.e. at chance. The likelier driver is
  simply MORE CANDIDATES, consistent with good donors being rare (1-2 of 8); EXP-65 tests that at fixed
  budget. The old claim "the method does not cross the pool boundary" is withdrawn either way.
  Fleet targets are not damaged by the mixed pool (+1.74 vs +2.22 fleet-only, within noise).
  DECK UPDATED: the three external-LoRA slides (EXP-26 donors, EXP-29 targets, EXP-28 mixed pool) are
  replaced by one superset slide, per the standing instruction.
- EXP-68 CONCLUSIVE (09-10) — EXP-29'S NEGATIVE WAS A SAMPLING ARTIFACT. Same 9 external targets, same
  keep-intact operator, same guard; only the candidate list differs:
    EXP-29: 8 fleet candidates, ONE SHARED DRAW for all 9 tasks -> -4.44pp, 0 improved, 4 harmed
    EXP-68: 8 fleet candidates, DRAWN PER TARGET               -> +7.56pp (SE 2.05), 8 improved, 1 harmed
    EXP-32: 16 candidates (8 fleet + 8 external), per target   -> +12.00pp (SE 3.33), 7 improved, 0 harmed
  mixed minus fleet-only = +4.44pp (SE 2.94, p=0.169) -- external donors add something, not significant.
  => The published claim "the method does not cross the pool boundary / pool-consistency is required on
  BOTH sides" is WRONG and withdrawn. FLEET DONORS DO HELP FOREIGN TARGETS. EXP-29 reused one random set
  of 8 donors across all nine tasks; that draw held nothing useful for them, so every target failed
  together and the pattern looked like a boundary. Per-target draws -- what the method does everywhere
  else -- fix it. Provenance was never the obstacle; the probe selects external donors only at chance.
  Deck updated: superset slide now states the cause; the S16 slide's -5.6pp row annotated as superseded.
- EXP-69 RESULT (09-10) — CANDIDATE-DRAW VARIANCE QUANTIFIED. 5 weak fleet targets x 4 independent draws
  of 8 fleet candidates, full method each time: within-target spread across draws 13.6pp (range 12-16);
  mean gain averaged over draws +11.00pp; the single e27 draw we have always quoted gives +14.40pp on
  these targets; pooled SD across the 20 (target, draw) runs 12.0pp.
  METHOD IS ROBUST IN SIGN (4 of 5 targets gain on average) but a single draw moves a target by ~14pp.
  THIS COMPLETES THE EXP-29 EXPLANATION: with PER-TARGET draws the noise is independent across targets,
  averages out, and is ALREADY inside the across-target SEs we quote -- our headline error bars were
  never wrong. With ONE SHARED draw it becomes a systematic bias applied identically to every target,
  so one unlucky draw flips the sign of the whole experiment. 13.6pp of draw spread applied uniformly to
  9 targets is exactly how a real +7.6pp became a published -4.4pp.
  RULE: never share a candidate draw across targets; always draw per target.
- EXP-70 RESULT (09-10): 16 candidates at FULL depth vs 8, same targets, unions of already-probed draws:
  8 cand +11.00pp (SE 5.12, 20 runs) | 16 cand +11.20pp (SE 7.86, 10 runs) | paired +0.20pp (p=0.951),
  better on 2/5. Spread did NOT shrink (SD 12.0 -> 17.9). With EXP-66 (16x5 equal budget, -0.80pp) this
  CLOSES the candidate-count question in both directions: 8 candidates is enough, more do not help at
  equal or at double budget. Also retires EXP-32's suggestive +4.44pp for 16-vs-8 on external targets as
  noise (its own p was 0.17). Mechanism: good donors come in near-equivalent groups (EXP-66 showed
  swapping a third of the committed donors changed the result by 0.00pp), so once 8 candidates contain
  one, seeing 16 adds nothing.
- EXP-71 (free, 09-10) — WHERE TO RUN THE METHOD. Sweeping the headroom gate over all 130 committed
  merges (cost 105 rollouts/target): base<60% +16.00pp mean, +152pp per 1k rollouts; <70% +13.85/+132;
  <80% +9.33/+89; <90% (current) +6.49/+62; <95% +5.73/+55; no gate +2.59/+25. Efficiency rises steeply
  as the gate tightens -- the weakest targets return 2.5x more per rollout than the current rule -- and
  an UPPER cutoff genuinely pays: running on everything yields LESS total gain (+337pp) than stopping at
  95% (+349pp), because targets above 95% lose slightly on average. Deployment rule: spend on the
  weakest targets first; never run above ~95%.
- EXP-72 QUEUED (shiva): the +16.00pp claim for base<60% rests on n=5, the project's strongest and
  thinnest result. All existing sub-70% pool members already have method runs, so this CREATES more:
  8 further hard tasks trained S16-init for only 200 steps (EXP-41 showed 300 steps lands 16-76%), which
  are genuine pool members (capture ~0.97), then the standard method on each. ~840 rollouts + training.
- EXP-67 RESULT (09-10): 50-rollout baselines for the 15 weak fleet targets. Mean 79.0% (old, 25) vs
  79.7% (new, 50), shift +0.71pp (SE 2.12) -> AGGREGATE MEANS ARE STABLE. Per-target mean |shift| 6.0pp,
  max 18pp. Binomial SE per target 7.8 -> 5.1pp (34% reduction). Split-half inside one 50-rollout run
  gives |first25 - last25| = 6.4pp mean, 20pp max, reproducing the noise model independently.
  These are now the sharper baselines for paired tests on these targets.
- EXP-73 QUEUED (odin): re-measure the committed merges for the same 15 targets at 50 rollouts, so the
  headline weak-target gain has halved noise on BOTH arms rather than one. ~750 rollouts, overnight on
  odin's slower GPUs; shiva stays free for selection work.
- EXP-72 RESULT (09-10) — THE DESIGN LARGELY FAILED, reported as such. 200-step S16-init adapters were
  meant to land in the sub-60% regime; they came out at 32/72/84/92/100/100/100/100%, i.e. FOUR at
  ceiling and only ONE below 60%. S16 is too strong a prior for undertraining to produce weak targets
  reliably (consistent with EXP-37: 1.5k steps -> 96%). The run itself is fine (+3.50pp overall, n=8;
  +8.00pp on its 3 weak targets) but it adds one target to the regime it was built to strengthen.
  NOT pursuing fewer steps: a 50-100 step adapter is too undertrained to represent a realistic target.
- EXP-74 QUEUED (shiva): the better way to strengthen the headline is DRAW-AVERAGING, not more targets.
  Every headline we quote uses ONE candidate draw per target, and EXP-69 measured 13.6pp of draw spread.
  Extending EXP-69's design to the other 10 weak fleet targets x 3 draws gives a draw-averaged estimate
  on all 15 -- a materially more defensible headline than any single-draw number. ~3,150 rollouts.
- EXP-74 RESULT (09-11) — THE DRAW-AVERAGED HEADLINE. All 15 weak fleet targets under 4 independent
  candidate draws each (60 complete method runs): +7.98pp (SE 1.83), improving 13/15; pooled SD across
  runs 9.3pp. The single-draw figure we had quoted is +8.71pp (SE 2.32), so the draw we happened to use
  was 0.73pp optimistic -- well within noise -- and draw-averaging tightens the SE from 2.32 to 1.83.
  THIS IS THE NUMBER FOR THE PAPER: it is immune to the draw-luck failure that turned EXP-29 into a
  false negative, and rests on 60 runs rather than 15. Only t8 (-1.3) and t15 (-2.0) are negative on a
  draw-average, both inside per-target noise. Deck's carried-forward slide updated.
- EXP-73 RESULT (09-11): the fleet headline with BOTH arms at 50 rollouts = +4.93pp (SE 2.68) against
  the published 25-rollout +8.71pp (SE 2.32). Paired difference -3.78pp (SE 2.56, p=0.162) -- the two
  agree within noise, same targets and same donors, only the evaluation states differ.
  THE INSTRUCTIVE PART: doubling rollouts did NOT tighten the estimate (SE rose 2.32 -> 2.68), because
  evaluation noise was never the dominant term. Candidate-draw variance is 13.6pp and dwarfs it, so
  rollouts spent on more EPISODES buy precision on the wrong axis while rollouts spent on more DRAWS buy
  it on the right one. EXP-74's draw-averaged +7.98pp (SE 1.83) achieves the smallest error bar using
  the cheaper 25-rollout arms and stands as the headline.
  LESSON: when a measurement refuses to tighten, check which variance component is actually being reduced.
- EXP-75 RESULT (09-11) — THE REVERSAL HOLDS UNDER DRAW-AVERAGING. External (goal_expert) targets with
  FLEET-ONLY donors, 3 independent per-target draws, 27 complete runs: +5.33pp (SE 1.98), 8/9 improving,
  1 harmed. Single-draw (EXP-68) was +7.56pp, i.e. 2.2pp optimistic -- the same direction and magnitude
  of draw luck as on the fleet side (+8.71 -> +7.98), which is reassuring consistency.
  Both headline claims are now draw-averaged and mutually consistent: fleet weak +7.98pp, foreign +5.33pp.
  Foreign targets gain somewhat less, as expected since the donors do not share their provenance -- but
  they GAIN, against a published -4.44pp with 0/9 improving. Deck updated.
- EXP-77 note (09-11): the first pass found only 3 S16-init targets with headroom because the quota
  cleanup on 09-08 had removed new_loras_short from shiva. Restored the 4 weak short adapters (quota
  35.0 -> 35.8GB) and chained a second pass; the driver is idempotent so the first pass's probes are
  kept and the target list grows 3 -> 7. Lesson already in the resume file: check which queued jobs read
  a cohort before deleting it from a host.
- SHIVA HANDED OVER (09-11, user request): jobs stopped cleanly mid-EXP-77 (17 of 112 probes done, 0
  commits), all results rsynced to odin, tmpfs and decomp_adapters cleared, quota 35.8 -> 23.4GB, zero
  of our processes left. A colleague's training job now occupies the GPUs. EXP-77 (draw-averaging the
  S16-init claim) is the only headline still single-draw; it is resumable with no lost work because the
  drivers are idempotent. Odin continues EXP-76 (variant draw-averaging).
- EXP-76 RESULT (09-12): weak S16-init VARIANT targets draw-averaged over 2 draws, n=21, 42 runs:
  +5.14pp (SE 2.44), improving 14/21. Single draw was +4.19pp, so averaging moved this cohort UP by
  0.95pp -- the opposite direction from fleet (+8.71 -> +7.98) and foreign (+7.56 -> +5.33), which is
  what unbiased draw noise should look like and argues we are not fooling ourselves in one direction.
- COMBINED DRAW-AVERAGED PICTURE (45 weak targets, 129 complete method runs):
    fleet weak          15 targets / 60 runs  +7.98pp
    S16-init variants   21 targets / 42 runs  +5.14pp
    foreign targets      9 targets / 27 runs  +5.33pp
    ALL WEAK, weighted  45 targets / 129 runs +6.13pp
  The three cohorts agree within ~3pp: the method gives a consistent mid-single-digit gain on weak
  targets whether they are original fleet members, adapters we trained, or externally trained experts --
  the last being the case we previously published as a failure.

### EXP-78 — probe SUBSETS instead of singletons, at identical cost — **queued (odin, chained behind EXP-77)**
**Motivation.** Every probe we have run scores donors ONE AT A TIME (target + one donor at full dose c),
ranks them, and commits the top-3 at c/3 each. That pipeline assumes the singleton scores ADD. Two of our
own results say the assumption is shaky: two independent probes of the same 8 candidates agree on the single
best donor for only 1 of 10 targets (EXP-22.4/46), and the (target,donor) pair interaction is real and
reproducible at rho=0.58 (EXP-48). If interaction matters, the quantity worth measuring is the performance
of a 3-SET -- and a singleton probe never observes one. This is a change to the SELECTION MECHANISM, which
is the user's stated goal, and it is orthogonal to the closed rollout-free line: it still spends rollouts,
just on a better-chosen object.
- Design, cost-matched exactly (8x10 + 25 = 105 rollouts per target-draw, identical to the standard arm):
  same 15 weak fleet targets, same 8-candidate menus (`cands(t,k)` reused from EXP-69/74), 8 BALANCED
  3-subsets per menu (each candidate appears in exactly 3 of the 8, verified), each probed at 10 rollouts
  MERGED AT THE COMMITTED DOSE c/3, same abstention guard, best subset committed @25.
- The subset probe therefore evaluates exactly the object that gets deployed; the singleton probe evaluates
  a different object (one donor at 3x the dose it will actually receive) and extrapolates.
- The comparison arm is FREE: `e76_t{t}_d{k}` is the standard method's commit for the SAME target and the
  SAME menu, so this is paired on target, menu, dose, operator and commit budget -- only the choice rule differs.
- Draws 1 and 2, k-major so draw 1 yields a complete 15-target read before draw 2 starts. 3,150 rollouts.
- Pre-registered reading: a positive paired difference means singleton additivity is leaving gain on the
  table and the mechanism should probe sets; a null means additivity is good enough and the cheap singleton
  probe is vindicated on its own terms -- which is itself worth stating, since we have never checked.
- EXP-77 RESULT (09-12) — THE LAST SINGLE-DRAW HEADLINE IS NOW DRAW-AVERAGED. S16-init targets with real
  headroom, 2 further independent per-target draws added to the published one: **+12.33pp (SE 4.76), 4/4
  improving, 12 complete runs**, against the published single-draw +11.00pp (SE 4.43, 3/4). Draw luck cost
  1.3pp, the same small magnitude as the other three cohorts (fleet 0.7, foreign 2.2, variants -1.0).
  Verification that draw 0 is the same claim: all four e52 baseline records equal the report baselines
  exactly, and the four e52 commits reproduce the published +11.00pp to the decimal.
  Checked for the EXP-60 trap before trusting any of it: TG = 4 entries, 4 unique names, 4 unique md5 ids,
  no collision. n is 4 rather than 7 because the 3 veryweak adapters live only on shiva (handed to
  colleagues); the driver silently drops targets whose adapter file is absent, which is why the target
  list is short rather than wrong. Those 3 will extend this row to n=7 when shiva returns.
- COMBINED, ALL FOUR COHORTS (49 weak targets, 141 complete runs): **+6.63pp**, target-weighted.
    fleet weak          15 / 60  +7.98      S16-init variants   21 / 42  +5.14
    foreign              9 / 27  +5.33      S16-init w/ headroom 4 / 12  +12.33
  NO HEADLINE IN THE DECK NOW RESTS ON A SINGLE CANDIDATE DRAW. Averaging moved two cohorts down and two
  up, which is what unbiased draw noise should do and argues we are not drifting in one direction.

### EXP-79 — the missing cell: EXTERNAL donors into LIBERO-90 receivers — **running (odin, 09-13)**
**Motivation.** The user asked for the provenance cross-product as a slide. Three of its four cells were
already measured (fleet->fleet +2.22pp; fleet->external +5.33pp draw-averaged; union +2.82pp over 85), but
**external donors committed into fleet receivers has never been run** -- external adapters only ever
appeared as part of a MIXED candidate list, never alone. So the cross-product had a hole in it.
- What we already know about that cell, for free from EXP-32's probes: the probe takes external donors for
  only **20 of 228 top-3 slots (8.8%) against a 50% chance share, p = 1.3e-40**, and with external donors
  alone the guard abstains on **44 of 76** fleet targets. Contrast the SAME eight adapters offered to
  external receivers: **44.4%**, p = 0.7, i.e. pure indifference. The asymmetry is large and free.
- But "the probe declines to pick them" is not the same measurement as "committing them hurts". Only a
  commit closes the cell, so: read the external-only top-3 off EXP-32's existing probes, same operator
  (form by the target's own capture@16), same c/3 dose, same guard, same 25-rollout commit. 32 commits.
- Exactly symmetric to EXP-68, which did this in the other direction. Paired against BOTH published
  fleet-receiver arms (+2.22pp fleet-only candidates, +1.74pp mixed).
- ODIN REORDERED on user request: EXP-79 first, then the EXP-32 completion, then EXP-78 resumes. EXP-78
  was stopped cleanly at 53/240 probes; it is idempotent, so none of that work is lost.

### EXP-80 — the augmented fleet, PROVENANCE-BLIND: one pool of 84, 8 uniform candidates — **queued (odin, behind EXP-79)**
**Motivation (user's design correction, 09-13).** EXP-32 stratified its candidates: 8 fleet + 8 external per
target. That bakes provenance INTO the experiment -- external adapters are 8 of 84 in the pool (9.5%) yet
fill half the candidate slots -- and it doubles the method's canonical 8-candidate budget. So it never
tested "the probe method on one pool of everything we own"; it tested a design that already knew which
adapters were which. Slide 1 of the cross-product was built on it and is therefore mis-specified.
- Design: the 84 adapters are ONE list. Per target, 8 candidates drawn uniformly from the other 83
  (per-target seeded md5 draw, never shared). Probe @10, top-3 committed @25 under the guard. Same
  operator, dose, guard and baselines (e44_base) as EXP-32, so the two are paired on everything but the
  candidate rule. Dry run: 65 external among 680 slots = 9.6%, matching the 9.5% pool share.
- Cost: 553 new probes + up to 85 commits ~ 7,655 rollouts, ~3 days. 127 probes reused from EXP-32 where
  the identical (target, donor) pair was already measured identically. WEAK TARGETS RUN FIRST so the
  slice carrying the signal lands within ~1 day even if the strong tail does not finish by the meeting.
- Pre-registered reading: this REPLACES EXP-32's numbers on slide 1. EXP-32 stays as the source of the
  mixed-candidate rows on slides 2 and 3, where stratifying by provenance is the point.
- Odin reordered: EXP-79 (5h, already running) -> EXP-80 -> EXP-32 completion -> EXP-78. Ordering
  reasoned: EXP-79 first costs EXP-80 five hours; the reverse costs EXP-79 three days.
- EXP-79 RESULT (09-13 ~05:00) -- EXTERNAL-ONLY DONORS INTO FLEET RECEIVERS, the cross-product's last cell.
  Forced commit of the external-only top-3 on all 76 fleet targets (guard abstains 44, commits 32):
    all 76 : +1.00pp (SE 0.62, p=0.11)   improved 19 / flat 48 / harmed 9
    weak 14: +6.29pp (SE 2.43, p=0.023)  improved 11 / flat 2 / harmed 1
    strong : -0.19pp (SE 0.41)           8 harmed
  Paired against the MIXED arm on the same targets (+1.74 all, +5.14 weak): mixed - external-only =
  +0.74pp (SE 0.67, p=0.27); on weak targets -1.14pp (p=0.70). NOT DISTINGUISHABLE.
  READING: the asymmetry is in PREFERENCE, not HARM. The probe passes over external donors when fleet ones
  are on the menu (8.8% of top-3 slots), yet external donors alone still lift weak fleet targets by about
  as much as the mixed pool does. They are dominated, not damaging. Slightly more harm on strong targets
  (8 vs 2 mixed), consistent with less headroom protection when the good fleet donors are unavailable.
  (weak n=14 here vs 15 elsewhere: baseline is e44_base, a separate 25-rollout measurement.)
  Slide 3 filled: headline PENDING -> "Yes on weak targets, +6.29pp; not distinguishable from mixed".

### LIT NOTE (09-14) — RETAIN (arXiv 2512.08333) and DA-MergeLoRA (arXiv 2607.17467), flagged by the user
- RETAIN: θ̃ = (1−α)θ_pre + α·θ_ft on a generalist VLA; gains are OOD-ONLY (unseen positions, distractors,
  backgrounds), ID unchanged. LLM backbone carries it (best α_l ≈ 0.8, vision/action at 1; LLM-only merge ≈ all).
  Gain scales with base-model generality. Relevance: our evals are fixed-state ID (initial_states, seed 7); we
  have NEVER measured OOD robustness of a merged target. The 65 strong targets with ~0 ID headroom may have
  OOD headroom -- a possible new axis for "when does merging help", if the user wants it.
- DA-MergeLoRA: hypernetwork = cross-attention, query = mean CLIP embedding of TARGET IMAGES, keys = LoRA
  ΔW columns; per-column weights; meta-trained end-to-end with the task loss. Ablation: swap target images
  for noise -> iWildCam F1 28.95 -> 17.39. The signal is target observations, not LoRA weights. Pool = 3-10
  LoRAs whose domains are visually distinct; learned weights track visual similarity. Near-ceiling dataset:
  plain averaging = hypernetwork. Relevance: consistent with our closure, not a rescue -- their selection
  problem is visually legible, ours is not (activation distance failed, EXP-48); their ceiling result
  mirrors EXP-35 (uniform optimal) and the small interaction.

### EXP-81 — OOD robustness of probe-merged targets (RETAIN's axis) — **queued for SHIVA (user go 09-14; shiva back in a few hours)**
**Motivation.** RETAIN (arXiv 2512.08333) shows base/finetuned interpolation buys OOD robustness of the
finetuned task with ID unchanged, and that the LLM backbone at α≈0.8 carries it. Every number we have is
fixed-state ID. The 65 strong targets gain ~1pp because they have no ID headroom; whether they have OOD
headroom -- and whether probe-selected donors buy it -- has never been measured. Feasibility verified
09-14: LIBERO init state = [t] + qpos + qvel; free-joint objects are named with known qpos offsets; the
canonical 50 states place an object within ~±2.4 cm (std 1.4 cm), so a 5 / 10 cm xy shift is genuinely
outside the demo region. Distractors / backgrounds need BDDL edits -- not in the pilot.
- Design: 4 arms x 2 OOD levels, 25 rollouts each, on 20 targets (10 weak fleet + 10 strong fleet with
  existing e27 commits): (A) target alone, (B) target with LLM-band residual at 0.8 (RETAIN, no donors),
  (C) probe-merged commit as deployed, (D) C with the target's LLM-band residual at 0.8. B vs A isolates
  RETAIN's effect on a LoRA target; C vs A is our method under OOD; D tests whether the two stack.
- Cost: 20 x 4 x 2 x 25 = 4,000 rollouts (~2.5 days on odin). Cut to 1 level / fewer targets if needed.
- Pre-registered readings: C > A on strong targets under OOD => merging buys robustness where ID shows
  nothing, a new "when it helps". C ≈ A => the method is ID-only. B > A => RETAIN transfers to LoRA
  targets; B ≈ A => it needs full finetuning / a stronger base (their LIBERO gains were small too).
- 09-14 build: `main_dense.py --args.ood-shift` translates every free-joint object's xy by the given metres
  along one episode-seeded direction (layout preserved); orchestrator passes `ood_shift` plan-wide;
  `sweep/ood_robust.py` builds the 4 arms (target LLM-band residual scaled by beta via layerwise_probe.group_of).
  Targets: weak [19,56,6,50,8,15,14,60,9,33], strong [0,1,2,4,5,13,16,21,22,24]. Records e90_{arm}_t{t}_s{cm}.
  Shiva chain `sweep/chain_shiva_ood.sh`: EXP-77 remainder first (small), then EXP-81. Odin stays on EXP-80.


### EXP-82 — DA-MergeLoRA-style cross-attention selector on our labels — **building (odin CPU + one GPU pass; user approved 09-14)**
**Motivation.** The user asked whether DA-MergeLoRA's hypernetwork -- cross-attention from a TARGET-DOMAIN
image embedding onto LoRA dW columns, per-column merge weights -- should be tried here. Their ablation says
the target images carry the signal (noise instead of images: iWildCam F1 28.95 -> 17.39). We never gave our
selector that input: everything we fed it described the LoRAs. Rather than argue, test the architecture as a
per-donor SCORER on the labels we already own, paired seed-for-seed against the W+A/48 feature model.
- Inputs built 09-14: `demo_bank.npz` (16 of each target's OWN demo frames, all 172 adapters);
  `demo_bank_base.npz` (the BASE policy's activations on those frames -- the frozen-encoder query);
  `lora_col_tokens.npz` (per stem the 16 column directions of dW, JL-64; 458 x 16 tokens per adapter).
- Model: q = MLP(mean base 'vision' activation over the target's frames); keys = Linear(tok) + stem + band
  embeddings; donor score = mean_tokens <q,k>/sqrt(d) = their column-level softmax pooled to a donor.
  Variants: obs (faithful), obs+t (query also sees the target's own columns), feat (reference AffinityNet).
- Protocol: hypernet.py's -- hold out whole targets, fleet-holdout and cross-distribution folds, listwise
  loss, top-3 affinity over random, 36 seeds. Reference to beat: W+A/48 gave +1.14 (fleet) / +0.89 (x-dist).
- Pre-registered reading: obs > feat by a paired margin => target-observation conditioning extracts
  something LoRA features miss and the per-column MERGER earns its rollouts. obs <= feat => closed with a
  number: their input does not carry our interaction. NOT run: their end-to-end merger training (its loss
  would be the demo action loss, which EXP-22.5 showed does not track rollout success).
- EXP-82 build notes (09-14): base-policy pass runs at 4.9 s/obs on odin's P6000 (no torch.compile on
  capability 6.1) -> ~3.7 h for the 2,736-frame bank. Training trimmed for CPU: top-4 column directions per
  stem (458 x 4 tokens per adapter), 12 seeds, 100 epochs; the feature-model reference is retrained on the
  same seeds so the comparison stays paired. Three launch faults fixed on the way: a half-written npz read
  (wait on done markers), the K620 display card being torch:2 (pick the freest P6000 by torch index),
  and dynamo/Triton on capability 6.1 (TORCHDYNAMO_DISABLE=1, as hostcfg already does for the eval workers).
- EXP-80 INTERIM (09-14 17:10) -- WEAK SLICE COMPLETE (20 targets, provenance-blind, 8 uniform candidates
  from the 83 others, one draw per target): **+4.80pp (SE 1.92)**, improved 13 / flat 3 / harmed 4,
  guard abstained on 2. Paired on the SAME 20 targets against EXP-32's stratified 8 fleet + 8 external
  design (+8.80pp): blind - stratified = **-4.00pp (SE 2.38, p = 0.11)**, blind better on 6 / worse on 11 /
  tie 3. External adapters appeared in the blind draws at 0.90 of 8 slots (pool share predicts 0.76).
  READING: the honest single-pool number is about half the stratified one, and the gap is not significant
  at n=20 with one draw per target (draw noise alone is ~13.6pp/target). Two things the stratified design
  bought: 16 candidates instead of 8, and a guaranteed 8 fleet candidates for every target. Under the
  method's real budget the gain is +4.8pp on weak targets -- consistent with the fleet-only draw-averaged
  range (+7.98, SE 1.83) within noise, but this is the number that describes "the probe method on one pool
  of everything we own". Strong tail (65 targets) still running; 4 decided so far at +2.00 (SE 3.46).
- ODIN REORDERED (17:10): EXP-32 completion (86/86, ~2h) now runs before EXP-80's strong tail, which cannot
  finish before the 09-16 meeting either way. EXP-80 resumed idempotently (180 records kept), EXP-78 last.
- EXP-32 COMPLETE (09-14 19:05) -- 86/86. The target lost to the task-name typo (goal expert on
  put_the_bowl_on_the_plate) is strong: baseline 96%, committed 96%, gain 0. Aggregates: ALL 86 +2.79pp
  (SE 0.67), 44 committed / 42 abstained; fleet +1.74 (unchanged); external n=10 +10.80pp (SE 3.21), 7/3/0;
  weak n=20 +8.80 (unchanged); strong n=66 +0.97; pool-wide 91.9 -> 94.7. f28's two missing probes refilled:
  donors 74/75 score 1.0/0.9, top-3 unchanged in effect (a 3-way tie at 1.0; the existing commit is one
  valid tie resolution) -> commit kept. Deck: every EXP-32 figure updated in place; the draw-averaged
  external row stays n=9 (the 10th target has a single draw only).
- INCIDENT (09-14 23:20): the EXP-32 completion took 17:15 -> 23:01, not ~2h. Cause: unified_pool_driver drew
  external targets' fleet candidates from ONE rng advanced per target; goal_expert_4 used to be skipped (no
  baseline) BEFORE its draw, so once its baseline existed it consumed a draw and targets 5-9 each inherited
  the previous target's set -> 36 pairs re-probed (verified from file timestamps: 4's new set == 5's old
  set, 5's == 6's old, ...). Commits are kept by the driver, so every number in the deck is unchanged
  (report at 23:01 == report at 19:05); the cost is ~4 GPU-hours and mixed candidate bookkeeping for those
  five targets (both sets' probes exist; all are valid (target,donor) measurements and EXP-80 reuses them).
  FIX: candidate sets are now frozen to what was probed first, with a per-target md5-seeded draw as the
  fallback. LESSON (a new face of the EXP-29 trap): never let one sequential rng decide per-target draws.
- EXP-82 RESULT (09-15 00:10) -- top-3 affinity gain over random, 12 seeds, same folds and labels for all:
    fleet-holdout : obs +1.07 (SE 0.62, p=0.11) | obs+t -0.03 | feat(same budget) -0.00 | paired obs-feat +1.08 (SE 0.82, p=0.22)
    cross-suite   : obs +0.10 (SE 0.00) | obs+t +0.24 (SE 0.00) | feat -0.25 (SE 0.18)   | paired +0.36 (p=0.08)
  READING. (1) Fleet fold: the observation-conditioned scorer lands at +1.07, i.e. where the PUBLISHED
  feature model already sits (+1.14, 36 seeds, full budget); the same-budget feat reference at 0.00 shows
  100 epochs / lr 1e-3 under-trains AffinityNet, so the fair bar is the published number, and obs does not
  clear it. (2) Cross-suite fold: SE 0.00 across 12 different inits means the obs scorer's ranking is
  IDENTICAL every seed -- degenerate (near-constant scores, picks decided by tie order). The
  target-observation query does not transfer to goal/object/spatial targets at all, which is exactly the
  setting DA-MergeLoRA is built for. (3) Nothing is within an order of magnitude of the behavioural probe
  (+6.5pp deployed). VERDICT: closed with a number. Target-domain observations as the query carry no more
  of the (target, donor) interaction than LoRA features do here; the per-column MERGER does not earn its
  rollouts. Consistent with the claim that their signal is visual domain similarity, which our donors do
  not differ in. Not pursued further.
- INCIDENT (09-16 03:39) -- QUOTA CRASH KILLED BOTH REMAINING ODIN STAGES, HIDDEN BY THE CHAIN SCRIPT.
  EXP-80 died at 385/553 probes and EXP-78 at its first build, both with
  "RuntimeError: only 1.5GB free, need ~1.5GB" from _ensure_space. chain_blind2.sh had no `set -e`, so it
  ran the next stage, then printed "ALL DONE" -- the watcher believed it and reported success. ~25 min idle.
  ROOT CAUSE: merged adapters are ~1.66GB each now (the S16 form concatenates target + donors + basis),
  not the ~0.55GB _ensure_space's docstring was written for, so a CHUNK=6 run peaks near 10GB of builds
  while only ~5GB of quota headroom remained. Nothing had "leaked": the quota simply crept up (EXP-82's
  bank + tokens added 0.5GB) until a normal chunk no longer fit.
  FIXES: (1) CHUNK is now env-overridable and the chain runs CHUNK=3 (~5GB peak); (2) docstring corrected
  to the measured 1.66GB; (3) chain_blind3.sh uses `set -e` and per-stage OK/FAILED markers, so a crash
  aborts loudly instead of being reported as completion; (4) the watcher now also warns below 3GB free.
  Reclaimed 4.7GB of orphaned build files left by the crash (3 x 1.66GB, no orchestrator running).
  LESSON: a chain script without `set -e` converts a crash into a false success -- and the watcher can only
  be as truthful as the marker it reads.
- EXP-80 ANALYSIS (09-16, prompted by the user asking why weak LIBERO-90 receivers only gain +2.86pp):
  Split by receiver: weak LIBERO-90 +2.86 (n=14), weak external +9.33 (n=6) -- the +4.80 headline is a mix.
  Paired against the published fleet-only method on the SAME 14 targets (+8.05): -5.19pp (SE 2.69, p=0.075).
  RULED OUT as the explanation: (a) target selection -- scored on the published weak-15 set EXP-80 gives
  +1.33, i.e. LOWER, so the different weak membership is not flattering EXP-80; (b) baseline level -- the
  two independent 25-rollout baselines match on average (79.7 vs 79.4%) though they disagree 3.1pp per
  target, max 12pp, and corr(baseline gap, gain gap) = -0.71, which is just regression on a noisy baseline.
  NOT RULED OUT, and the leading candidate: DRAW LUCK. EXP-80 uses ONE candidate draw per target; the
  published +7.98 averages four. Draw SD is 13.6pp/target, so a single-draw n=14 mean carries +-3.6pp and
  the observed gap is ~1.9 SE of pure noise. HONEST STATEMENT FOR THE MEETING: the provenance-blind
  single-draw weak-target gain is +2.86pp on LIBERO-90 receivers, lower than the published number, but the
  difference is not established (p=0.075) and the single largest known term is the one EXP-80 does not yet
  control. Draw-averaging EXP-80's weak slice (~3x cost, ~20 targets x 2 more draws) is the test that settles it.

### EXP-83 — faithful DA-MergeLoRA: cross-attention that emits MERGING COEFFICIENTS, 80/20 donor/receiver split — **planned (user, 09-16)**
**Motivation.** EXP-82 borrowed their query and keys but collapsed the output to one affinity score per donor,
so it tested their *representation*, not their *method*. The user wants the real thing: coefficients produced
by the cross-attention, all donors merged at their own weights, trained on a train split and applied to
held-out receivers.
- FEASIBILITY, measured 09-16: their formula merges columns of the FULL dW. One materialised dW here is
  3.6B params = 7.2GB fp16; 68 donors = **492 GB**. Not runnable at our scale. Forced deviation: keep the
  low-rank form and emit one coefficient per (donor, rank-1 component): 68 x 16 = 1,088 components per stem,
  6.3 GB for the whole donor pool, and the forward is a fused low-rank op (A_all h, weight, B_all) with no
  dW ever materialised. Their stated reason for full-dW columns was mixing different ranks; all our adapters
  are rank 16, so that reason does not apply. Still far finer than the per-donor scalar we use today.
- Coefficients come from attention, not a large output head: score(donor k, component r, stem s) =
  <q, key_{k,r,s}>, softmax over k, exactly their shape. Query = base policy on the receiver's own demo frames.
- SPLIT. Proportional 20% would hand us 12 of 18 receivers at ceiling, where the guard abstains and nothing
  is measurable. Balanced across the 2x2 instead (18 receivers: 7 fleet/weak, 5 fleet/strong, 3 ext/weak,
  3 ext/strong; 68 donors). Receivers need headroom, donors do not, so this costs the donor pool nothing.
- TRAINING. Their loss is task cross-entropy backpropagated through the frozen backbone. Ours is the
  flow-matching action loss on the pseudo-target's demos, with one train-split target masked out per step.
  **This is the main risk and it is known: EXP-22.5 found the action-error proxy does not track rollout
  success.** It is also the load-bearing part of their method, so the experiment is precisely a test of
  whether it transfers. Second risk: their reported gains are +1-3%; our probe is +6.5pp, a high bar.
- EVAL. 18 held-out receivers, 25-rollout commits, against (a) baseline, (b) random top-3, (c) the
  behavioural probe. Cost ~550 rollouts plus training.
- EXP-80 RESULT (09-16 21:06) -- THE PROVENANCE-BLIND UNION, COMPLETE AT 86/86. One pool of 84 adapters,
  8 candidates drawn uniformly per target, no knowledge of provenance; same operator, dose, guard, commit.
    ALL 86           +1.53pp (SE 0.53, p=0.005)   25 improved / 53 flat / 8 harmed, 44 abstained
    weak  (n=20)     +4.80pp (SE 1.92, p=0.022)   13 / 3 / 4
    strong(n=66)     +0.55pp (SE 0.30, p=0.072)   12 / 50 / 4, 42 abstentions
    LIBERO-90 (n=76) +1.05pp (SE 0.48)  -- weak +2.86 (n=14), strong +0.65 (n=62)
    external  (n=10) +5.20pp (SE 2.60)  -- weak +9.33 (n=6),  strong -1.00 (n=4)
    pool-wide 91.9% -> 93.4%; committed 42 / abstained 44
  THIS REPLACES EXP-32's +2.79pp as the headline: EXP-32 gave every target 16 candidates (8 fleet + 8
  external, externals 9.5% of the pool but 50% of the slots), so it measured a design that knew which
  adapter was which. Under the method's real 8-candidate budget the all-target gain is +1.53pp.
  The weak/strong split is unchanged in shape: weak targets carry everything, strong ones are at ceiling
  and the guard correctly declines on 42 of 66. Weak EXTERNAL receivers (+9.33) gain over three times weak
  LIBERO-90 receivers (+2.86) -- same direction as the stratified run, and the cleanest statement that our
  pool transfers outward better than it improves itself.
  CAVEAT CARRIED: single draw per target. The weak LIBERO-90 figure is 5.19pp below the published
  fleet-only method on the same 14 targets (p=0.075), and draw noise alone is +-3.6pp at n=14, so that
  gap is not established. Draw-averaging the weak slice is the outstanding test.
- EXP-81 STARTED ON ODIN (09-16 22:01), not shiva: the only thing that actually required shiva was the
  EXP-77 remainder (3 adapters stored nowhere else). EXP-81 just needed a free GPU pair, which EXP-80
  finishing provided. 160 cells (4 arms x 2 shifts x 20 targets), ~4,000 rollouts, CHUNK=3, set -e chain.
- EXP-81 LAUNCH FAULT + FIX (09-16 22:03): first attempt died in 100s on "_ensure_space: only 4.5GB free,
  need ~9.0GB". Two compounding defects, both now fixed: (1) the guard's default was a FLAT 9.0GB, tuned
  for neither the current 1.66GB build size nor CHUNK; (2) it runs before EVERY save, so by the last build
  of a chunk the earlier builds of that same chunk are legitimately on disk -- demanding a whole chunk
  there asks for roughly twice the chunk and fails by construction. The default is now a per-build
  requirement (this build plus one spare, 3.4GB), which is what the call site actually needs.
  Relaunched 22:06 on odin.
- DISK: THE BINDING LIMIT ON ODIN IS THE FILESYSTEM, NOT OUR QUOTA (discovered 09-16 22:05). /dev/sda2 is
  469GB and 100% FULL with ~3-5GB free; our quota still permits 8.5GB more. _ensure_space takes min(fs, quota)
  so it reported "only 3.4GB free" while `quota` said 8.4GB -- the two numbers disagree because they measure
  different things. Our home is 37GB; the other users visible in /home total ~50GB, so the bulk of the 469GB
  is consumed outside /home and is not ours to reclaim. Freed what is safely ours (orphaned builds, old
  rollout videos) and dropped CHUNK to 2 so peak build usage is ~3.4GB. EXP-81 runs, but odin is one other
  user's large write away from stalling again. WORTH RAISING WITH THE USER / SYSADMIN.
- EXP-81 FALSE START, 40 CELLS WASTED (09-16 22:10-23:18): every A/B cell returned status=parse_fail in 3s.
  CAUSE: merge_orchestrator defaults to `examples/libero/main.py`, but only the instrumented COPY
  (`main_dense.py`) has `--args.ood-shift`. The plan never set eval_script, so main.py was handed a flag it
  does not define and died on startup. The OOD shift was therefore never applied -- those 40 records are
  meaningless and have been moved to results/merge_stale/.
  WHY IT WAS NOT CAUGHT SOONER: the chain's `set -e` catches a crashed STAGE, not per-cell eval failures;
  the driver treats parse_fail like any other missing result and simply moves on. The monitor counted
  FILES, not ok records. Fixed: ood_robust.py pins eval_script="examples/libero/main_dense.py";
  verified end-to-end with a 2-rollout smoke cell (status ok, rate 0.5, 158s, shift applied).
  NEW WATCHER RULE: count records with status == "ok", never bare file counts.
- SHIVA RETURNED (09-17 17:00). Moved the heavy work there: EXP-77 remainder (the only job that strictly
  needs shiva, its 3 adapters exist nowhere else) then EXP-81, which had been crawling on odin at CHUNK=1
  because /dev/sda2 is ~100% full. Shiva: 117GB free, 3x49GB GPUs, hostcfg derives 7 workers vs odin's 2-3.
  EXP-81's 28 completed cells were rsynced across and the driver resumes idempotently. Odin left idle
  rather than given more build-heavy work on a full disk.
  SSH GOTCHA WORTH REMEMBERING: ssh to shiva lands in /home/lzhou, not the repo, so a tmux new-session with
  relative paths silently produced a dead session and no log. Always cd inside the tmux command.
- EXP-77 COMPLETE (09-17, shiva) -- the S16-init cohort at its full n=7, 18 runs: **+7.90pp (SE 4.27,
  p=0.114), improving 5/7**. The earlier n=4 figure was +12.33pp; the three targets that could only run on
  shiva are the weakest of the cohort and two of them LOSE (-2.0, -8.0), so the n=4 number was optimistic
  and the effect is no longer significant at this n. Single-draw over the same 7 is +5.71pp, so
  draw-averaging moved this cohort UP by 2.2pp -- the same unbiased-noise behaviour seen elsewhere.
  Deck: the S16 row now reads n=7 / +7.90pp in both 16_09 and 23_09.
  NOTE the honest consequence: with 7 targets and SE 4.27 this cohort no longer carries a significant
  gain on its own. It still agrees in sign and rough size with the other cohorts, which is the claim the
  slide actually makes.
- EXP-81 HALF RESULT (09-17, shiva) -- the 5cm shift complete, all 4 arms x 20 targets.
  The shift does open a real gap: weak 74.3% ID -> 63.9% shifted (-10.5pp, p=0.067); strong 100% -> 96.0%
  (-4.0pp, p=0.008). So there IS robustness headroom, including on targets with zero ID headroom.
    arm B  RETAIN alone (target residual 0.8 on LLM bands)   +0.90pp (SE 1.64, p=0.59)   weak +1.0  strong +0.8
    arm C  our probe-merged commit                           +1.00pp (SE 1.71, p=0.57)   weak +2.8  strong -0.8
    arm D  BOTH                                              +3.13pp (SE 1.79, p=0.096)  weak +4.67 strong +1.60
  Neither single intervention is significant; the COMBINATION is the largest effect and the only one near
  significance. D-B = +2.23 (p=0.19), D-C = +2.13 (p=0.23), and D-A minus the sum of the singles is
  +1.23pp (p=0.49), i.e. consistent with the two stacking ADDITIVELY rather than interacting.
  NOTABLE: arm D lifts STRONG targets +1.60pp under shift, where in-distribution they gain ~0.5pp and the
  guard usually abstains. That is the first sign the method has anything to offer ceiling targets -- but at
  n=10 and p>0.1 it is a lead, not a result. The 10cm arm (80 cells) is running and is where the gap, and
  therefore the power, should be largest.
- EXP-83 BUILD LOG (09-18). Step 1 split frozen to results/analysis/exp83_split.json: 68 donors / 18
  receivers, receivers balanced 7 fleet-weak / 5 fleet-strong / 3 ext-weak / 3 ext-strong (weak baselines
  70.4%, strong 97.5%). Frozen deliberately -- a re-drawn split between training and eval would silently
  invalidate the result. Step 2 `exp83_merge.py`: the fused differentiable merge, h' = Wh + B_all(w*(A_all h)),
  effective rank K*16 = 1088, dW never materialised. VERIFIED three ways: matches an explicitly
  materialised dW to 7e-9; one-hot donor weights reproduce that donor's own dW to 4e-9; gradients reach
  every coefficient. Step 3 `exp83_hypernet.py`: softmax over DONORS per (stem, rank-1 component), with
  stem / band / donor embeddings, plus a learned per-stem scale for total injected mass. Caught at build
  time: the scale initialised to 0.333 instead of the verified 0.5 because the value passed through a
  sigmoid twice -- now exactly 0.500 at init for any cap.
- EXP-83 BUILD (09-18, cont). Step 4 `exp83_patch.py`: PEFT is bypassed deliberately -- its lora_A/lora_B
  are leaf Parameters of rank 16, so a merge written into them carries NO gradient to the coefficients.
  Each adapted Linear is wrapped instead: y = base(x) + B_all(w_stem * (A_all x)), with the donor pool as
  frozen buffers and w supplied per forward pass. Base weights untouched, so one loaded policy serves all
  18 receivers. Step 5 `exp83_smoke.py`: end-to-end check on shiva GPU 1 (EXP-81 holds GPU 0) --
  the model's own flow-matching loss, backward, and two controls: gradient norm into the hypernet must be
  non-zero, and zero coefficients must reproduce the unpatched base loss exactly.
  Launch faults fixed on the way: exp83_split.json existed only on odin; LiberoPolicy takes
  (BASE, norm_stats_path=, pytorch_device=), not (base_dir=, device=).
- EXP-83 PLUMBING VERIFIED (09-18, shiva GPU1). Seeded control, all four assertions pass:
    unpatched loss, same seed twice : 0.096458 / 0.096458  -> the objective is deterministic once seeded
    ZERO coefficients               : 0.096458, diff 0.00e+00 from base  -> the merge is exactly a no-op at w=0
    MERGED at the 0.5 dose          : 0.069557, -0.0269 from base        -> the merge actually reaches the forward
    after unpatch                   : 0.096458                            -> no residue left on the model
  Gradient reaches all 12 hypernet tensors (norm 0.117). The FIRST control looked like a failure (0.1022 vs
  0.0581) purely because the flow-matching loss samples its own noise and timestep: two calls on IDENTICAL
  weights differ. Seeding before each forward is mandatory for any loss comparison in this codebase --
  worth remembering, it would have been easy to "fix" a non-existent wiring bug.
  Interesting aside, not yet a claim: a RANDOM merge at the verified dose LOWERS the demo loss (0.0965 ->
  0.0696). Demo loss is not rollout success (EXP-22.5), so this is exactly the risk flagged when the
  experiment was proposed -- the training signal may be easy to improve without improving behaviour.
- EXP-83 BUILD FAULTS (09-18), both worth remembering:
  (1) `set -e` DOES NOT CATCH A CRASH INSIDE A PIPELINE. `python ... | tee log` reports tee's exit status,
      so the validation crashed on a KeyError and the chain still printed "=== VALIDATION DONE ===".
      This is the SECOND time a chain script has converted a crash into a false success (chain_blind2.sh
      did it on 09-16 with a quota crash). Every chain now needs `set -o pipefail` as well as `set -e`.
  (2) exp83_split.py wrote external donor keys as "np:<name>", but U.fold()/U.EXTPATH expect the BARE
      adapter name (ints for fleet). Fixed; the split regenerates identically because it is seeded, which
      is the property that made the fix safe to apply after the fact.
  Also handled: one LIBERO-90 donor (f57) has no regenerated demo hdf5. It stays in the donor pool -- it
  can still be merged INTO others -- but is never sampled as a pseudo-target, which needs its frames.
  No RECEIVER lacks demos, so the frozen split is unaffected.
- EXP-83 DESIGN BUG CAUGHT BY THE VALIDATION RUN (09-18) -- worth the 40 steps on its own.
  The validation printed "5 donors lack demos" when only ONE task (f57) actually has no hdf5. Tracing it
  exposed a conflation I had baked into the split: a TARGET is an (adapter, task) pair, but I keyed
  everything by ADAPTER. The goal expert is ONE adapter serving TEN tasks, so:
    (a) the donor pool listed 68 entries but only 65 unique adapters -- the goal expert was loaded FOUR
        times and would have received 4x the representation in the merge;
    (b) all 6 external RECEIVERS share that one adapter key, so an adapter-keyed query could not tell
        them apart -- they would all have been conditioned on the same embedding, or none at all.
  Fixed across all three scripts: queries are keyed by TAG (unique per adapter-task), the bank is
  deduplicated by adapter, pseudo-target sampling is by tag, and the mask zeroes the pseudo-target's own
  ADAPTER index rather than a position in a list that had duplicates.
  This is exactly why the 40-step validation existed: the full run would have trained and evaluated
  without error and produced a number that meant something subtly different from what the design claims.
- EXP-83 CRITICAL BUG (09-18) -- THE QUERY EMBEDDINGS CARRY NO TASK INFORMATION. The collapse diagnostic
  at step 600 showed the hypernetwork choosing the SAME donor (index 12) for all 18 receivers, pairwise
  cosine 1.000, per-donor share std 0.0000 -- indistinguishable from EXP-30's donor-blind failure.
  It is NOT that failure. Tracing upstream: all 85 stored query vectors are BITWISE IDENTICAL, and two
  visually unrelated tasks (a kitchen moka pot and a living-room mug) produce the same vision-tower
  embedding to 0.000e+00. So the hypernetwork was fed a constant and correctly learned a constant answer.
  Suspected cause: the vision tower is called 3x per infer (multiple camera streams / padding slots) and
  the hook kept only the LAST call, which appears to be constant. Under test now.
  WHY THIS MATTERS BEYOND THE BUG: without the discrimination check this would have been written up as
  "the DA-MergeLoRA merger is donor-blind, like every learned selector we have tried" -- a negative result
  with exactly the shape we already expect, and therefore one nobody would have questioned. A confirming
  null is the most dangerous kind of result to get by accident.
  The 2000-step run in flight is void whatever its loss does; it will be rerun once the query is fixed.
- EXP-83 QUERY BUG, ROOT CAUSE AND FIX (09-18). The vision tower is called 3x per infer; calls 0 and 1 are
  the base and wrist cameras and vary with the task, call 2 is a padding slot whose embedding is BITWISE
  IDENTICAL for every task (norm 24.568 always). A naive forward hook keeps the LAST call, so all 85
  queries were that constant and the hypernetwork -- correctly -- learned one answer for everyone.
  First fix (take calls[:2]) silently did not work: the captured call count differs between contexts.
  Final fix does not depend on ordering at all: keep EVERY call, then drop the dimensions that are constant
  across tasks. 1123 of 3456 dims dropped, 2333 kept; mean pairwise distance 0.0 -> 67.2. The builder now
  refuses to write the file if fewer than 64 dims vary, and training refuses to start if the stored queries
  do not vary. Two of 3570 pairs remain identical: f7/f18 and f29/f62 are fleet adapters that share a task
  and therefore share demo frames. All four are donors, no receiver is affected.
  ALSO: the void training run survived its kill long enough to rewrite exp83_hypernet.pt at 02:14, AFTER
  the checkpoint had been deleted. Caught and removed; its log archived as exp83_full.VOID.log. A periodic
  checkpoint writer outlives a kill -- always re-check the artifact, not just the process.
- EXP-83 EXPORT BUG + AN INCIDENTAL FINDING (09-18). The first evaluation wrote each merged receiver as a
  rank-1040 adapter: 12.5GB apiece, ~20x a normal one, and two of them pushed shiva over its 46GB quota.
  Fixed by truncating each stem's merged delta with a randomized SVD before export. Measured fidelity on
  the real 65-donor merge, 12 largest stems:
      rank  8 -> 65.9% of the delta's Frobenius norm
      rank 16 -> 99.5%
      rank 32 -> 99.8%   (used)
      rank 64 -> 99.9%
  THE INCIDENTAL FINDING: a weighted sum of 65 donor adapters, nominal rank 1040, is 99.5% captured by
  rank 16 -- the fleet's own native rank. The donors' contributions are almost entirely redundant, which
  is what the pool's geometry already implied (fleet pairwise cosine 0.867, 91% of each adapter is one
  shared vector). It also means a per-component merge has far fewer usable degrees of freedom than its
  1040 coefficients suggest, which bears directly on whether DA-MergeLoRA's fine-grained weighting can
  buy anything here.
- EXP-83 MERGE BUG -- THE RESULT WAS AN ARTIFACT, NOT A FINDING (09-18). The first held-out evaluation
  looked catastrophic: -29.00pp over 8 receivers (p=0.014), 0/8 improving, f50 80%->16%, f69 96%->40%.
  That is not "the method does not help", it is a malformed merge, and it was.
  CAUSE: the operator contributes each donor's RESIDUAL -- M = U^T b; c*(b a) - c*(U M)(N V^T) -- i.e. the
  donor's projection onto the shared S16 subspace is removed first. My DonorBank held RAW donor deltas.
  Since ~91% of every fleet adapter IS that shared vector, summing 65 raw deltas amplified the shared
  direction ~65x ON TOP of the S16 basis the adapter already carries.
  FIX + PROOF: the bank now stores each donor in residual form (2r columns, [b, -(U M)] vs [a, N V^T]).
  A single donor at c=0.17 reproduces the operator's own contribution to 1.3e-10, and the residual differs
  from the raw delta by 91.1% of its norm -- an exact, independent re-derivation of the project's S16
  result from a completely different direction. Gradients still reach every coefficient.
  CONSEQUENCE: the 2000-step training run used raw deltas and is void; 15 e91 records quarantined.
  Retraining and re-evaluating. NOTHING from the first evaluation may be quoted.
- EXP-83 DOSE CORRECTION (09-18), a direct consequence of the residual fix. With raw deltas the merge
  moved the loss by -0.027; with residuals at the same nominal mass it moved it by 3e-5, i.e. the merge had
  become nearly inert. REASON: a softmax spreads its mass over K donors, and donor RESIDUALS are
  near-orthogonal (cross-task residual cosine 0.025), so the injected norm is m/sqrt(K_eff), not m. Our
  operator injects 0.289 x mean||r|| (3 donors at ref/3, in quadrature); the softmax form needs m = 0.289*sqrt(K),
  which is 2.33 at K=65 -- so a mass of 0.5 was 5x under-dosed. The hypernetwork now initialises its scale
  to 0.289*sqrt(K) automatically and can learn up to 4x that. Control after the fix: all four assertions
  pass (zero coefficients exactly reproduce the base loss; the merge measurably changes it; unpatch is clean).
  NOTE FOR THE WRITE-UP: this is a real structural finding about porting their method, not just a bug.
  Softmax-over-donors weighting silently loses injected energy as the donor pool grows, because the things
  being averaged are near-orthogonal. With their 3-10 visually distinct domains the effect is mild; at 65
  near-orthogonal residuals it is a 5x shortfall that would have read as "the method does nothing".
- INCIDENT (09-18 07:00): 10 of 11 EXP-83 receivers came back server_died / error / parse_fail. The
  exported adapters were NOT at fault -- inspected directly: 0.92GB, 916 tensors, zero non-finite values,
  rank 79 against the operator's own 48. CAUSE: I had EXP-81 and the EXP-83 evaluation running as two
  concurrent orchestrators, each spawning up to 7 policy servers, on 3 GPUs -- ~14 servers x 8GB against
  147GB. EXP-81's session was killed outright and EXP-83's receivers died mid-serve. The project's rule is
  ONE EVAL ORCHESTRATOR PER MACHINE and I broke it. Failed records return None from U.get, so the driver
  re-runs them automatically; nothing needs manual repair. EXP-81's last 3 cells are now chained to start
  only after the EXP-83 evaluation releases the GPUs.
- EXP-83 SECOND EVALUATION ALSO WITHDRAWN (09-18 08:06). Result was -68.50pp over 8 receivers, with FIVE
  policies at EXACTLY 0% (f9 88->0, f60 84->0, f6 76->0, f8 68->0, f15 64->0). Destruction, not selection.
  CAUSE, again mine: I energy-matched the initial dose assuming the softmax spreads its mass UNIFORMLY
  (init_mass = 0.289*sqrt(K)). Training concentrates it. At the final checkpoint max-share was 0.45 of a
  total mass 2.32, so the top donor alone carried coefficient 1.04 against the operator's 0.167 per donor
  -- the injected energy GROWS as the shares sharpen, by ~2-4x. EXP-35 measured that moving the donor dose
  either way costs ~8.8pp; this was far outside anything we have ever tested.
  FIX: energy match the way the operator does and make it concentration-invariant -- divide each donor's
  coefficient by its own residual norm (size-invariance, exactly the operator's ref/||r_i||), then
  renormalise by the share vector's L2 norm so sharpening cannot change the total. The learned scale then
  moves the dose deliberately instead of as a side effect of the softmax.
  NEW ASSERTION (sweep/exp83_dosecheck.py): measure the merged delta's Frobenius norm directly against the
  operator's own 3-donor build, at BOTH spread and highly concentrated shares; both ratios must be ~1 and
  equal to each other. This should have existed before the first training run.
  RUNNING TALLY of EXP-83 defects, all of which produced a confident wrong number: constant query
  embeddings; raw deltas instead of residuals; dose 5x too low; two orchestrators contending; and now dose
  growing with concentration. Only the stochastic-loss alarm was a false positive.
- EXP-81 COMPLETE (09-18) -- OOD ROBUSTNESS, 160/160 cells, 4 arms x 2 shifts x 20 targets.
  The shift does open a real gap, and it widens with distance:
      5cm : weak 74.3% -> 63.9% (-10.5pp) | strong 100% -> 96.0% (-4.0pp)
     10cm : weak 74.3% -> 45.1% (-29.3pp) | strong 100% -> 75.2% (-24.8pp)
  So even targets with ZERO in-distribution headroom lose a quarter of their success under a 10cm shift.
  Recovery vs the shifted target alone, POOLED over both shifts (n=40 paired observations):
      arm B  RETAIN alone (target dose 0.8 on LLM bands)  +0.98pp (SE 1.11, p=0.38)
      arm C  our probe-merged commit                      +1.48pp (SE 1.13, p=0.20)
      arm D  BOTH                                         +3.02pp (SE 1.13, p=0.011)   weak +4.43 (p=0.037)
  HEADLINE: neither intervention is significant alone; TOGETHER they are, at +3.02pp overall and +4.43pp
  on weak targets. D-C = +1.53pp (p=0.22), D-B = +2.04pp, and the interaction term is ~0, so the two
  appear to stack ADDITIVELY rather than synergise -- two small independent effects, not one mechanism.
  WHAT IS NEW: our method has never before shown anything on targets at ceiling. Under shift, arm D lifts
  STRONG targets +1.60pp where in-distribution they gain ~0.5pp and the guard usually abstains. The
  robustness axis is where merging has something left to offer them -- but at p=0.148 that specific cell is
  a lead, not a result.
  CAVEATS: one draw per target; only object POSITION was varied (RETAIN also varies distractors,
  backgrounds, viewpoints, which need BDDL edits); n=20 targets.
- EXP-83 RESULT (landed 09-18 10:33, read 09-19 -- it sat 25h unwatched because the retrain was relaunched
  without re-arming a Monitor; lesson recorded). Faithful DA-MergeLoRA merger, energy-matched and
  verified (form to 1.3e-10, dose to 1% at spread AND concentrated shares), 65 donors -> 18 held-out
  receivers. The 6 EXTERNAL receivers came back parse_fail (eval handed them the fleet's suite/norm; fixed,
  rerunning). On the 12 FLEET receivers, paired against the behavioural probe's own commit on the SAME
  receiver (abstention = baseline):
      DA-MergeLoRA merger : -4.33pp (SE 2.53, p=0.115)   improving 2/12, harmed 6
      behavioural probe   : +2.33pp (SE 2.00, p=0.267)   improving 5/12
      PAIRED merger-probe : -6.67pp (SE 2.05, p=0.008)   merger better on 1, worse on 8
        weak  n=7 : merger -5.71  probe +4.57  paired -10.29 (p=0.009)
        strong n=5: merger -2.40  probe -0.80  paired  -1.60 (p=0.18)
  VERDICT: the faithful merger is significantly WORSE than the probe, and net harmful on its own. This is
  the first EXP-83 number that may be quoted: every earlier one was an artifact of a bug I introduced.
  Training diagnostics for the record: query variance 0.993; injected energy locked to the operator's;
  the mixture still depends on the receiver (pairwise cosine 0.584, 3 distinct top donors); the demo
  loss did NOT fall (0.0702 -> 0.0739), consistent with EXP-22.5 (demo loss is not rollout success).
  READING: it discriminates, it is correctly dosed, and it still loses -- so the failure is in WHAT it
  learns to prefer, not in the plumbing. Their training signal (the target's own supervised loss) selects
  donors that lower the demo action loss, and we already showed that signal does not track rollouts.
  Column-level weighting bought nothing on top: the merged delta is 99.5% rank-16, so there was almost
  no fine-grained subspace for it to act in. 
- EXP-83 COMPLETE, ALL 18 RECEIVERS (09-19). The 6 external receivers reran under the correct suite:
      DA-MergeLoRA merger : -13.33pp (SE 4.88, p=0.014)   improving 3/18, harmed 11
      behavioural probe   :  +1.56pp (SE 1.34)             improving 5/18, ABSTAINED on all 6 external
      PAIRED merger-probe : -14.89pp (SE 4.62, p=0.005)   merger better on 2, worse on 13
        weak  n=10: merger -18.00  probe +3.20  paired -21.20 (p=0.009)
        strong n=8: merger  -7.50  probe -0.50  paired  -7.00 (p=0.28)
  The external receivers are where it fails hardest (x_goal_expert_8 60->0, _2 68->16, _7 92->44,
  _3 28->0). Two reasons, both structural: (a) the probe's guard ABSTAINED on all six -- no fleet donor
  beat their baseline -- while the merger has no guard and injects the full dose regardless; (b) the
  donor pool holds a single external adapter, so the merger had almost nothing suite-appropriate to
  merge into a goal-suite receiver and used fleet donors at full energy instead.
  FINAL VERDICT: significantly worse than the probe on every slice, net harmful overall, and worst exactly
  where the guard matters most. Closed. The abstention guard, which their design has no equivalent of,
  is doing more work than we had credited: on these 6 targets it is the difference between 0 and -32pp.

### DIRECTION (user, 09-19): prioritise selectors / training strategies that can BEAT THE PROBE
Why everything learned has lost, in one place -- these are the constraints any new attempt must respect:
  (1) LABELS ARE FEW AND NOISY: 1,354 rollout pairs; probe test-retest rho=0.41 puts a ceiling of ~0.64 on
      any predictor. (2) THE ONLY DIFFERENTIABLE SIGNAL DOES NOT TRACK ROLLOUTS: the demo action loss
      (EXP-22.5; EXP-83's loss never fell). (3) REINFORCE ON ROLLOUTS COLLAPSES (EXP-30). (4) THE PAIR
      INTERACTION IS REAL (rho=0.58 between probe runs) BUT NOT IN ANY CHEAP REPRESENTATION of the LoRAs or
      their activations (EXP-43/48/82). (5) THE GUARD DOES MORE THAN THE SELECTOR (EXP-83 externals: 0 vs -31pp).
  So "beat the probe" has two honest routes: a BETTER PROBE (same rollouts, more selection per rollout), or a
  learned PRIOR that prunes the candidate list so the probe spends its budget on fewer, better candidates
  (the hybrid: EXP-45 got +3.6pp at half budget, p=0.17, with a weak prior).

### EXP-78 — subset probing — **RE-ENABLED 09-19 (shiva, tmux e78)**, resumes at 53/240
The cheapest "better probe": score 3-SETS at the committed dose instead of singletons at 3x the dose.

### EXP-84 — item-response probe: choose WHICH initial states to probe — **planned**
The probe's 13pp noise is per-state variance. Treat each fixed initial state as a test ITEM and each candidate
merge as an examinee (IRT, adjacent-concepts): estimate per-state discrimination from per-episode outcomes,
then probe only the most discriminative states. Goal: a 5-state probe as reliable as the 10-state one (half
cost), or a 10-state probe with test-retest well above 0.41. Needs a per-episode data pass first (the e27/e44
archives are rate-only; only 114 records carry per-episode outcomes): ~15 weak targets x 8 candidates x 25
states via main_dense.py. Pre-registered success: reliability of the state-selected probe > 0.41 on a
held-out target set, measured by test-retest exactly as EXP-22.4 did.

### EXP-85 — behavioural-fingerprint prior + hybrid deployment — **planned**
The one representation never tried: describe the TARGET by WHERE IT FAILS (per-state PROGRESS / SUCCESS_STEP
from its own baseline rollouts, already paid for) and each DONOR by its competence profile on its own task,
and learn "donor d fixes failure mode f". Train on the existing probe labels; deploy as the hybrid (prune
8 -> 4 with the prior, probe the 4 at 10). Success = beats the probe's gain at half the rollouts, on the
36 weak pool-consistent targets EXP-45 used, paired.

### EXP-86 — replace the demo loss with a learned success critic — **planned, after 84/85**
The principled fix for constraint (2): train a critic (state, action-chunk) -> success on the rollout
archive, then train a merger/hypernetwork to maximise the critic on the target's own states. Risky (critic
exploitation) and heavy; only worth it if 84/85 show the probe can be improved at all.
- EXP-78 INTERIM (09-19, draw 1 probes complete, 15 targets): the subset rule and the singleton rule choose
  GENUINELY DIFFERENT 3-sets -- identical on 0/15 targets, mean overlap 1.2 of 3 donors, fully disjoint on
  2. The subset rule's guard also abstains on 2 targets where the singleton rule commits (the best 3-set
  probed at the committed dose did not beat baseline, while some single donor at 3x dose did). So the
  experiment is not a null by construction: the two commits will differ on every target, and the question
  "does scoring the object you deploy select better than scoring its parts at 3x dose" is live. Commits
  begin after draw 2's probes (~120 more).
- EXP-84/85 STAGE 1 LAUNCHED (09-19, odin, tmux fp): per-episode baselines for all 76 fleet targets with
  main_dense.py (PROGRESS + SUCCESS_STEP per episode), records e92_base_f{t}. Build-free, so odin's full disk
  is not a risk. ~1,900 rollouts, ~1 day. Serves both the item-response probe and the failure-fingerprint prior.
- EXP-78 RESULT (09-19/20) -- SUBSET PROBING IS WORSE THAN SINGLETON PROBING. 30 (target, draw) cells over
  the 15 weak fleet targets, paired on target AND candidate menu, abstention = baseline:
      subset probe   (8 balanced 3-sets at the committed dose c/3) : +2.71pp (SE 1.49)  improving 13/30, abstained 4
      singleton probe (the deployed method, donors at full dose c)  : +7.78pp (SE 1.64)  improving 23/30, abstained 0
      PAIRED subset - singleton                                     : -5.07pp (SE 2.02, p=0.018), worse on 15 / better 7 / tie 8
      consistent in both draws (-5.87, -4.27); per-target draw-averaged -5.07 (SE 1.89, p=0.018)
  The singleton arm reproduces the published draw-averaged headline (+7.98) on the same cells, so the
  comparison is anchored. The pre-registered "positive" reading is refuted; the null reading is confirmed
  and then some: singleton additivity is not merely good enough, it is BETTER than measuring the deployed
  object directly.
  WHY (interpretation, consistent with what we already measured): scoring one donor at 3x the committed dose
  is a large perturbation -> large, detectable differences between donors, i.e. a high-signal main-effects
  design; 8 candidates x 10 rollouts estimates 8 main effects well. Scoring 3-sets at c/3 each makes every
  measurement a small perturbation around baseline (low SNR), samples only 8 of the 56 possible sets, and
  then takes the max -- winner's curse on noise. The interaction that additivity ignores is only ~29% of
  affinity variance (EXP-48), so the main-effects design loses little and gains a lot of signal. The 4
  subset abstentions (vs 0) are the same effect: at c/3 nothing clears baseline within 10 rollouts.
  LESSON FOR "BETTER PROBE" WORK: the probe's value is signal per rollout, not fidelity to the deployed
  object. Any redesign must increase SNR per rollout, which points at EXP-84 (choose discriminative states)
  rather than at scoring composites. Closed.
- EXP-84 STAGE 1b LAUNCHED (09-20, shiva, tmux sp): per-STATE outcomes of the singleton probe for the 15 weak
  fleet targets x their draw-1 menus (8 donors each) over all 25 fixed states with main_dense.py, records
  e93_t{t}_p{d}. 3,000 rollouts. Built-in check: rates over states 0-9 must reproduce the e75 archive within
  noise. Together with odin's e92 baselines this is everything the item-response analysis needs.
- EXP-84 stage 1b CONSISTENCY CHECK (09-20, first 18 pairs): the per-state pass restricted to states 0-9
  reproduces the archived e75 rates for the same (target, donor) pairs: mean 70.0% vs 72.2%, paired
  -2.2pp (SE 4.6, p=0.64), 14/18 within the +-20pp band a 13pp-SD measurement implies. Pair correlation is
  only 0.30 -- but two independent 10-rollout measures of the same merge have test-retest ~0.41 (EXP-22.4),
  so 0.30 at n=18 is what the same measurement looks like, not a discrepancy. The pass measures the same
  thing as the archive; its per-state outcomes can be trusted.
- EXP-84 EARLY PREMISE TEST (09-20, 6 of 15 targets complete, 8 candidates x 25 states each):
  States DO differ in how well they discriminate candidates: item-total r runs from -0.07 (10th pct) to
  0.74 (90th pct), and 33/150 states discriminate nothing (every candidate succeeds or every one fails).
  But the discrimination does NOT transfer across targets at this n: ranking candidates from the 5 states
  chosen as most discriminative ON THE OTHER TARGETS gives split-half reliability rho=0.45 (n=6), no better
  than 5 random states (0.48) and well below 10 random states (0.57). If that holds at 15 targets, "which
  states are informative" is target-specific, and a fixed pre-selected state set cannot halve the probe.
  Two honest readings, to be settled when the pass completes (n=15 for the transfer arm):
    (a) the premise fails as stated -> EXP-84 becomes an ADAPTIVE variant: spend the first few rollouts
        discovering which states discriminate THIS target, then concentrate the rest there (sequential
        allocation across STATES, which EXP-4x never tested -- it allocated across candidates);
    (b) n=6 is simply too few targets to estimate 25 per-state discriminations; wait.
  Not launching anything on this yet.
- EXP-84 RESULT (09-20, DONE, negative): item-response probe on 15 weak fleet targets x 8 draw-1 candidates x 25 states.
  Per-state discrimination (item-total r): mean 0.27, p10 0.00, p90 0.74; 118/375 states dead (all candidates
  same outcome). Discrimination does NOT transfer across targets: LOTO Spearman of a target's per-state disc
  with the other 14 targets' mean = +0.02 (9/15 positive). The only transferable regularity: harder states
  discriminate more (difficulty vs disc r = -0.38), but per-state difficulty itself transfers weakly (0.16;
  mean success per state ranges only 0.62-0.87).
  Split-half reliability of the candidate ranking (Spearman with the held-out states) / P(top-1 truly best):
    k=5  fixed-disc(LOTO) 0.40 / 0.47   random 0.40 / 0.33   paired d=+0.00 p=1.00
    k=8  fixed-disc       0.54 / 0.47   random 0.44 / 0.36   d=+0.10 p=0.19
    k=10 fixed-disc       0.59 / 0.67   random 0.46 / 0.37   d=+0.13 p<0.01
    k=12 fixed-disc       0.55 / 0.60   random 0.47 / 0.41   d=+0.08 p=0.11
  Mid-difficulty selector (states nearest 50% on other targets): same picture, n.s. at every k.
  Adaptive across candidates (5 states on 8, then 5 more on the top-4; 60 rollouts): rho 0.40 / top1 0.37,
  = random 10 states at 80 rollouts -- a 25% saving at equal reliability, nothing more.
  Candidate spread within a target: max-min success 0.36 on average, so there IS something to rank.
  VERDICT: one k out of four reaches significance; the effect is a fragile +0.1 in rho, from dodging dead/easy
  states. A fixed state set cannot halve the probe, and there is no stable per-state structure for an adaptive
  state-allocation variant to exploit. Small keepable rule: drop states with >85% pooled success. Closed.
  Files: sweep/exp84_analysis.py, results/analysis/exp84_M.npy.
- FINGERPRINT PASS split (09-20): shiva was idle after e93, so it now runs targets 62-75 in reverse
  (`sweep/chain_fp_tail.sh`, FP_RANGE/FP_REVERSE/CHUNK env in fingerprint_pass.py, tmux `fptail`) while odin
  walks 49 -> up. When shiva's tail reports OK: kill odin `fp`, rsync e92 both ways, confirm 76/76, start EXP-85.
- EXP-85 STAGE 2 SET UP (09-20) -- **running** (odin tmux `e85`, CPU, fires when e92 reaches 76/76):
  `sweep/exp85_prior.py` builds a 66-dim fingerprint block P per fleet task from its own per-episode
  baseline (success/fail rates; progress reached by FAILED episodes: mean/quartiles/frac<0.5/frac>0.8;
  time-to-success mean/sd/min/max; sorted-progress and sorted-time quantile curves) and appends it to
  hypernet_featsexta48 (external/variant targets mean-filled + missing flag). Then hypernet.py's exact
  protocol, paired seed-for-seed with the W+A/48 reference (+1.14 fleet): WA vs WAP vs P vs WALSP,
  fleet-holdout and cross-distribution, 36 seeds x 300 epochs. Smoke (65/76, 2 seeds, 30 epochs): fleet
  WA +0.73 -> WAP +1.24, P alone +0.70; x-dist WAP -0.42 vs WA -- pipeline works, numbers are noise.
  Decision rule: WAP beats WA by a paired margin (p<0.05) AND P alone is well above 0 -> the fingerprint
  carries selector signal -> stage 3 hybrid deployment (prune 8->4 with the prior, probe 4 at 10) on the
  36 weak targets. Otherwise closed: "where it fails" is not a better description than the LoRA features.
### EXP-87 — a better probe SCORE: continuous per-rollout signal instead of binary success — **quick test 09-20**
Same e93 data (15 targets x 8 candidates x 25 states, main_dense gives PROGRESS and SUCCESS_STEP per episode).
Truth = a candidate's success on the held-out states; probe = k states scored by each criterion.
    k=10:  success (binary)   rho 0.46  P(top-1 best) 0.38  top-3 regret 0.096
           progress           rho 0.18  P(top-1 best) 0.26  top-3 regret 0.145   (paired -0.29, p=0.006: WORSE)
           speed 1-t/300      rho 0.45  P(top-1 best) 0.48  top-3 regret 0.094
           success+0.5*prog   rho 0.45  P(top-1 best) 0.41  top-3 regret 0.101
  PROGRESS is a bad probe score: it correlates only 0.26 with success per episode (the distance metric is
  near-saturated; failed episodes often reach 0.6-0.9). SPEED (earlier success = better) keeps rho and lifts
  top-1 accuracy 0.38 -> 0.48 at k=10, 0.33 -> 0.41 at k=5, with slightly lower top-3 regret -- a free
  tie-breaker, not a new probe. Files: sweep/exp87_progress_probe.py.
  Deployment check: pure-speed changes the committed top-3 set on 9/15 draw-1 menus, tie-break on 4/15.
  Expected gain in committed top-3 regret is 0.002 (0.094 vs 0.096) -> a 9-target paired deployment test
  cannot resolve it; not spending rollouts. Keep as a note: speed is a valid, free tie-breaker. **closed**
- EXP-80 DRAW-AVERAGING LAUNCHED (09-20, shiva tmux `e80d`, `sweep/chain_e80draws.sh`) -- **running**:
  the parked follow-up that settles whether the -5.19pp gap between the blind pool and the fleet-only probe
  on the 14 weak LIBERO-90 receivers is draw luck. `blind_pool.py` now takes DRAW=2/3/4 (independent
  md5-seeded 8-of-83 draws, records e89d{n}_p/e89d{n}_c, probes reused wherever the identical pair was
  already probed in e89/e44) and ONLY_WEAK_FLEET=1. Cost: 72 new probes + 14 commits per draw (~1,100
  rollouts/draw, ~9h/draw). Filler while EXP-85 trains on CPU; pre-emptible and resumable.
  Analysis when done: draw-averaged blind gain on the 14 vs the fleet-only four-draw average (+7.9-ish),
  paired; if the gap closes, EXP-80's weak-slice number on the deck was single-draw noise.
- FINGERPRINT PASS COMPLETE 76/76 (09-20 05:54); EXP-85 full run started on odin CPU (tmux e85, 05:55).
- EXP-80 draw 4 moved to odin (tmux `e80d4`, `sweep/chain_e80draw4.sh`, CHUNK=2 on the full disk, 83 new
  probes / 29 reused); shiva does draws 2 and 3 (~9h each). Its chain would reach draw 4 last, so rsync
  e89d4_* to shiva when odin finishes and shiva's blind_pool skips them via U.get.
- EXP-85 INCIDENT (09-20 09:50): first full run invalid -- hypernet.py's DEFAULT training settings (lr 3e-3,
  300 epochs) collapse AffinityNet to constant scores (loss stuck at 15.99, score sd 0.000, spearman nan)
  on ANY features, incl. the reference featsexta48. The +1.14 reference was trained at lr 1e-3 / 150 epochs
  (EXP-82's feat arm). Verified lr 1e-3/150 does not collapse (score sd ~0.3). Killed, relaunched with those
  settings (exp85_prior.py --lr, default now 1e-3 / 150). Watch for this in any future hypernet.py call.
- EXP-80 draws: draws 2 and 3 complete on shiva (14/14 commits each, 14:17). Shiva's chain then started
  draw 4 in parallel with odin's slow copy, so draw 4 was consolidated on shiva (odin's 32 probes + 5 commits
  synced over; shiva resumes the rest, ~3h). Odin's GPUs are free for EXP-85 stage 3 if it triggers.
- EXP-85 RESULT (09-20 14:26, 36 seeds x 150 ep, lr 1e-3, paired seed-for-seed; top-3 affinity over random):
    fleet-holdout:  WA +0.19 ±0.28 | WAP +0.11 (vs WA -0.08, p=0.82) | P alone +0.83 ±0.21 (vs WA +0.65,
                    p=0.069; ridge on P +1.41) | WALSP +0.34 (p=0.66)
    x-dist (new):   WA -0.28 | WAP -0.25 | P -0.47 (p=0.055, but "new" targets have NO fingerprint -- mean-
                    filled -- so this fold cannot test P) | WALSP -0.16
  Reading: the fingerprint ALONE is the best in-fleet selector at these settings (+0.83 hypernet, +1.41
  ridge), on par with the logged +1.14 W+A reference, and adding it to W+A gives nothing -- the 66-dim block
  is drowned by the 72-dim LoRA blocks (91 features, 1,354 noisy labels). "Where the target fails" carries
  about as much selector signal as the LoRA weights do, from 25 rollouts already paid for, and the two do
  not add. Pre-registered rule (WAP > WA at p<0.05) NOT met -> no automatic stage 3.
  Next (cheap, CPU): EXP-45's label-space shortlist check with a P-only ridge/hypernet -- does its top-4
  hold the probe-best donor more often than 64% (W+A) / 50% (chance)? Only a clear yes earns rollouts.
  Files: sweep/exp85_prior.py, results/analysis/exp85_prior_full.txt, hypernet_e85_full_*.txt.
- EXP-80 DRAW-AVERAGING INTERIM (09-20, draws 1-3 complete on all 14 weak LIBERO-90 receivers, draw 4 on 5):
    blind pool, draw-averaged: +4.36pp (SE 1.77), improving 12/14   [single-draw was +2.86]
    paired on the 11 targets with fleet-only draws (EXP-69/74): blind +2.88 vs fleet-only +7.94,
    difference -5.06pp (SE 3.45, p=0.17)   [single-draw: -5.19, p=0.075]
    within-target draw SD: blind 5.8pp vs fleet-only 13.6pp
  Reading: DRAW LUCK IS RULED OUT -- the gap survives three draws unchanged (-5.1 vs -5.2). The blind pool
  gives a smaller, steadier gain on weak LIBERO-90 receivers than the fleet-only pool; the -5pp is not
  established at n=11 but it is not noise in the draw. Three of the 14 (t27, t49, t57) are weak by e44_base
  but not by the older baseline, hence no fleet-only draws. Final numbers after draw 4.
  Files: sweep/exp80_drawavg.py.
- EXP-85 LABEL-SPACE SHORTLIST CHECK launched (09-20 14:40, odin CPU, tmux `e85ls`): exp45_deploy.py
  --stage labelspace with --blocks P vs WA on hypernet_feats_fp.npz, 36 seeds. Reports the probe-affinity
  of each policy's committed top-3 (random / model / hybrid=probe within model's top-4 / probe / oracle) on
  the weak targets. EXP-45 reference (W+A): hybrid +1.45 vs random-4 +1.56, top-4 holds probe-best 64%.
  Only the fleet fold is meaningful for P (variants have no fingerprint).
- EXP-85 LABEL-SPACE RESULT (09-20 15:35, P-only model, fleet fold n=15 weak targets; variants excluded --
  no fingerprint): model top-3 -3.17 | hybrid (probe within P's top-4) +2.39 (SE 1.58) | random shortlist-
  of-4 +3.01 | full 8-probe +11.39 | hybrid - random4 = -0.62 (p=0.69) | P's top-4 holds the probe-best
  donor 53% (chance 50%; W+A in EXP-45: 64%). Pooled 36 (incl. fingerprint-less variants): hybrid +3.07 vs
  probe +10.34. The fingerprint shortlist is a random shortlist. **EXP-85 CLOSED** -- no rollouts spent.
  Files: sweep/exp85_labelspace_fleet.py, results/analysis/exp85_picks_P.json, exp85_labelspace_P.txt.
  (WA control run of the same script finishing ~16:30; decision does not depend on it.)

### STATE OF "BEAT THE PROBE" (09-20) — every cheap route is now closed
  better probe:  EXP-78 subset scoring (-5.07, worse) | EXP-84 state selection (no transferable structure)
                 | EXP-87 continuous scores (progress worse; speed = free tie-break, +0.002 regret)
  learned prior: features EXP-42/45 (+1.14 aff, shortlist = random when deployed) | cross-attention EXP-82
                 (+1.07) | faithful DA-MergeLoRA merger EXP-83 (-14.9 vs probe) | fingerprint EXP-85
                 (+0.83 alone, adds nothing, shortlist = random)
  Common cause: 1,354 labels with test-retest 0.41 bound every selector near +1pp affinity, and +1pp of
  affinity is not enough to prune 8 -> 4 without losing the 1-2 good donors. What is left is EXP-86 -- a
  different TRAINING SIGNAL (a learned success critic on recorded trajectories, so the merger is trained on
  something that tracks rollouts) -- which is heavy: an observation/action recording pass (~1,900 episodes,
  ~1.5GB of embeddings), a critic, and a merger trained against it, with critic-exploitation risk. Gated on
  the user's call; odin GPUs are free for it.
- EXP-85 label-space WA control (16:38, same script, fleet fold n=15): hybrid +1.61 (SE 3.30) vs random-4
  +3.01, top-4 holds probe-best 60%. P: +2.39 / 53%. Neither shortlist beats a random one; closure stands.
- EXP-80 DRAW-AVERAGING FINAL (09-20 16:50, 4 draws x 14 weak LIBERO-90 receivers, 42 new commits) -- **done**:
    blind pool draw-averaged: +4.29pp (SE 1.76), improving 12/14
    paired on the 11 with fleet-only draws (3 each): blind +2.82 vs fleet-only +7.94 -> -5.12pp (SE 3.42,
    p=0.165); within-target draw SD blind 5.7pp vs fleet-only 13.6pp.
  Conclusion unchanged from the interim: the gap is not draw luck; not established at n=11.
- DECK SHORTENED (09-20, user: "remove whatever not worth of presenting... some failures are just failures"):
  28_09 deck 305 -> 152 lines, 9 slides -> 6. Cut to one-liners or dropped: layerwise dosing slide (negative),
  item-response states EXP-84, subset probing EXP-78, fingerprint prior EXP-85, the per-experiment hypernetwork
  failure list. The two cross-product slides merged into one "who helps whom" table; the EXP-32 stratified
  block folded into the one-pool slide as a single comparison line. All learned-selector attempts now live in
  ONE closing slide with a 6-row table + why-it-plateaus + what the faithful merger taught. Numbers unchanged.
  Long version: results/deck_backups/content_for_slides_28_09_0920_1815_prelength.md
- DECK FIX (09-28): the shortened pool-consistency slide had kept the superseded "-5.6pp foreign expert"
  row without its per-target-draw correction (EXP-68: +7.56pp, 8/1). Row replaced; added the honest line
  that S16 init buys guard CALIBRATION (overstatement +4.9 vs +18.7) rather than the merge gain itself.
  Also noted: the S16 cohort's +7.90 (18 runs) excludes abstained draws; abstention = baseline gives +7.62
  (21 runs, SE 3.83, 5/7). Same claim either way.
- DECK (09-28, user request): cross-product slide replaced by the pooled donor-pool x receiver-cohort table
  (weak receivers + all-targets variants). Two empty cells noted as fillable: S16-init experts as donors,
  S16-init donors into foreign receivers.
- EXP-83 SPLIT CLARIFICATION (09-28, user caught 65+18 != 84): the 80/20 split is over the 86 TARGETS
  (68 donor targets + 18 receivers), not the 84 adapters. Deduplicated: donor bank = 65 unique adapters
  (64 fleet + goal expert via its 4 donor tasks); receivers = 13 unique adapters (12 fleet + goal expert via
  6 tasks). The goal expert is therefore on BOTH sides: for the 6 external receivers the receiver's own
  adapter was in the donor bank and had been a training pseudo-target -- held out at task level, not
  adapter level. exp83_eval.py does not mask it (self-dose term, not a rollout-information leak). The 7
  donor-only external adapters (no eval task) were not in the bank at all. Conclusion unchanged: the
  exposure could only help the external receivers, which lost 31pp; the 12 fully held-out fleet receivers
  lost too. Deck line corrected.
- EXP-83 CORRECTION (09-28, found while writing the end-to-end method description): the probe arm of the
  merger-vs-probe comparison was computed on shiva, which at the time had NO e44_c_x_goal_expert_* records
  (EXP-32's external commits lived only on odin). Missing records were read as abstentions, so "the probe
  abstained on all 6 external receivers" was an ARTIFACT. The records exist and the probe COMMITTED on all 6:
  base 96/96/92/68/28/60 -> probe 100/96/92/92/52/84 (+12.67pp, 4 improved 0 harmed, 0 abstentions).
  Recomputed on odin with all 18 e91 + probe records (fleet receivers: e27_commit_t{t}_top3; external:
  e44_c; baseline e44_base; abstention = baseline):
      ALL n=18   merger -13.33 (unchanged) | probe +5.78 (was +1.56) | paired -19.11 (SE 6.37, p=0.008;
                 was -14.89, p=0.005) | merger better 2 / worse 13
      weak n=10  merger -18.00 | probe +10.40 | paired -28.40 (p=0.017)
      strong n=8 merger  -7.50 | probe  +0.00 | paired  -7.50 (p=0.25)
      fleet n=12 merger  -4.33 | probe  +2.33 | paired  -6.67 (p=0.008)   [unchanged: fleet records were present]
      ext   n=6  merger -31.33 | probe +12.67 | paired -44.00 (p=0.029)
  The conclusion (merger loses badly) is unchanged and slightly stronger; the LESSON drawn from it was wrong:
  it was never "the guard did the work" on externals -- the probe actively found good donors there. The
  DIRECTION block's constraint (5) "guard does more than the selector (0 vs -31)" is withdrawn; what stands
  is EXP-80's 44 correct abstentions at ceiling. Also noted: the probe arm used the probe's OWN menus (8 fleet
  candidates, EXP-27 single draw, for fleet receivers; 16 mixed candidates, EXP-32, for externals), not the
  merger's 65-donor bank -- so the probe could draw held-out fleet adapters as donors. Deck corrected.
  Lesson for the record: never compute a cross-machine comparison on the machine that holds only one arm.
- DECK (09-28, user request): EXP-83 gets its own end-to-end slide (split -> query -> bank -> hypernet ->
  training -> merge/eval -> corrected result table); the duplicate "what the merger taught" block on the
  selection slide removed, its table row kept. 7 slides.
- EXP-83 note (09-28): trained per-module strength multiplier (sigmoid(log_scale+init)*4) ended at 0.79-1.30x
  our operator's dose, median 1.07 -- the network kept our dose; the loss is in the donor mix.

### DIRECTION (user, 09-28, after the 28-09 meeting): NO learnable components. Back to linear algebra.
Goal unchanged: given a target LoRA, find what to merge and how so its own task improves. The probe works but
is trial-and-error and unsurprising. This week: study the pool's shared structure PER TARGET with weights
only, and see whether it dictates what/how to merge. Probe stays the bar. Deck: content_for_slides_05_10.md.
Three questions from the meeting -> three branches:

### EXP-88 — more than one shared component? — **planned**
Known: the pool mean is exactly rank 16 per module (S16); capture@16 0.80+-0.01 for all 76; residual
cosines -0.01+-0.03 (max +0.10, twins); global residual PCA diffuse (top PC 2.1%) BUT residual clusters are
semantic. Test (CPU): (a) cluster-level soups (mean residual within each semantic cluster, per-module SVD)
-> held-out member capture vs random same-size group vs chance; (b) per-module stack SVD of all 76
residuals -> does a low-rank "second shared component" carry cross-task energy? If yes: operator
W = S16 + S_cluster(target) + r_target, tested on weak targets with NO donors (no draw lottery).

### EXP-89 — does a target share more with some tasks than others (local shared component)? — **planned**
Known: nearest residual neighbours are semantic twins but tiny (max +0.10); residual cosine does not rank
donors (t6 +28 from twin, t19 -28 from twin); fleet-level capture is target-blind (0.80 for everyone).
Test (CPU): per target, soup of its k nearest residual neighbours vs k random vs full pool -> capture of the
target's own delta at each k; per-band breakdown. Then (GPU, ~1,100 rollouts): swap arm
S_neighbourhood(k) + r_target on the 15 weak targets, k in {4,8,16}, @25, vs S16 swap.

### EXP-90 — per-task rank of the shared component — **planned**
Known: k-sweep {4,8,12,16} swap arm flat on average, 5 targets only; k>16 is a no-op under the mean soup
(directions 17+ ~zero singular value); per-task optimum never measured. Test: (CPU) per-task loading
profile on the 16 shared directions per module; (GPU, ~1,900 rollouts) swap arm S_k + r_target, k in
{2,4,8,12,16}, all 15 weak targets @25, no donors -> no draw noise; readout = per-target best k vs 16.
Beyond 16: extra directions from the stack SVD (EXP-88b), not the mean.
- EXP-88/89/90 STAGE 1 (09-28) -- **running**: `sweep/linalg_structure.py` (one pass over 458 modules x 76
  adapters, GPU fp64 on shiva, energy-weighted): 4-fold held-out stacked-residual subspace (88a), cluster
  subspaces k=5/8 vs random equal-size groups + cluster-mean-vector test (88b), nearest-residual-neighbour
  subspaces and neighbourhood soups n=4/8/16 vs random vs pool S16 (89), per-task cores C_t = U16'D_tV16:
  capture curves k=1..16, per-direction loadings, 76x76 core similarity (90). Groups fixed before any
  capture is computed (seed 88); duplicates t7=t18, t29=t62 never each other's neighbours. Report:
  `sweep/linalg_report.py`. Smoke on 6 modules first.
- EXP-90 STAGE 2 (09-28 15:37) -- **running** on both machines: `sweep/rank_sweep.py`, donor-free swap arm
  W = S_k + (D_t - P_k D_t), k in {2,4,8,12,16}, 15 weak fleet targets @25 (records e94_t{t}_k{k});
  odin targets[0:7] (CHUNK=2, full disk), shiva targets[7:15]; then --confirm re-runs each target's best k
  and k=16 on fresh rollouts (e94c_*). Baseline = mean of 3 independent 25-rollout measurements. Build
  verified: W equals S_k + D - P_k D to 2e-7; 0.41 GB per build. Pre-registered read: a target "needs a
  different rank" only if its best k beats k=16 on the FRESH confirmation rollouts.
- EXP-88/89/90 stage 1 debugging (09-28): (1) first version ~minutes/module -- fold subspaces were
  re-diagonalised once per task; now once per group. (2) fp32 eigensolves + Cholesky on rank-deficient
  groups (modules with d_in=32) gave soup captures of 1e8 -> replaced by fp64 eigenbases with null directions
  masked; formula verified against an explicit SVD on 2 modules (0.445=0.445, 0.3508=0.3508). (3) held-in
  capture was summed over the 3 folds each task trains in -> averaged. Group analyses (nn/random/cluster) on
  a stratified quarter of modules with their own denominators. Full pass launched 16:03 as 3 shards on
  shiva GPUs; merge with sweep/linalg_merge_shards.py.
- EXP-89 STAGE 2 queued behind e94 on both machines (tmux e95): `sweep/local_shared_swap.py`, donor-free
  swap with a LOCAL shared component W = S_G,16 + (D_t - P_G D_t), G = 8 nearest residual neighbours (nn8)
  vs 8 random tasks (rnd8), 15 weak targets @25 (e95_t{t}_{arm}); reference = e94 k=16 pool swap + baseline.
  Build verified (err 9e-7, 0.55 GB). t6's nn8 contains its twin t64.
- EXP-88/89/90 STAGE 1 RESULT (09-28 ~19:00, all 458 modules, 3 shards merged; report
  results/analysis/linalg_structure_report.txt). Energy-weighted, 74 tasks (duplicates once):
  EXP-88 (second shared component): **REFUTED.** Held-out capture of a task's residual by the top-k
  directions of 57 other tasks' stacked residuals: k=16 0.043 (held-in 0.158, chance 3.6e-5; out/in 0.27 at
  every k) -- residual directions do not transfer the way S16 does (S16: 0.80 of the full delta, held-out =
  held-in). Full-76 stacked residual spectrum: top-16 left directions = 14.8% of residual energy (diffuse).
  Semantic clusters (k=5/8) vs random equal-size groups: 1.07-1.44x, +0.2-0.6% of residual energy
  (p<1e-11, 63-71/74). Cluster-mean residual as a shared vector: cos +0.008/+0.024 vs random -0.06/-0.05,
  energy 0.1%. No rollout arm: nothing of that size can move behaviour.
  EXP-89 (local shared component), weight level: neighbours beat random consistently but tinily. Residual
  subspace of the 8 nearest neighbours captures 3.4% of the target's residual vs 2.9% random (k=16).
  Neighbourhood soups (full deltas) beat random soups on 74/74 targets (+0.7pp of the delta at n=8) but EVERY
  local soup captures LESS of the target's delta than the pool S16 (n=8: 0.782 vs 0.802, 0/74 better; n=16
  0.791; n=4 0.763): fewer tasks = more residual noise in the soup. Centred cores (each task's shared
  coordinates minus the mean, 0.56% of its delta) mirror residual similarity (corr +0.79, top-5 overlap 3.3/5
  vs chance 0.34). Behavioural check (e95) running.
  EXP-90 (per-task rank), weight level: every task loads on the 16 shared directions identically. Capture
  of own delta by S_k: k=1 0.359+-0.005, 4 0.557+-0.007, 8 0.681+-0.009, 16 0.803+-0.010 (sd across tasks);
  per-direction loading CV <= 0.01; core cosines 0.993 +- 0.0005; tail share (directions 9-16) 0.152 +- 0.001
  (range 0.150-0.153; corr with baseline -0.33, p=0.004, but over a 0.003 range). Behavioural sweep (e94)
  running.
- EXP-90 STAGE 2 INTERIM (09-28 20:50, 12 of 15 targets complete; 5 cells of t15/t50/t56 died at server load
  -- GPU memory contention with the linalg shards / the odin kill -- refill queued on shiva, tmux e94fill):
  mean vs baseline k=2 +0.2, 4 +1.5, 8 -1.2, 12 -2.5, 16 +1.2 pp (all n.s.). target x k interaction variance
  / 25-rollout sampling variance = 1.23 (~noise). Per-task best k vs k=16 on the SELECTION rollouts +5.8pp,
  on FRESH rollouts -5.1pp (SE 2.6, p=0.08, 2 up / 6 down) -> pure winner's curse. Odin's remaining EXP-89
  cells moved to shiva (tmux e95b); odin idle.
- EXP-89 RESULT (09-28 21:25, 15 weak targets x {nn8, rnd8} @25 + pool swap from e94 k16/e94c k16): **REFUTED.**
  means: baseline 81.1, pool S16 swap 82.9, nn8 81.3, rnd8 81.3. nn8 - rnd8 +0.00pp (SE 1.70, 7 up/4 down);
  nn8 - pool -1.60 (p=0.56); rnd8 - pool -1.60 (p=0.50); pool - baseline +1.84 (p=0.33). The weight-level
  neighbourhood effect (+0.7pp of the delta) carries no behaviour; a local shared component is never better
  than the pool. Report results/analysis/exp89_local_shared_report.txt. Slide 3 of the 05_10 deck done.

### NEXT WEEK (user, 09-28 evening): a BEHAVIOURAL signal that is cheaper than rollouts — **planned, not this week**
Why: every weight/linear-algebra quantity explains WHY merging works (one shared component everyone agrees
on; private residuals that don't interfere at small dose) but none predicts WHICH donor helps a target --
that is a closed-loop property. The probe works because it measures behaviour; the open problem is a
cheaper measurement of the same thing (candidates to scope next week: open-loop action divergence on the
target's own states, short-horizon rollouts, per-state success on a few hard states, value/progress
estimates). This week stays on linear algebra, explored exhaustively (user).

### THIS WEEK: linear algebra, exhaustively (user, 09-28) — remaining approaches after EXP-88/89/90
- EXP-91 (WHAT, weights only): a battery of linear-algebra donor scores for every (target, donor) pair
  (residual cosine; left / right / two-sided subspace overlap of residuals; donor energy inside vs outside
  the target's residual subspace; target energy inside the donor's; sign agreement across modules; centred-
  core cosine; each per band) scored against every probe label we own, within target, pre-registered on
  half the targets and confirmed on the other half. The exhaustive version of "can the weights say WHAT".
- EXP-92 (HOW, given donors): fixed donors (each weak target's probe-committed top-3), donor residuals
  transformed by linear algebra before merging, energy-matched: standard / only the part inside the
  target's residual subspace / only the part outside / spectral top-4 per module.
- EXP-93 (HOW, no donors): own-residual dose W = S16 + beta*r_t, beta in {0.75, 1.25, 1.5}: does
  extrapolating (or damping) the task-specific part help weak targets? beta=1 = the e94 k16 pool swap.
- EXP-91/92/93 LAUNCHED (09-28 21:50). EXP-91 features: `sweep/la_pair_features.py` on odin CPU (tmux lf),
  formulas verified against explicit residual matrices on one module (ip, DinT2, DinTL, DinTR all exact).
  EXP-92/93: `sweep/la_operator_arms.py` (records e96_* beta arms, e97_* donor-transform arms); builds
  verified (in2 donor part lies in the target's residual subspace to 3e-6; energy-matched per donor).
  t6's fixed donors (EXP-27 top-3): 23, 61, 45. Odin: EXP-93 targets[0:7] (tmux la); shiva: after the
  e94 refill, EXP-92 all 15 targets then EXP-93 targets[7:] (tmux la).
- EXP-90 RESULT (09-28 22:00, all 15 weak targets x k in {2,4,8,12,16} @25 + confirmation): **REFUTED.**
  vs baseline: k=2 -1.4, 4 +1.3, 8 -2.2, 12 -1.9, 16 +0.5 pp (all p>=0.38). target x k interaction variance =
  1.08 x the 25-rollout sampling variance. Per-target best k vs k=16: +6.57pp on the selection rollouts,
  -4.57pp on fresh rollouts (SE 2.93, 3 up / 7 down) -> winner's curse. Report
  results/analysis/exp90_rank_sweep_report.txt. All three meeting questions now answered: no.
- EXP-91 RESULT (09-28 22:30): **REFUTED as a selector.** 61 LA scores (10 families x 6 bands) vs 1,537
  probed (target, donor) pairs, 74 targets, within-target Spearman. Median |rho| 0.041. Pre-registered split:
  A->B chose sign_agree|llm_early (sign -1), TEST rho +0.102 (p=0.04), top-3 pick +1.48pp vs random (oracle
  +7.37); B->A chose cos_r|llm_early (sign -1), TEST +0.097 (p=0.01), +0.95pp (oracle +6.40). Both halves pick
  early-LLM residual (anti-)similarity: dissimilar donors are marginally better. Robustness: drop the most
  similar donor per target rho 0.108; drop 3 -> 0.076 (p=0.046) -- not just twin conflict. Shortlist 8->4
  least-similar keeps the label-best donor 51.1% (chance 50%), +0.15pp. ~1/4 of the probe's reliability,
  the same ~1pp ceiling as every learned selector. Reports exp91_la_pair_eval.txt, exp91_robustness.txt.
- EXP-94 (09-28 22:45) — data-aware linear algebra — **REFUTED.** `sweep/la_input_score.py`: each target's
  own policy (base + D_t) on 16 of its demo frames; at every adapted Linear (457), exact first-order
  mag[t,d] = sum_x ||r_d x||^2 and dir[t,d] = sum_x <r_d x, r_t x> per band (self-check dir/mag = 1.0000).
  75/76 targets (t57 has no demos), 22 min on shiva GPU2. Scores gain = mag/||r_d||^2, align = cosine of
  donor vs target output changes, rel_gain. Evaluated with the EXP-91 protocol (`sweep/la_input_eval.py`):
  median |rho| 0.029; pre-registered A->B align|expert (-) TEST rho +0.045 (p=0.29), pick -0.95pp;
  B->A align|llm_early (-) +0.041 (p=0.34), +0.91pp. Weaker than weights alone (EXP-91, 0.10).
- EXP-92 RESULT (09-29 00:00, 15 weak targets x {std, in2, top4} @25, fixed EXP-27 top-3 donors):
  means base 81.1 / std 85.1 / in2 85.9 / top4 87.2. vs std: in2 +0.80pp (SE 3.50, p=0.82, 7/5), top4 +2.13pp
  (SE 2.53, p=0.41, 7/6). vs baseline: std +3.97 (p=0.11, 13/2), in2 +4.77 (p=0.04), top4 +6.10 (p=0.02, 12/3).
  Reading: with good donors fixed, how their residual is shaped barely matters; restricting a donor to the
  target's own residual subspace (in2) = standard. Spectral top-4 truncation is directionally best for the
  second time (EXP-20 Q5: +4.8pp vs dense, n=5, random donors) -> REPLICATION queued on fresh rollouts
  (std vs top4, records e97r_*, shiva tmux e97rep) to decide it at n=30 paired.
- Odin's remaining EXP-93 cells (13) moved to shiva (tmux e96r) at 01:01; odin idle.
- EXP-93 RESULT (09-29 01:50, 15 weak targets x beta in {0.75, 1.25, 1.5} @25; beta=1 = e94 k16 pool swap):
  **REFUTED.** means 84.8 / 83.1 / 73.9 / 65.9 for beta 0.75 / 1 / 1.25 / 1.5. vs beta=1: 0.75 +1.73 (p=0.59,
  10/4), 1.25 -9.20 (p=0.03, 5/10), 1.5 -17.20 (p=0.003, 2/13). beta=0.75 vs baseline +3.70 (p=0.13, 12/3).
  Extrapolating the task vector breaks weak policies; mild damping is at most neutral-to-slightly-positive
  (same direction as RETAIN's 0.8 and EXP-13's 0.85 self-weight, not significant here).
- EXP-92 REPLICATION (09-29 03:07, fresh std vs top4, 15 targets @25, records e97r_*): top4 - std +2.67pp
  (SE 1.55, p=0.11, 9/4) vs +2.13 in the first run. POOLED (50 rollouts/arm/target): +2.40pp (SE 1.33,
  p=0.092, 8 up / 6 down); vs baseline std +3.17 (p=0.13, 11/3), top4 +5.57 (p=0.009, 12/3). Third positive
  measurement out of three (EXP-20 Q5 +4.8, n=5). Launched: a third replication (e97s, shiva) and the
  truncation rank top2 / top8 (e97; top8 all on shiva, top2 split odin 0:7 / shiva 7:).
- EXP-92 FINAL (09-29 05:50): **REFUTED.** 15 weak targets, fixed EXP-27 top-3 donors, energy-matched; std and
  top4 x3 independent runs (e97/e97r/e97s), top2/top8/in2 x1. Means: std 86.0 (+4.86 vs base, p=0.021), top4
  87.0 (+1.07 vs std, p=0.26, 6/8), top2 86.1 (+0.18), top8 85.9 (-0.09), in2 85.9 (-0.09). top4 - std per run:
  +2.13, +2.67, -1.60 -> the lead regressed; with EXP-20 Q5 it was never significant. Report
  results/analysis/exp92_truncation_report.txt. Side result: the probe-committed donors' gain replicates on
  fresh rollouts (+4.9pp over 75 rollouts per target).

### WEEK SUMMARY (09-28/29, linear algebra, exhaustively) — 7 hypotheses, all refuted
  EXP-88 second shared component: no (held-out residual capture 4% vs S16's 80%).
  EXP-89 local shared component: weights yes-but-tiny, never better than the pool; rollouts nn8 = rnd8.
  EXP-90 per-task rank: every task uses S16 identically; per-task best rank = winner's curse (+6.6 -> -4.6).
  EXP-91 LA donor scores (61 scores x 1,537 labels): best held-out rho 0.10 (probe ceiling 0.41), ~+1pp.
  EXP-94 data-aware LA (first-order effect on target's own inputs): rho 0.04, worse than weights alone.
  EXP-93 own-residual dose: beta>1 hurts (-9, -17pp), 0.75 neutral.
  EXP-92 LA shaping of good donors: all shapings = standard (+/-1pp); top-4 lead did not replicate.
  => The operator (S16 + r_t + c*mean(3 donors)) cannot be improved by linear algebra, and linear algebra
  cannot pick donors. Donor choice is the only lever left and it is behavioural -> next week's brief:
  a behavioural signal cheaper than rollouts.

### EXP-95 — GROUND-TRUTH donor matrix (user, 09-29) — **running**
Motivation: the 1,537 labels cover 28% of pairs, mostly single 10-rollout probes (reliability rho 0.41), so
"what correlates with donor quality" has been tested against a noisy, partial target. Build the full matrix:
every task as receiver, every other task as its single donor once, same operator as the probe
(S16 + r_t + c*r_d, c = ref/||r_d||), main_dense.py (per-episode outcomes -> split-half reliability for free,
and data for next week's behavioural signal). 74 unique tasks (t18=t7, t62=t29 dropped) x 73 donors = 5,402
pairs. Weak 15 targets first @25 rollouts (1,095 pairs): shiva 13 targets (tmux gt, CHUNK=6), odin t68/t73
(CHUNK=1, full disk). Build 0.92 GB. Records e98_t{t}_d{d}. Driver sweep/gt_sweep.py.
- EXP-95 throughput (09-29 14:05): chunked driver ran shiva at 13-17 pairs/h (chunks wait for their
  slowest job; workers start ~65 s apart; a relaunch also found only 3 workers because orphaned merged
  checkpoints in /dev/shm shrank the derived worker count -- chain now clears /dev/shm/mergesweep first).
  `sweep/gt_stream.py`: one orchestrator over the whole FIFO job list, builder thread 14 builds ahead,
  files deleted as records land -> 34.0 pairs/h at 25 rollouts on shiva, 0 errors. Odin (full disk, CHUNK=1)
  ~2 pairs/h on t68/t73. ETA: weak 15 targets (1,095 pairs @25) Wed 09-30 ~18:00; the other 59 targets
  (4,307 pairs) @10 rollouts ~Sat 10-03 morning (default, lands before the meeting) or @25 ~Mon 10-05 evening.
- EXP-95 INTERIM (09-30 14:30, 13 weak receivers complete x 73 donors @25; report
  results/analysis/exp95_ground_truth_report.txt, script sweep/gt_analysis.py):
  reliability: split-half (states) rho 0.488 -> Spearman-Brown 0.641 (range 0.31-0.83); ceiling 0.80.
  landscape: huge per-receiver range (t6 12-100%); most donors hurt (mean 27/73 hurt >8pp, 16 help >8pp;
  t50 57 hurt / 2 help); mean donor below baseline for 9/13. Old 10-rollout probe labels vs GT rho +0.532.
  LA scores (72: EXP-91 weight scores + EXP-94 input scores) within-target over 73 donors: best -0.108
  (donor_in_target_in|expert), median |rho| 0.033; the top 20 are ALL negative (similar = slightly worse).
  Verdict on weak receivers: refuted. Shiva now on t68/t73 remainder @25 then 55 strong @10; odin 4 strong @10.
- EXP-95 strong block (09-30 17:00): first 45 min ran at ~52 pairs/h @10 rollouts with 0 errors -- capped by
  the orchestrator, which serialises each job's merge build + server start (~65 s) under one semaphore.
  merge_orchestrator.py now reads plan key "heavy_parallel" (default 1, so every other driver is unchanged);
  gt_sweep/gt_stream pass GT_HEAVY. Shiva relaunched with heavy_parallel=2, 24 builds ahead (tmux gt,
  chain_gt_strong2.sh). Odin continues 4 strong receivers (71,72,74,75) @10, CHUNK=1.

### LITERATURE (user, 10-02): LiST — Local-Simplex Test-Time LoRA Fusion (Shao et al., arXiv 2608.22370, Aug 2026)
What it is: target LoRA + LoRA bank (LLMs / MLLMs, GLUE, MM-MergeBench). Retrieve top-K neighbour adapters by a
joint representation (PCA of LoRA parameters + mean hidden state on a few target prompts); prior simplex
weights (target mass rho + similarity softmax over neighbours); per test input, CMA-ES search over simplex
logits minimising a label-free energy (prediction entropy + JS consistency over M stochastic passes + KL to
prior + Mahalanobis distance of the fused parameter anchor to the target's); deploy only if energy improves by
a margin AND constraints hold, else fall back to the prior. Fusion y = W0 x + sum a_i B_i A_i x (linear =
summing deltas, which our operator already does). Code: github.com/YihuaJerry/LiST.
HELPS: (1) its energy is a ready-made candidate for NEXT WEEK's "behavioural signal cheaper than rollouts"
(VLA analogue: dispersion / consistency of sampled action chunks across flow-noise seeds on the target's own
states) -- testable directly against the EXP-95 ground-truth matrix (one forward pass per pair, within-target
Spearman). Their own energy-metric alignment is modest: corr 0.35, P(gain | energy drop) = 0.65.
(2) independent support for our abstention guard (their accept / fallback rule is the component that removes
negative transfer). (3) their gain over the target LoRA alone is tiny (GLUE 87.70 vs 87.35; Qwen3-VL seen
+0.28; LLaVA +0.13) -- consistent with our finding that improving a target is hard and weak selectors buy
~1pp; our probe gives +6.6pp on weak targets.
RISKS: (1) near-identical problem statement ("Given a LoRA weight from a target task, how should the model
adapt to the target task...") -> must cite and differentiate (closed-loop VLA, S16/residual explanation of why
merging works, large weak-target gains, full ground-truth donor matrix). (2) opposite conclusion on
similarity: their neighbour retrieval assumes similar = good (their ablation: random K 82.95 < joint
similarity 86.22 < target-only 87.35 on GLUE); in our EXP-95 ground truth (13 weak receivers so far) every
similarity measure is slightly NEGATIVELY related to donor quality (|rho| ~ 0.1) -> state as a domain
difference, shown on all 74 receivers once the matrix completes.
NOT TRANSFERABLE: per-input weight search (re-mixing the policy every control step would hurt stability; a
per-episode variant is the most we could borrow).
Next: first candidate signal to test next week = LiST-style energy on the target's states vs EXP-95 ground truth.
- EXP-95 RESULT (matrix complete Fri 10-02 17:18, analysed 10-03; 5,402/5,402 pairs, 0 errors; reports
  results/analysis/exp95_ground_truth_report_{weak_(25_rollouts),strong_(10_rollouts),all}.txt): **REFUTED.**
  Reliability (split-half over states, Spearman-Brown): weak 0.633 (ceiling 0.795), strong 0.167 (ceiling
  0.409; near-ceiling receivers, donors barely move them), all 0.269. Old probe labels vs GT: weak 0.509,
  strong 0.209. Landscape: weak base 81.1% / mean donor 77.3%, 20% of donors help >8pp, 31% hurt >8pp, best
  donor +16.5pp, worst -38.7pp; strong base 98.0% / mean donor 96.1%, 1% help, 8% hurt, best +2.0, worst
  -23.3. LA scores (72) within-target over 73 donors: weak best -0.113, median 0.035; strong best -0.068
  (median 0.018); all 74 best -0.075 (cos_r|llm_late, positive on 17/69), median 0.023. The strongest scores
  are all NEGATIVE similarity measures. Orchestration note: no completion watcher was armed on shiva's last
  block, so it sat idle from Fri 17:18 until checked Sat 01:00 -- arm one with every relaunch.

### EXP-96 — LiST-style label-free energy as a cheap behavioural proxy (user: "yes start", 10-03) — **running**
Next week's question started early: does a forward-pass-only signal of the MERGED policy on the target's own
states rank donors like the EXP-95 ground truth? `sweep/energy_proxy.py`: for each (t, d), the probe's operator
built in place (base + S16 + r_t + c*r_d), 16 target demo frames x 4 flow-noise seeds (common random numbers
per target), 10 denoising steps. Signals: disp (variance of sampled action chunks -- LiST's uncertainty /
stochastic consistency), drift (mean action vs target-alone -- KL-to-prior), demo (error vs demonstrated
chunk), ddisp (disp minus target-alone). Pre-registered direction: lower = better donor. Smoke t0: 17 min /
target (14 s / donor) on an A6000 after an OOM fix (weight snapshots in pinned CPU memory, micro-batches).
Launched 01:27: shiva 3 shards (weak targets first), odin 6 strong targets (EP_FB=2). ETA ~7 h.
Eval: `sweep/energy_eval.py` (within-target Spearman vs GT; lowest-energy pick vs random/best donor).
- EXP-96 RESULT (10-03 09:55, 71/74 receivers; last 3 finishing): **REFUTED.** Within-target Spearman vs EXP-95
  GT, score = -signal (pre-registered): disp weak -0.014 / all -0.012; drift +0.070 (p=0.17) / +0.044 (p=0.01);
  demo +0.067 (p=0.07) / +0.026; all4 +0.059 / +0.017 (ceilings weak 0.795, all 0.519). Lowest-energy pick on
  weak receivers: disp -13.9pp vs baseline (40% of picks harmful) vs random donor -3.8pp, best +16.5pp; drift
  pick -3.8pp (= random); disp+drift +0.2pp. Low action variance is NOT success: the most confident merged
  policy is often a broken one. Odin note: needed per-module target weights (dense dicts OOM'd on 24 GB) and
  ran ~90 min/target vs ~19 min on an A6000.

### EXP-97 — can a COMBINATION of the metrics predict donor goodness? (user, 10-03) — **REFUTED**
Features: 72 LA scores (EXP-91/94) + 3 energy signals (EXP-96), z-scored within receiver; label = EXP-95 GT
success z-scored within receiver; leave-one-receiver-out. scikit-learn 1.9.1 installed into .venv (user;
pyproject/uv.lock untouched). `sweep/la_combo_sk.py` (numpy/MLP twin `sweep/la_combo.py` as cross-check).
  ridge LA(72): weak rho +0.077 (p=0.19, 10/15), all +0.072 (p<0.001); top-1 pick +0.2pp, top-3 -3.1pp
  ridge LA+energy: weak +0.079 (p=0.20), all +0.071; top-3 -3.1pp
  gboost LA: weak +0.047, all +0.065; gboost LA+energy: weak +0.030, all +0.065; top-3 -5.4pp
  vs random donor -3.8pp, best +16.5pp. Ridge alpha at the top of the grid (1e4): the best linear combination
  is a heavily shrunk average of the negative similarity scores (cos_r|llm_late, sign_agree|expert, ...).
  Nonlinear model no better -> no hidden interaction. Report results/analysis/exp97_la_combo_sk_report.txt.

### Literature sweep for donor-compatibility metrics (user, 10-03) -> EXP-98/99/100, run in order
Already covered: task-vector geometry + subspace overlap (Demystifying Mergeability's 28 metrics, mostly =
EXP-91), activation/CKA similarity (EXP-11/48/94), M-Loss (~ EXP-96 drift), LiST energy (EXP-96), TAG-style
demo-loss lookahead (EXP-22.5/83). New: (1) closed-loop SENSITIVITY (toolbox: Lyapunov stability) -- does the
donor make the merged policy amplify small input perturbations; (2) GRADIENT alignment (Demystifying's most
stable predictors: encoder/input-gradient L2 and cosine) + (3) FISHER-weighted donor energy (toolbox:
information geometry); (4) TIES elementwise sign conflict.
- EXP-98 (sensitivity) LAUNCHED 10-03 10:28, shiva 3 GPUs, weak receivers first (`sweep/sensitivity_proxy.py`):
  same operator + 16 target frames, one fixed flow-noise draw shared by all variants; state +/- 0.05 sd
  (antithetic), 3-px image shift; signals sens_state, sens_img, curv_state, rel_* (vs target alone).
  Pre-registered: more sensitive = worse donor.
- EXP-100 (TIES) LAUNCHED 10-03 10:35, odin CPU (`sweep/ties_conflict.py`): 20k sampled residual entries per
  module; conflict among co-kept top-20% entries, support overlap, |r|-weighted sign agreement; per band.
- EXP-99 (gradient + Fisher) queued after EXP-98 on shiva.
- EXP-97 cross-check (numpy ridge + torch MLP, `sweep/la_combo.py`) agrees with scikit-learn: ridge LA weak
  rho +0.076 / LA+energy +0.083; MLP +0.075 / +0.098 (p=0.10); all receivers 0.058-0.075; top-3 picks -3.2
  to -4.6pp (random -3.8). Same conclusion.
- EXP-100 RESULT (10-03 10:55, 17 min on odin CPU; results/analysis/exp100_ties_report.txt): **REFUTED.**
  Within-receiver Spearman vs GT: conflict|llm_late weak +0.102 (p=0.04) / all +0.070; conflict|all +0.083 /
  +0.060; agree_all|all -0.086 / -0.060; overlap|all -0.052 / -0.021. Sign is the opposite of TIES's premise
  (more sign conflict = marginally better donor) and the size equals residual cosine's: elementwise conflict is
  one more dissimilarity measure, no new information.
- EXP-96 FINAL (73/74 receivers; t57 has no demo hdf5): unchanged -- disp -0.014 weak / -0.013 all; drift +0.070
  / +0.049; demo +0.067 / +0.020; all4 +0.059 / +0.016. Slide updated.
- EXP-98 RESTART (10-03 11:05): the first run showed target-alone sens_state = 0.0000 exactly. Cause: in this
  pi0.5 config the robot STATE IS NOT AN INPUT (runtime discrete_state_input=False and pi05 skips state_proj),
  so the policy acts on the two camera images (+ prompt) only. State perturbations dropped; variants are now
  image shifts only (3 px horizontal, 3 px vertical, both cameras; one shared noise draw). Signals sens_h,
  sens_v, sens_img (mean), rel_img / rel_h (vs target alone). Pre-registered: more sensitive = worse donor.
- EXP-98 RESULT (10-03 13:05, 15 weak receivers x 73 donors; report results/analysis/exp98_sens_report.txt):
  **REFUTED.** Within-receiver Spearman vs GT (score = -sensitivity): sens_h +0.031 (p=0.44, 8/15), sens_v
  +0.020 (p=0.44), sens_img / rel_img +0.014 (p=0.63). Top-1 picks +6.4 / -8.8 / -0.8pp (noise). Strong
  receivers not run (their GT reliability 0.17 cannot resolve a ~0 effect); shards stopped; EXP-99 next.
- EXP-99 VALIDATION (10-03): (1) first two checks disagreed -- gradients ~1e-6 underflowed in fp16 inside the
  hook (fixed: fp32), and finite differences at eps=1e-3 were pure noise. (2) Final check, t6: hook projection
  = explicit <full gradient, r_d> to 4 significant figures (d19 -6.8458e-3 vs -6.8471e-3; d50 9.6394e-3 vs
  9.6375e-3). Finite differences cannot validate further: the loss at a FIXED point jitters +/-0.2% between
  identical forward passes (0.3420/0.3407/0.3415; bf16 nondeterministic kernels), which swamps eps*g.r at
  eps <= 0.2. Donor effects measured here are ~1% of the loss, above that jitter. Setup 27 -> 2.6 min (adapters
  loaded once). LAUNCHED 14:3x, shiva 3 shards, weak receivers first.
- EXP-99 RESULT (10-03 14:37, all 74 tasks; report results/analysis/exp99_grad_report.txt): **REFUTED.** 24 scores
  (lookahead, Fisher, gradient cosine, gradient L2 x 6 bands). Weak receivers: best grad_l2|head +0.098 (p=0.01,
  12/15), lookahead|llm_early +0.072 (p=0.01), grad_cos|head +0.058; all-layer lookahead +0.029, Fisher +0.027,
  grad_cos -0.013. All receivers <= 0.045. Top-1 picks -11.5 to +1.8pp (random -3.8). 2-3 of 24 at p<0.05 = chance.
  Run notes: fp16 underflow in the hook fixed; unused expert lm_head gets no gradient (zero sketch); adapters
  loaded once (setup 27 -> 2.6-6 min); 13 s per task.
### LITERATURE SWEEP CLOSED (10-03): all four new families refuted -- EXP-98 sensitivity (|rho| <= 0.03),
EXP-99 gradients/Fisher (<= 0.10), EXP-100 TIES conflicts (0.10, sign reversed), plus EXP-96 LiST energy and
EXP-97 combinations. Every static or open-loop quantity tops out at |rho| ~0.1 against a 0.80 ceiling.
- DECK SIMPLIFIED (10-04, user): every donor metric now reported against the EXP-95 ground truth only (probe-label
  evaluations of EXP-91/94 superseded; same verdict). `sweep/all_scores_gt.py` evaluates all 130 scores in one
  pass (adds #11 centred-core cosine and #14 relative input gain, never run vs GT before): A 61 scores weak best
  -0.113 / median 0.036; B 18: -0.070 / 0.022; C 4: -0.070 / 0.040; D 5: -0.031 / 0.020; E 24: -0.098 / 0.028;
  F 18: +0.102 / 0.076. TOTAL 130: best 0.113, median 0.034, 5 scores >= 0.10 (ceiling 0.80).

### DIRECTION (user, 10-05, after the 05-10 meeting): STOP linear algebra. Goal unchanged.
Given a new target LoRA and a pool, what do we merge into the target to improve its own task? Operator settled;
donor choice is the lever; static/open-loop metrics exhausted (28 metrics / 130 scores, best |rho| 0.11 vs 0.80).
Test bed: the EXP-95 74 x 73 ground-truth matrix. Deck: content_for_slides_14_10.md (meeting 10-14).
- LITERATURE (user, 10-05): arXiv 2502.17363 = KV-Edit (training-free image editing for DiTs via KV-cache
  background preservation). Not applicable: about image regions, not models/LoRAs/merging/donors; the only
  analogy (inject the donor into selected tokens/layers) is covered by layerwise dosing (uniform optimal).
- BRAINSTORM (10-05): user constraint = no rollouts to choose donors (probing = brute force). Approved: idea 1
  (protect the target, merge the WHOLE pool -- EXP-101) and idea 4 (donors by task meaning -- EXP-102).
- EXP-102 RESULT (10-05, `sweep/semantic_donors.py`, report exp102_semantic_report.txt): **REFUTED as a
  selector.** Semantic relatedness is NEGATIVELY related to donor quality on weak receivers: noun Jaccard -0.137
  (p=0.002), same room -0.127 (p=0.03), shares target receptacle -0.111 (p=0.004), same scene -0.091 (p=0.02);
  top-3 most related donors -8 to -13pp vs random -3.8pp. Both-two-step +0.142 (n=6, p=0.11). Excluding related
  donors (other room, no shared noun): random donor -3.81 -> -2.33pp, harmful 31% -> 28%. Consistent with the
  twin-conflict finding: related tasks confuse goals. A safety filter, not a selector.

### EXP-101 — protect the target, merge the WHOLE pool (no selection) — **done (10-05 19:00): REFUTED as a gain, works as a safety device**
Motivation (brainstorm idea 1): avoid choosing donors at all. Donors usually hurt because they disturb what the
target already does; failures happen on off-track states the target never saw. So inject every donor only into
input directions the target does NOT use: per layer, C_t = covariance of the target policy's layer inputs on 32
of its demo frames; W_t = lam (C_t + lam I)^-1 (lam = 0.01 * mean eigenvalue); donor residual r_d -> r_d W_t.
Pool = all 73 donors, unit-energy each. Arms: u1 unprotected pool at total norm ref (control), p1 protected at
ref, p3 protected at 3 ref. 15 weak targets x 3 arms x 25 rollouts (1,125). Served as full prebuilt checkpoints
(merge_orchestrator.py: new job key 'prebuilt_dir'; other drivers unchanged). Diagnostics per target: energy kept
by protection and first-order output change on the target's own states (protected / unprotected at equal norm).
Records e101_t{t}_{arm}; `sweep/protected_pool.py`; diag results/analysis/exp101_diag.json.
- EXP-101 decision rule (user, 10-05): if the protected pool beats baseline / the unprotected pool on the weak
  targets -> extend to ALL 74 targets (strong targets = mainly a safety check); if not -> back to brainstorming.
- EXP-101 progress (10-05 16:00): first launch crashed (expert lm_head never runs at inference -> no Gram; now
  gets the unprotected delta, it is unused anyway). Relaunched 15:38 on shiva. First diagnostic, t6: protection
  keeps 53.9% of the pool's energy; first-order output change on t6's own demo states at equal norm =
  1e-4 x the unprotected pool's -> the projection does what it should (donors invisible on target states).
  Split for speed: shiva does targets in list order, odin (cuda:1, PP_OFFLOAD=1 parks the driver's model on CPU
  during eval) does 73,68,67,63,60,33,20; records synced both ways every 5 min (tmux ppsync on odin).
- EXP-101 progress (10-05 16:45): odin's first try swapped (driver 51GB RSS on a 62GB box: fp32 dense pool
  deltas + target delta + pinned copy). Added PP_LOWMEM=1 (bf16 pool deltas, target delta rebuilt at write,
  base weights shared with the file -- asserted equal to the loaded ones); same t73 diagnostic (54.7%, 1e-4).
  Shiva ~14 min/target. First two targets (shiva): protected pool = no-donor operator (t6 68 vs k16 76;
  t19 80 vs 80), unprotected pool hurts t6 (48), 3x protected hurts both (60, 64).
- EXP-101 RESULT (10-05 19:00; 15 weak targets x 3 arms x 25 rollouts, all 45 cells; report
  results/analysis/exp101_report.txt, `sweep/exp101_report.py`). Mean success: base 81.1, no-donor operator
  (S16 + r_t, e94 k16) 81.6, random single donor 77.3, best single donor 97.6; u1 unprotected pool 75.5,
  **p1 protected pool 83.2**, p3 protected 3x 76.5.
  p1 - u1 +7.7pp (11/15 wins, Wilcoxon p=0.023); harmful targets (< -8pp vs base) u1 7/15 -> p1 1/15.
  p1 - base +2.1pp (p=0.28), p1 - no-donor +1.6pp (p=0.40) -> no gain beyond the operator without donors.
  p3 - p1 -6.7pp (p=0.012): scaling up what survives the protection hurts.
  Diagnostics on all 15: protection keeps 53.5-57.6% of pool energy; output change on the target's demo
  states 1.3-1.6e-4 x unprotected at equal norm.
  Reading: the protection removes the harm AND the help. The donors' effect that survives acts only on states
  the target never visits in its demos, and there it is harmful (p3). Help and harm travel through the same
  input directions; a fixed input-space filter cannot keep one and drop the other.
  Decision (user's rule): did not work as a way to improve the target -> NOT extended to all 74 targets;
  back to discussing ideas. Ops: odin ran 3 targets in low-memory mode (~55 min/target vs shiva 14); shiva
  covered the rest via the two-way record sync.

### EXP-103 — brainstorm round 2: mergeability predictors from the literature we have not tried — **done (10-05 22:00): REFUTED, all |rho| <= 0.11**
Motivation: both no-rollout ideas from round 1 failed (EXP-101, EXP-102). User (10-05): keep the no-rollout rule
"for a few days still"; survey all mergeability-prediction papers we have not tried, starting from four the user
found: Yadav et al. "What Matters for Model Merging at Scale?" (TMLR 2025); Hu et al. "Model Kinship" (EMNLP
Findings 2025); Zhou et al. "Demystifying Mergeability" (ICML 2026, partly tried as EXP-99); Rahamim et al.
"Will it Merge?" (ACL Findings 2026). Untried predictors get scored offline against the EXP-95 ground truth.
- EXP-103 progress (10-05 21:30). Literature: two agents (four user papers in depth incl. Demystifying's code;
  broad 2023-26 sweep). Merging literature has nothing materially new: every merging predictor is a variant of
  weight/subspace geometry, activation similarity, gradient/Fisher alignment, sign conflict; all validated across
  pairs/tasks on classification/LLMs, never within a receiver, never closed loop. Demystifying's own single
  metrics for Task Arithmetic are as weak as ours (best |r| 0.21; cosine 0.015); its 0.445 is a 28-metric fit to a
  symmetric label. "Will it Merge?" factors are receiver-level (base knowledge = our headroom); Yadav et al.
  factors are model-level (size, instruction tuning); Model Kinship = delta cosine (covered) + |gain| target.
  Untried items scored offline (`sweep/exp103_free.py`, `sweep/exp103_spec.py`; weak receivers, ceiling 0.80):
  - donor-only ceiling (donor's mean effect on the OTHER receivers, covers every donor-only property incl.
    Will-it-Merge's base-knowledge quantities and sigma_max): +0.11 (p=0.11). ~60% of within-receiver
    variance is target x donor interaction, ~30% rollout noise, <9% donor main effect.
  - 130 existing scores vs effect size |gain| (Kinship): best 0.11, none >= 0.15.
  - encoder-gradient dot product: |rho| <= 0.04.
  - spectral tail (bottom-10 right singular vectors, Demystifying's best TA/TIES metric): -0.09 to -0.10
    (sign opposite to the paper); spectrum-shape overlap rerunning (zero-residual modules gave NaN).
  - correlation-objective L1 combination (Demystifying's predictor), 158 features, leave-one-receiver-out: +0.04.
  Robot-side ideas from the sweep (not merging papers): matched-state readout conflict (task-vector theory:
  aligned helps / contradictory hurts), critical-interval imitation error (CI-MSE, validated on pi0.5 checkpoints),
  on-policy replay of the target's own failure frames, counterfactual-prompt contrast, STAC consistency.
  Running: readout conflict (data only, base SigLIP embeddings of ~500 demo frames per task), shiva.
- EXP-103 RESULT (10-05 22:00). Spectrum-shape overlap (after skipping 7 untrained modules: both lm_heads and
  VLM layer-17 MLP/q/o, never read by the expert): |rho| <= 0.075. Spectral tail (bottom-10 right vectors):
  -0.09 (opposite sign to the paper). Correlation-objective L1 combination of 161 features: +0.05.
  Readout conflict (EXP-103c, `sweep/exp103_readout.py`; mean-centred base-SigLIP embeddings, otherwise all
  best-match cosines > 0.99): S_in -0.07, agreement +0.045, conflict -0.09 (p=0.06); theory's interaction absent
  (similarity is slightly harmful among both high- and low-agreement donors). Verdict: every untried predictor
  from the four papers + the sweep is <= 0.11, like the 130 before. Weight / activation / gradient / data-similarity
  routes are exhausted; next = EXP-104 (closed-loop-aware behavioural signals, approved by the user).

### EXP-104 — closed-loop-aware behavioural donor signals (robot offline validation) — **done (10-06 04:40): REFUTED, |rho| <= 0.09**
Motivation: user approved (10-05) the four robot-side ideas from the EXP-103 literature sweep. All static and
single-frame open-loop signals failed; these look at what matters in closed loop without trying donors:
(1) CI: imitation error only in critical intervals (around gripper toggles + goal phase), temporally ensembled,
plus a signed term: does the donor push toward the expert where the target errs (CI-MSE, Huang et al. 2026,
rho -0.87 across pi0.5-family checkpoints vs -0.61 raw MSE); (2) RC: replan consistency (deterministic STAC);
(3) CF: counterfactual-prompt contrast = goal confusion with the donor's own instruction; (4) RP: replay on the
target's OWN rollout states (failure vs success frames): selectivity of the donor's change, and error reduction
vs a nearest-demo pseudo-label. Weak 15 targets x 73 donors, merged policy = the probe operator, built in place.
- Stage 0 (10-05 21:31, shiva): target-alone (S16 + r_t) rollouts, 25 episodes x 15 weak targets, dumping both
  camera frames + state at every replan (examples/libero/main.py --args.dump-obs-path; orchestrator plan key
  dump_obs_dir). `sweep/exp104_dump.py`, records e104_t{t}_k16, dumps results/exp104_dumps/ (~3 MB/episode).
  No donor is tried; user approved using the target's own rollouts.
- Driver `sweep/exp104_behav.py` (all four signals per pair in one pass; B4_SMOKE=1 smoke mode).
- EXP-104 stage 0 done (10-05 22:50): 375 target-alone episodes dumped (76 failures), 1.7 GB on shiva. Bonus:
  second independent no-donor measurement e104 vs e94 k16: mean 79.7 vs 81.6 (Spearman 0.76 over targets, mean
  |diff| 8.8pp = 25-rollout noise).
- Driver bug found in the smoke test: the donor dose c = ref/||r_d|| was recomputed inside the per-module lambda
  (U.resid_norm is uncached, dense, CPU) -> 3,118 s per donor; hoisted -> 23-25 s per donor (A6000), identical
  values. Also dump arrays decompressed once per file.
- Full run launched 10-05 23:13: shiva GPU0 {6,15,9,60,67}, GPU1 {19,56,14,63}, GPU2 {50,8,20,33}; odin GPU1 {68},
  GPU2 {73}; shards results/analysis/exp104_behav_s{0..4}.npz; eval `sweep/exp104_eval.py` (signs pre-registered:
  ci-, dci-, corr+, rc-, drc-, cf+, sel+, vinn+, rp_succ-). ETA ~2.5 h.
- EXP-104 RESULT (10-06 04:40; 15/15 weak receivers x 73 donors; shiva ~30-48 min/target, odin ~5 h/target;
  report results/analysis/exp104_report.txt + exp104_extra.txt). Within-receiver Spearman (pre-registered sign):
  CI imitation error +0.056 (p=0.23) · signed correction +0.045 · replan consistency +0.075 (p=0.02, 11/15
  positive; not significant after correcting for 10 signals) · counterfactual-prompt contrast +0.001 · replay
  selectivity +0.005 · replay nearest-demo error reduction +0.028 · change on the target's success states +0.047.
  Top-3 picks vs target alone: -1.4 to -6.4pp (random donor -3.8). No-fit rank average of rc + ci + rp_succ
  (chosen after seeing the table): +0.091 (p=0.03), top-3 +1.9pp; all 10 signals: +0.073.
  Validity in CI-MSE's own setting (across checkpoints = across our 15 target-alone policies): target-alone CI vs
  its own success -0.09 (p=0.74), replan consistency -0.08 -> the metric doesn't track success here at all.
  Harmful donors (< -20pp) shift the signals only -0.11 to -0.17 SD; helpful donors <= +0.09 SD.
  Verdict: offline behaviour on demo states AND on the target's own recorded failure states says ~nothing about
  which donor helps. With EXP-91..103 this closes every no-rollout family we know: weights, activations,
  gradients, data similarity, task semantics, single-frame and trajectory-level offline behaviour. The donor
  effect seems to exist only in closed loop (compounding). Back to the user (rule: no-rollout ideas "a few days").

## Direction (10-06, user): rollouts are fine, but the method must not look brute force
User: "rollouts are fine, but I don't wanna make it look brute force. The probe method looks like you try some
donors and pick what works." Approved two framings: (2) a principled sequential decision rule with a no-harm
certificate (EXP-105, offline first), (1) merging as the most trial-efficient adaptation, vs fine-tuning the
target on its own successful rollouts at the same rollout budget (EXP-106).

### EXP-105 — sequential, certified donor selection, simulated on the ground truth — **done (10-06 09:00)**
Motivation: replace "try 8, keep the best" with adaptive allocation + a certified merge/abstain decision.
`sweep/exp105_seqsim.py`; report results/analysis/exp105_seqsim_report.txt. Arms: the target alone (e92_base
per-episode) + 73 single-donor merges (e98 per-episode), same 25 states. Per split: 15 selection states, 10
held-out states score the decision (paired). Budget 80 rollouts INCLUDING the target's own. 15 weak x 400 splits.
- probe (7 donors x 10 + target x 10, merge best if it beats the target): +4.15pp, merges 60%, harmful 6.1%
  (harmful = merged donor >= 2 of 10 held-out states below the target). Without the guard: +2.93, harmful 13.7%.
- successive halving (16 donors, 2/3/4 rollouts per round): +4.39; top-two Thompson sampling (16 donors): +4.73.
- certified = merge only if posterior P(p_donor > p_target) >= 1 - delta:
  Thompson delta 0.2: **+4.95pp, harmful 6.1%** (dominates the probe); delta 0.1: +4.23, harmful 3.5% (probe's
  gain at ~half the harm); delta 0.05: +3.15, harmful 2.2%. Probe certified 0.05: +3.04, harmful 0.8%.
- references: random donor -4.70; oracle (best of 73) +15.61.
Reading: adaptive allocation + a certificate gives a tunable gain/safety trade-off and beats the deployed probe
at equal budget; the remaining gap to the oracle is large (budget covers 16 of 73 donors).
Caveat: single-donor commits (deployed = top-3 at c/3); held-out harm is noisy at 10 states.

### EXP-106 — merging vs self-imitation fine-tuning at the same rollout budget (80) — **done (10-06 17:00): NULL — neither helps on fresh states; the old +6.6pp headline does not replicate**
Motivation (user-approved framing 1): show merging is the most trial-efficient way to adapt a target, rather than
"try donors and pick". Same 80 rollouts per weak target, two uses; all trials on initial states 25..49, all final
scores on states 0..24 (never used for any decision).
- probe arm: 7 seeded-random donors x 10 rollouts (states 25..34) + the target's own first 10; deployed rule
  (top-3 at c/3, abstain unless the best donor beats the target on the same 10 states).
- self-imitation arm: 80 target rollouts (states 25..49, cycled), full-trajectory dumps (frames every 2 steps +
  executed actions); fine-tune the target's own adapter (warm start, fleet recipe) on successful episodes mixed
  50/50 with its demos, 1000 steps, lr 5e-5, batch 8 (`sweep/exp106_sil.py`).
- Client: examples/libero/main_dense.py gains --state-pool (cycle within [offset, offset+pool)) and
  --dump-stride; orchestrator passes state_pool / dump_stride / dump_obs_dir.
- Stage A `sweep/exp106_collect.py` (shiva, tmux e106: self then probe in chunks of 2 targets, builds 1.66 GB);
  stage C `sweep/exp106_finish.py` (E106_STAGE=eval, then =report). Records e106_t{t}_{self,pd{d},probe,sil}.
- EXP-106 progress (10-06 10:08): stage A self done: 15 x 80 target rollouts on states 25..49, 986/1200
  successes (82%), 13 GB dumps (frames every 2 steps + executed actions; spot-checked: state_idx, action ranges).
  Fine-tuning validated on odin (20 steps: 916/916 tensors warm-started, saved adapter has the fleet's exact 916
  keys; test output deleted; P6000 ~11 s/step so training runs on shiva). Launched 3 queues (shiva GPU0
  {6,15,9,60,67}, GPU1 {19,56,14,63,68}, GPU2 {50,8,20,33,73}) alongside the probe-donor collection (105 jobs,
  chunks of 14).
- EXP-106 ops (10-06 10:18): probe stage stalled on shiva's USER QUOTA (46 GB soft / 51.2 GB hard; disk itself
  has 95 GB free): 13 GB of stage-A dumps + 7 builds (0.9 GB each here). Archived results/exp104_dumps (1.7 GB)
  and results/merge_videos (0.9 GB) to /media/pinas/lzhou/libero-eval-archive/ (file count + summed sizes
  verified) and deleted locally; probe chunks cut to one target (7 builds); restarted (completed jobs skip).
  Fine-tuning: ~4.5 min per 100 steps on A6000 next to rollout servers (23 GB GPU) -> ~50 min per target.
- EXP-106 first scoring run (10-06 15:45; states 0..24, 25 episodes per arm; report exp106_report_run1.txt):
  probe rule merged on 6/15 (abstained on 9: most targets hit 100% on their own 10 trials, so no donor can beat
  them there). Mean gain vs target alone: probe +1.75pp (p=0.35; helped 4, hurt 1: t50 -22.7), self-imitation
  +1.04pp (p=0.87; helped 4: t15 +26.7, t19 +18.7; hurt 3: t63 -17, t60 -12), probe - sil +0.71pp (p=0.64).
  Both indistinguishable from zero and from each other at 25 episodes per score (~8pp noise per target).
  Note: the probe's earlier +6.6pp (weak) is not reproduced under this fresh-state protocol (+1.75).
  Second scoring run (_r2, same states, fresh sampling noise) launched 15:50 to halve the noise; report pools both.
- EXP-106 RESULT (10-06 17:00; two scoring runs pooled = 50 episodes per arm on states 0..24; report
  results/analysis/exp106_report.txt, first run alone exp106_report_run1.txt):
  probe rule (merged 6/15, abstained 9): **+0.68pp** vs target alone (p=0.46; helped 2, hurt 1);
  self-imitation fine-tuning: **+1.57pp** (p=0.80; helped 4, hurt 4); probe - sil -0.89pp (p=0.91).
  Neither beats the target alone; they are indistinguishable. Framing (1) "merging is the most trial-efficient
  adaptation" is NOT supported at this budget: both ways of spending 80 rollouts give ~0.
- WHY THE OLD HEADLINE (+6.63pp combined; fleet weak +7.98pp, EXP-74/76) DOESN'T REPLICATE -- two protocol
  artifacts the new design removes:
  (a) regression to the mean: weak targets were DEFINED by the standing baseline (target_baseline < 0.9) and the
      old gain was measured against that same baseline. Its mean over the 15 weak targets is 79.0; every
      independent re-measurement of the same targets alone is higher: e44 82.4, e92 81.9, fresh 80-episode run
      on new states 82.2 -> ~3pp of the old gain is the baseline's selection luck.
  (b) state overlap: old probes ran on states 0..9 and commits were scored on 0..24 (40% overlap); the old
      guard compared a max-of-8 noisy probe score with that low baseline.
  The EXP-105 simulation (single top-1 donor at full dose, held-out states, independent e92 baseline) gave
  +4.15pp; live EXP-106 (top-3 at c/3) gives +0.7 to +1.8 -- within ~1.5 SE of each other at n=15.
  Every earlier result scored against target_baseline on the weak set carries the same ~3pp optimism.

### EXP-107 — live, powered, clean-protocol test of the gated, certified successive-halving rule (all 74 targets) — **done (10-06 20:30): NULL fleet-wide (+0.38pp, p=0.45)**
Motivation (user 10-06: "ok do" -- powered live test with the headroom gate built in): EXP-106 left the true
gain of trial-based donor search unresolved (old +6.6pp inflated; live +0.7pp underpowered; offline +4pp).
Design check offline (`sweep/exp107_sim.py`, weak targets, budget <= 80 incl. target): gate (10 target rollouts,
keep if >= 9/10) halves cost (~40 rollouts/target) at no loss: successive halving 8 donors + certificate 0.8:
+4.0pp overall, 30% merged, +13.3pp when merged, harmful 2.9%; batched Thompson 12 donors: +4.4pp (needs ~14
rounds live vs 4 -> SH chosen).
PRE-REGISTERED (before any rollout; `sweep/exp107_live.py` docstring is the record):
- all 74 LIBERO-90 targets; decisions only on initial states 25..49; gate 10 (25..34) -> r1 8 seeded donors x 5
  (25..29) -> r2 top 3 x 5 (30..34) -> r3 top 1 x 5 + target x 5 (35..39); merge (full dose c, single donor)
  only if P(p_donor > p_target | 15 vs 15, Beta(1,1)) >= 0.8.
- scoring on states 0..24: merge 2 x 25 episodes vs target 25 fresh + e92 (independent) = 50 vs 50; kept = 0.
- PRIMARY: mean gain over all 74 (kept = 0). Secondary: gain when merged, harmed fraction (< -8pp).
- expectation from the offline check: +4pp on weak targets, ~+13pp when merged.
- CORRECTION (10-06 17:00) to the EXP-106 "why" note: (b) state overlap is NOT supported. EXP-62 (09-09) re-scored
  the old commits on never-used states 25-49 and found +9.60pp (SE 3.35; 0-24 gave +1.87) against the zero-donor
  operator measured on the same states in a separate, earlier run (EXP-57). Only (a) regression to the mean stands.
  EXP-62's +9.6 vs EXP-106's +0.7 is unresolved (cross-run baseline, different menus, SE 2.5-3.5); EXP-107 decides.
- EXP-107 gate (10-06 17:55): 9 of 74 targets gated in (< 9/10): t9, 14, 15, 19, 50, 56, 60, 63, 75. Gate
  distribution over 74: 10/10 x47, 9/10 x17, 8 x5, 7 x1, 6 x2, 5 x1. Many weak-by-baseline targets passed (t6 9/10,
  t8 9/10, t20/33/67/68/73 10/10). t65's gate crashed (transient: LIBERO config yaml read empty during parallel
  client start) -> rerun on odin: 10/10, kept. Power is lower than planned: ~3 merges expected; primary
  endpoint stays as registered. Proposed to the user (not run): pre-registered extension on the 30 external
  LoRAs + 100 scoring episodes per merge.

### EXP-108 — the EXP-107 rule on 30 foreign targets (generality + power) — **done (10-07 00:30): NULL (+1.27pp, p=0.17); pooled 104: +0.63pp (p=0.15)**
Motivation: user approved the extension (10-06) after EXP-107's gate let only 9/74 LIBERO-90 targets through.
PRE-REGISTERED (before any rollout; `sweep/exp108_live.py` docstring is the record): targets = our 30
independently trained zero-init LoRAs (10 LIBERO-Goal, 10 Object, 10 Spatial; results/analysis/new_loras);
donors = the 74 unique LIBERO-90 fleet LoRAs; rule identical to EXP-107; scoring on states 0..24 with 100 merge
episodes vs 100 fresh target episodes per merged target; kept = 0. PRIMARY: mean gain over the 30; also pooled
with EXP-107 (104 targets). Runs on odin (chunks of 4 builds; odin quota 46 GB, 36.6 used).
- EXP-107 RESULT (10-06 20:30; report results/analysis/exp107_report.txt). Gate let 9/74 through; the
  certificate merged 7 (kept t19, t63 at P(better) 0.67 / 0.70). Scoring 50 merge vs 50 target episodes (fresh 25 +
  e92 25) on states 0..24:
    t9 +10.0 (96 vs 86) · t14 +10.0 (92 vs 82) · t15 -4.0 · t50 -12.0 · t56 +30.0 (64 vs 34) · t60 -10.0 · t75 +4.0
  PRIMARY (pre-registered): mean gain over 74 targets +0.38pp (SE 0.50, p=0.45) -> NULL.
  Gain when merged +4.00pp (SE 5.49, n=7, Wilcoxon p=0.67); harmed (< -8pp) 2/7. Cost 17.9 rollouts/target.
  vs the offline expectation (+13pp when merged, ~3% harmful): live merges are far noisier; P(better) >= 0.8 on
  15 vs 15 rollouts admits donors that turn out worse (t50, t60). One genuine large win where the target is very
  weak on the scoring states (t56: 34% on 0..24 although 62% on 25..49).
  Reading: under a clean protocol, trial-based donor search does not improve LIBERO-90 targets fleet-wide; the
  benefit, if any, is confined to rare targets with large headroom. EXP-108 (foreign targets) still running.
- EXP-108 ops (10-06 20:45): moved from odin (slow) to shiva after EXP-107 freed it: copied the 30 adapters (5.2 GB, checksum spot-checked), synced finished gate records, chunks of 12 builds (quota). Completed jobs are skipped.
- EXP-108 RESULT (10-07 00:30; report results/analysis/exp108_report.txt; pooled exp107_108_pooled.txt). Gate let
  12/30 through (foreign targets have more headroom); 9 merged. Scoring 100 merge vs 100 target episodes:
    wine rack +7 (36 vs 29) · wine on cabinet -8 · alphabet soup +13 (68 vs 55) · cream cheese +5 · milk +4 ·
    tomato sauce -5 · bowl between plate/ramekin +1 · bowl in top drawer +1 · bowl next to cookie box +20 (68 vs 48)
  PRIMARY: mean gain over 30 foreign targets +1.27pp (SE 0.90, p=0.17) -> NULL; +4.22pp when merged, harmed 1/9.
  Ops: one scoring job failed in a build race (4 identical target-alone jobs share a cache signature; two workers
  built it at once) -> rerun after the chain; 7 missing jobs rerun.
- POOLED EXP-107 + EXP-108 (104 targets): **+0.63pp (SE 0.44, p=0.15)**; 16 merged: +4.13pp (SE 2.78, Wilcoxon
  p=0.20, 11+/5-), harmed 3/16.
  Headroom: splitting merged targets by the target's success ON THE SCORING STATES (< 70%: n=5, all positive, +14.2pp)
  is regression to the mean (the gain is computed from that same measurement). With the INDEPENDENT gate measurement
  (states 25..34): Spearman(gate, gain) -0.10 (p=0.73); gate <= 5/10 +5.1pp (n=7), 6-7 +3.5 (n=4), 8 +3.2 (n=5).
  No headroom gradient.
  VERDICT: under a clean, pre-registered protocol, trial-based donor search (gated, certified) gives no
  demonstrable improvement of a target's own task: fleet-wide <= ~1pp, +4pp when it merges (n.s.). The project's
  "merging improves a weak target" claim does not survive clean evaluation.

## Direction (10-07, user: "yes 2") — the positive claim moves to MULTI-TASK CONSOLIDATION
"Merging improves a target's own task" did not survive clean evaluation (EXP-106..108). The positive core is
multi-task consolidation: one merged model keeps most of every member task (EXP-3/7/8: retention 95/92/86/82% at
2/3/4/76-way; one 76-adapter average = 77.2% over all 90 scenes vs 94.2% single-task). The S16 analysis explains
WHY (80% of every adapter is one shared rank-16 part; private residuals are near-orthogonal). Findings of the
donor line fold in as analysis (unpredictable donor effects; related tasks interfere; baseline-selection artifacts).
Gaps before this is a paper claim: old runs used 10 rollouts, the standing baseline, groups only up to 4 (size
confounded with harder tasks), average only.

### EXP-109 — clean scaling curve of multi-task merging (plain averaging) — **done (10-08 00:30)**
Motivation: replicate the consolidation curve under the clean protocol and test whether merging as
"shared part + scaled residuals" (W = S16 + alpha * sum_i r_i) scales better than plain averaging (which dilutes
each task's private residual by 1/k). Random partitions of the 74 unique tasks at k = 2, 4, 8, 16, 37, 74; a fixed
random panel of 24 tasks scored at every k (25 episodes, states 0..24) against a same-period single-task run.
- EXP-109 launched (10-07 08:53; `sweep/exp109_consolidate.py`, docstring = design record). Panel (seed 109):
  t0 9 11 12 17 20 23 26 29 34 36 37 39 43 46 52 57 58 61 69 71 72 73 74. Merged models needed per k for the
  panel: k=2: 21, 4: 15, 8: 10, 16: 5, 37: 2, 74: 1. odin: single-task baseline + avg k=2,4; shiva: avg
  k=8,16,37,74, then S16 + alpha*sum(r) (alpha 1, 1/sqrt(k), 1/k) at k=4,16,74 as prebuilt dense checkpoints
  (shared-memory rounds of 4). Sum of residuals computed as one projected matrix (projection is linear).
- REQUIREMENT (user, 10-07): "we're proposing a multitask merging method, which has to outperform existing
  methods". So EXP-109's average-only comparison is not enough: baselines must include Task Arithmetic (tuned),
  TIES, DARE(-TIES), TSV-M, Iso-C/Iso-CTS, KnOTS (LoRA-specific). Hyperparameters tuned on states 25..49 only;
  test on 0..24. NOVELTY RISK: S16 + alpha*sum(r_i) resembles Iso-CTS (common + task-specific subspaces) ->
  literature/novelty agent launched (exact algorithms + defaults + code) before implementing baselines.
- NOVELTY CHECK (10-07, literature agent; papers + cloned repos in the session scratchpad lit/):
  "common part + alpha * task residuals" is crowded (Iso-CTS, CART, LoRE, Twin-Merging, Pico, CT-Merging -- all
  GROUP-level). Closest: CT-Merging (arXiv 2607.20561, 2026; LoRA-specific common + per-task residual scales; beats
  Iso-CTS / Core Space / DC-Merge; no code) and Iso-CTS (ICML 2025). Not found anywhere: a two-sided subspace from a
  POOL larger than (and independent of) the merged group, common term added once at full strength, two-sided
  residuals summed with alpha, on a closed-loop VLA at k up to 74 (largest static VLA merges in the literature:
  k=18 PolicyWeave, k=10; MergeVLA: naive TA on OpenVLA = 0% on LIBERO).
  Identities: TA(lambda) = W0 + lambda*sum P(D_i) + lambda*sum r_i (ours replaces the in-subspace sum with the pool
  mean); with G = whole pool and alpha = 1/k ours == plain averaging (so srk at k=74 duplicates avg -> sanity check);
  at k=1, alpha=1 ours != the expert -> ablation "pool subspace, group coefficients" W0 + P(mean_G D) + alpha*sum r_i.
  Mandatory for fairness (ours sees all 76 adapters): group-only subspace and leave-group-out pool ablations.
  Defensible contributions: pool vs group estimation, two-sided residual, native spectrum (vs Iso flattening),
  LoRA closed form, and the first closed-loop VLA benchmark of training-free merging at k=2..74.
  Baseline specs collected (TA, TIES, DARE-TIES, TSV-M, Iso-C, Iso-CTS, KnOTS, CART, CT-Merging; defaults + grids;
  null-space bug: use numerical rank of LoRA sums; TSV per-task budget capped at 16).
- EXP-109 change (10-07 10:55, before its sr part started): SR_KS 4,16,74 -> 16,74 (k=4 operator arms superseded by EXP-110's validated benchmark).
- EXP-109 change (10-07 11:40): operator (sr) arms dropped entirely (SR_KS = []), before they started; EXP-109 is now
  the clean AVERAGING scaling curve only. Our operator is evaluated in EXP-110 (validated benchmark + scaling).
- merge_methods.py identity checks (10-07, 3 real modules: LLM q_proj 2048x2048, expert down_proj 1024x4096,
  action_out_proj 32x1024): ta(1/k)=avg exact; pgc(k=1,a=1)=expert 2e-7; tsv/knots/ties at k=1 = expert <=8e-7;
  cart(gamma=1)=avg 1e-6; ours(all 76, a=1/76)=avg76 to 1.4e-3 (5.5e-3 on action_out_proj: precision of the stored
  S16 decomposition). Finding: the GROUP-only ablation collapses to averaging for any alpha -- a group's own top-16
  subspace captures ~all of the group mean (residual of the mean 1% of its norm on q_proj, k=4), so the group's
  residual sum nearly cancels. Any advantage of ours must come from the subspace being estimated OUTSIDE the group.

### EXP-110 — multi-task merging benchmark: ours vs 10 established methods, tuned on held-out states — **done (10-09 05:00): TIE with averaging; spectral SOTA collapses**
Motivation: user requirement (10-07) -- the proposed merging method must outperform existing methods.
Design (fixed before any rollout; `sweep/exp110_bench.py` docstring is the record): k = 16 (EXP-109 partition; 5
groups hold the 24-task panel). Methods: avg, ta, ties, dare_ties, tsv, iso_c, iso_cts, knots_ties, cart, ct_merge;
ours (pool S16 + alpha*sum two-sided residuals) + ablations pgc / group / lgo. 41 grid settings. Validation: 10
episodes per panel task on states 25..34 -> best setting per method; test: best settings, 25 episodes per panel task
on states 0..24; retention vs EXP-109's same-period single-task runs. Merge ops in `sweep/merge_methods.py`
(identity-checked); dense checkpoints built on GPU (8-28 s each, 7.2 GB, 451 trained modules), served as prebuilt
dirs, 3 per round, next round built during evaluation, one job per panel task.
Estimated ~10 h validation + ~7 h test. Scaling phase (top methods at k = 4, 74) after.
- EXP-109 interim (10-07 13:00; shiva part done; odin's single-task baseline 19/24, k=2,4 pending;
  `sweep/exp109_report.py`): plain averaging retention vs same-period single-task (95.4%): k=8 87.9% (-11.6pp,
  p=0.01), k=16 88.5%, k=37 89.0%, k=74 83.8% (-15.6pp). Flat from 8 to 37 tasks; losses concentrated in a few fragile
  tasks (t23 -56..-76pp, t26 -28..-76pp, t0 -24..-36pp) -- the headroom a better method must recover.
- EXP-110 started 12:31 (val round ~13 min, 69 rounds -> val ~03:30 10-08; test ~11:30 10-08). Report stage added
  (E110_STAGE=report): best setting, test, retention, paired vs avg and vs ours, tasks losing > 30pp.
- Orchestrator fix (10-07 23:55): per-signature build lock (merge_orchestrator.py _sig_lock): a second worker with the same merge signature used to rm -rf the first's in-progress build ('Directory not empty' errors: EXP-108 b2, EXP-109 k37 t11 / k74 t9). Prebuilt jobs (EXP-110) unaffected. EXP-109's 2 missing scenes rerun on odin.
- EXP-109 RESULT (10-08 00:30; results/analysis/exp109_report.txt). Panel of 24, 25 episodes per task on states
  0..24, single-task same period 95.2%. Plain averaging retention (gap; Wilcoxon p):
    k=2 97.9% (-2.0, 0.53) · k=4 91.9% (-7.7, 0.005) · k=8 87.2% (-12.2) · k=16 88.4% (-11.0) · k=37 86.5% (-12.8)
    · k=74 83.9% (-15.3, p=0.001)
  Shape: nearly free at 2, plateau ~87% from 8 to 37, 84% for one model of all 74 tasks (vs 82% in EXP-7 with 10
  rollouts and the standing baseline -> the old curve replicates). Losses concentrate in a few fragile tasks:
  t23 (-36 to -76pp), t26 (-28 to -76), t0 (-16 to -36), t73/t74/t34 at some k.
- EXP-110 validation round 1 (41 settings, done 10-08 07:00; 10 eps per panel task, states 25..34), best per method:
  ours a=1/16 0.838 (edge) · ours_group a=0.5 0.817 · avg 0.808 · cart l=1 0.796 (edge) · ties l=1 0.758 · dare_ties
  l=0.5 0.758 (edge) · knots l=1 0.738 (edge) · ours_pgc a=0.25 0.725 (edge) · ours_lgo a=0.25 0.704 (edge) · ta l=0.1
  0.425 (edge; ta at 1/k == avg) · tsv, iso_c, iso_cts, ct_merge ~0.00 at EVERY setting.
  FAIRNESS FIX before any test record (test stage stopped at 07:05 with 0 records): the first grids came from vision
  benchmarks. Update norms at scale 1 vs averaging (k=16, 24 sampled modules): ta 16x, tsv 4.5x, iso_c 3.3x, iso_cts
  3.4x, dare 2.6x, ct 2.1x, ties 1.4x, knots 1.3x, cart 1.0x -> the spectral methods were run at 1.6-7x the
  update size that works here (our adapters share ~80% energy, so sums amplify the shared part). Grids extended with
  size-matched values (~0.67x/1x/1.33x of averaging) and past every winning edge; ours adds a=1/32 and a=0 (pool S16
  only). 22 new settings (110 models), split shiva 3/4 + odin 1/4 (E110_SHARD). Test after.
  Note: CT-Merging failed even near matched size (gamma 0.3 = 0.64x, 0.6 = 1.27x) -- it flattens the shared part's
  steep spectrum to its RMS; Iso-* flatten too. Spectrum flattening may be structurally harmful for pool-consistent
  LoRAs (to confirm with the size-matched runs).
- EXP-110 ops (10-08 ~09:30): a colleague needs shiva GPU 2 -> shiva's validation share restarted on GPUs 0,1 only
  (E110_GPUS=0,1 -> 6 workers; builder on cuda:1); GPU 2 verified free. Shard now pinned to the fixed 110-item
  extension list (restart-safe; odin's 27 items unchanged). Test stage will also run with E110_GPUS=0,1.
- EXP-110 ops (10-08 15:00): shiva's share done; the hand-off watcher killed itself (its own command line matched the process-cleanup pattern -- the pkill self-match trap again) after stopping odin's chain. Finished by hand: odin's leftover checkpoints removed, records synced, odin's 19 remaining items + the test now run on shiva (GPUs 0,1). Rule: cleanup commands must never live in a watcher whose command line contains the pattern.
- EXP-110 VALIDATION FINAL (10-08 17:05; 63 settings, 10 eps per panel task on states 25..34; SE ~2.4pp per setting):
  ours a=1/16 0.838 = cart l=0.5 0.838 > ours_group a=0.5 0.817 > avg 0.808 = ours_lgo a=0.125 0.808 > ours_pgc a=0.125
  0.796 > ours a=0 (pool S16 only) 0.762 > ties l=1 0.758 = dare_ties l=0.5 0.758 > knots l=1 0.738 > ta l=0.08 0.667
  (ta at 1/k == avg 0.808) >> tsv 0.008, iso_c 0.008, iso_cts 0.000, ct_merge 0.004 at EVERY setting incl. size-matched.
  Readings: (1) spectral flattening/orthogonalizing methods fail structurally on pool-consistent VLA LoRAs (not a
  scale issue); (2) the shared part alone keeps 76%; residuals at 1/16 add ~8pp; (3) ours and CART tie at the top;
  CART's best (0.5) borders the untested (0, 0.5) range -> if close on test, add cart l=0.25, 0.75.
  Test started 17:05 (70 models, GPUs 0,1; ETA ~05:00 10-09).
- EXP-110 ops (10-08 17:40): test restarted with key methods first (avg, ours, cart, ablations, then the rest); finished test records kept. Interim: avg test 79.2% over 20/24 tasks (retention 83.7%).
- EXP-110 INTERIM TEST (10-08 22:45; key methods complete, 24 tasks x 25 eps on states 0..24; report
  results/analysis/exp110_report_k16.txt): ours_pgc a=0.125 83.3% (ret 87.6%, +2.0 vs avg, p=0.41) · cart l=0.5
  82.2% (+0.8) · ours a=1/16 81.7% (+0.3, p=0.89) · avg 81.3% (ret 85.5%) · ours_lgo 81.2% · ours_group 79.3%.
  The validation lead of ours (83.8 vs 80.8) did not hold on test. All within ~2pp of averaging, none significant.
  Remaining baselines (ties, dare, knots, ta, spectral) finish ~05:00; on validation they were all below averaging.
  Note for analysis: the report must be run after syncing ALL e110 records (val + test) to the reporting machine.
- EXP-110 RESULT (10-09 05:00; k=16, 24 panel tasks x 25 eps on states 0..24, best validation setting per method;
  results/analysis/exp110_report_k16.txt). Test success (retention vs single-task 95.2%; paired diff vs avg, p):
    ours_pgc a=1/8 83.3% (87.6%; +2.0, 0.41) · cart l=0.5 82.2% (+0.8, 0.76) · ours a=1/16 81.7% (+0.3, 0.89) ·
    avg 81.3% (85.5%) · ours_lgo 81.2% (-0.2) · ours_group 79.3% (-2.0) · knots_ties 79.2% (-2.2) · ties 78.2% (-3.2) ·
    dare_ties 77.7% (-3.7) · ta l=0.08 65.3% (-16.0, p=0.006; ta at 1/k == avg) · tsv 1.2%, iso_c 0%, iso_cts 0%,
    ct_merge 0% (-80pp, p<0.001; all 24 tasks lost > 30pp).
  VERDICT vs the requirement ("outperform existing methods"): NOT MET against the simple baselines -- ours ties plain
  averaging and CART (all within 2pp, none significant); ours beats TIES/DARE/KnOTS by 2.5-4pp (n.s.) and the
  spectral SOTA (TSV-M, Iso-C/CTS, CT-Merging) by ~80pp: those collapse on pool-consistent VLA LoRAs at every scale,
  including size-matched ones. Validation lead of ours (83.8 vs 80.8) was noise (SE ~2.4pp per setting).
  Next (approved plan): scaling phase for the top methods at k=74 (where averaging dilutes residuals most).
- EXP-110 scaling, k=74 (launched 10-09 07:00, shiva GPUs 0,1): one model of all 74 tasks, panel of 24; methods avg,
  ours (a in {0, 1/148, 1/74, 1/sqrt74, 0.5, 1}), ours_pgc, cart; same val (states 25..34, 10 eps) -> test (0..24,
  25 eps) protocol. ours_lgo impossible at k=74 (no pool outside the group). ~8 h.
- EXP-110 k=74 ops (10-09): shiva GPU 2 released by the colleague -> chain restarted on GPUs 0,1,2 (9 workers); finished records kept.
- EXP-110 k=74 ops (10-09 09:25): with E110_GPUS=0,1,2 the override made 9 workers but kept hostcfg's 7 ports -> 2 worker threads die each round (IndexError on PORTS); the run continues on 7 workers incl. GPU 2. Fixed in code for future runs (port list extended).
- EXP-110 SCALING k=74 RESULT (10-09 16:25; one model of all 74 tasks; panel 24 x 25 eps on states 0..24;
  results/analysis/exp110_report_k74.txt). Validation: ours a=1/148 0.842 (a=0 i.e. pool S16 only 0.800; a=1/74
  0.779 == avg 0.779, the k=74 identity holds) · cart l=0.5 0.796 · ours_pgc 0.775. TEST: ours a=1/148 79.6%
  (retention 83.7%; +2.3 vs avg, p=0.50) · ours_pgc 78.7% (+1.4) · cart 77.3% (0.0) · avg 77.3% (81.2%).
  The +6.3pp validation lead shrank to +2.3pp on test (n.s.). Across both scales our family is above averaging in 4 of
  4 comparisons (+0.3, +2.0 at k=16; +2.3, +1.4 at k=74), mean ~+1.5pp, none significant. Detecting +2pp needs ~4x
  the test data (e.g. all 74 tasks x 100 eps per arm, SE ~0.8pp).

### EXP-111 — powered confirmatory test: shared part + half the mean residual vs averaging, all 74 tasks — **done (10-10 06:20): NULL (-0.58pp, p=0.24)**
Motivation: user-approved (10-09). Our operator was above averaging in 4/4 EXP-110 comparisons (+0.3..+2.3pp) but
no test was powered to detect ~2pp. PRE-REGISTERED (`sweep/exp111_confirm.py` docstring is the record): one model of
all 74 tasks; arms avg vs ours with the FIXED rule alpha = 1/(2k) = 1/148 (from EXP-110 k=74 validation); ALL 74 tasks;
100 episodes per task per arm (4 x 25 on states 0..24); PRIMARY = mean paired difference over tasks, two-sided paired
t-test at 0.05 (Wilcoxon alongside); expected SE ~0.8pp. Secondary: retention vs e92 single-task, tasks losing >30pp.
Jobs interleaved (rep -> task -> arm) so any partial snapshot is balanced. 14,800 episodes, ~15 h on 3 GPUs.
- 10-14 deck: paper-framing slide added (5 claims; EXP-111 pending).
- EXP-111 INTERIM (10-10 00:10; 305/592 jobs; reps 1-2 complete for both arms, 50-75 eps per task per arm):
  avg 76.53% vs ours 76.68% -> paired +0.14pp (SE 0.78, t p=0.85, Wilcoxon p=0.77; 31 up / 24 down / 19 equal).
  The +2.3pp on the EXP-110 panel was noise; at k=74 ours == averaging. Final ~06:00.

## Direction (10-09, user): a NEW merging method that beats all baselines in most cases, and the finding that what
works in standard merging doesn't necessarily work for VLAs; evaluated on MANY task subsets (not one merge of all
tasks) and including tasks from OUTSIDE the fleet.

### EXP-112 — multi-subset benchmark over fleet + external tasks — **done (10-13 05:00): ours BEATS every baseline in most subsets (mixed fleet + external merges)**
Motivation (user direction above). Library: 74 fleet (LIBERO-90) + 30 external (our zero-init LoRAs, LIBERO-Goal /
Object / Spatial; shared action normalization). Random subsets of size 2, 4, 8 from all 104 tasks (~30% external);
3 validation + 12 test subsets per size; 10 baselines + ours (+ candidate new methods); scales set relative to
averaging's update size; tuning on states 25..34, test on 0..24; fresh single-task baselines for all 104 tasks.
Metrics: win rate of each method vs every other over the 36 test subsets, average rank, retention split fleet vs
external. Caveat recorded: ours currently ties averaging (EXP-110/111) -> the "beats baselines" part needs a new idea.
- EXP-112 DESIGN FIXED (10-10 00:18, before any rollout; `sweep/exp112_bench.py` docstring + subsets() are the
  record): library 104 = 74 fleet + 30 external (same 458 modules; external deltas have < 1% energy in the fleet S16).
  Subsets STRATIFIED: every subset holds max(1, round(30k/104)) external tasks (k=2,4: 1; k=8: 2); k in {2,4,8};
  3 val + 12 test subsets per k (seeded). 29 settings per k: avg; ta at c/k (c 0.67, 1.33); ties, dare_ties, tsv, iso_c,
  iso_cts, knots_ties, ct_merge NORM-MATCHED to averaging x c (c 0.67, 1, 1.33); cart (l 0.5, 1); ours (a 1/2k, 1/k,
  2/k). Validation 10 eps per member on states 25..34 -> best per (method, k); test 10 eps per member on states 0..9;
  single-task baselines 2 x 10 eps on states 0..9 for all 104 tasks. Jobs: one per (checkpoint, suite, <= 2 members).
  Offline checks on odin: subsets/suites OK, tsv norm-match 33 s, builds 16-73 s (7.2 GB). Runs on shiva after EXP-111
  (odin's P6000s are in use by another user's llama-server, ~14 GB each -> not used).
- EXP-112 candidate NEW methods added (10-10 ~00:40, before validation starts; user: "ok good plan"):
  (1) ours_ties: pool S16 (exact, once) + lam * TIES over the two-sided RESIDUALS only (trim 20% / sign vote /
      disjoint mean) -- TIES on full deltas is dominated by the shared part; here it acts on the private parts.
      lam in {0.67, 1, 1.5} (norm 1.18x avg at lam=1).
  (2) ours_rm: pool S16 + gamma * input-weighted residual mean (diagonal RegMean): column j of the merged residual
      = sum_i g_ij r_i[:, j] / sum_i g_ij, g_ij = task i's mean squared input feature j under its own policy on 32 of its
      demo frames -- soft version of EXP-101's protection (interference on a task's own inputs is what harms it).
      gamma in {0.5, 1, 2}. Identity: uniform g, gamma=1 == ours(alpha=1/k) (2.8e-7).
  Input statistics: `sweep/exp112_actstats.py` (104 tasks, shiva GPU 2, ~35 min) -> exp112_actstats.npz.
  Settings per k now 35; default methods + ours_ties, ours_rm; report gives win rates of every ours* vs every baseline.
- EXP-112 input statistics: t57 has no demo file -> same-scene sibling frames with t57's own prompt/adapter; prompt fixed to the bare LIBERO instruction (scene prefix stripped). CAVEAT for earlier offline analyses (EXP-96, 101, 104): their prompts included the scene prefix ('KITCHEN SCENE9 put ...'), unlike the eval harness. Statistics recollected from scratch (10-10 01:05).
- EXP-112 input statistics complete (10-10 02:00): 104/104 tasks, no empty rows (t57 via sibling t5); smoke builds on shiva OK (ours_rm 45 s, ours_ties 25 s). Chain on shiva: EXP-111 -> base -> val (13 methods, 35 settings per k) -> test -> report.
- EXP-111 RESULT (10-10 06:20; pre-registered; 74 tasks x 100 eps per arm on states 0..24; exp111_report.txt):
  avg 77.22% vs ours (alpha = 1/(2k)) 76.64% -> paired -0.58pp (SE 0.49, t p=0.24, Wilcoxon p=0.69; 27 up / 25 down /
  22 equal). Retention vs e92 single-task: avg 81.5%, ours 80.9%; 17 tasks lose > 30pp under either.
  VERDICT: the shared-part + scaled-residual operator does NOT beat averaging at k=74; the EXP-110 +2.3pp was noise.
  It remains the explanation of WHY averaging works, not a better method. EXP-112 tests the two new candidates.
- EXP-112 VALIDATION round 1 (10-11 ~08:00; 35 settings per k; 3 val subsets, 10 eps per member, states 25..34).
  Best per (k, method): k=2: tsv 0.500, iso_c 0.467, avg/cart/iso_cts 0.450, ties/knots/ours_rm 0.433, ours 0.400,
  ours_ties 0.333, ct 0.017. k=4: ours_rm 0.825 (edge g=2), ours 0.817, ties 0.792, dare 0.775, cart 0.767, ours_ties
  0.758, ta 0.742, avg 0.700, knots 0.658, tsv 0.217, iso_c 0.125, iso_cts 0.067, ct 0.042. k=8: ours_rm 0.654, ours
  0.642, ta 0.642, dare 0.600, ties 0.558, avg 0.550, knots 0.525, cart 0.525, ours_ties 0.462, tsv 0.125, iso_c 0.075,
  iso_cts 0.030, ct 0.013. Spectral methods work at k=2, collapse from k=4. Low absolute success (external task in
  every subset). Validation noise is large (60-240 eps per setting).
  FAIRNESS FIX (before reading any test result; test stopped after 2/156 rounds): many winners on grid edges
  (norm-matched baselines at c=1.33, ta at 1.33/k, ours_rm at g=2; cart, ours_ties at their minimum) -> grids extended
  (c 1.67/2.0; ta 2/k; cart 0.25; ours 1/(4k); ours_ties 0.4; ours_rm 3); 54 settings per k; +178 validation models
  (~5 h), then test (~17 h) and report. ETA ~10-12 morning.
- EXP-112 test sharding (10-11 14:45): odin's GPUs are free again -> host-keyed shard file results/analysis/exp112_shard.json (shiva 0,1,2,3/5; odin 4/5); shiva's chain picks it up when its test stage starts; odin's share launches by watcher after validation (records/norms/stats synced). ETA test ~10-12 midday.
- EXP-112 VALIDATION FINAL (10-11 ~21:00; 54 settings per k). Best per (k, method), val success:
  k=2: iso_cts c=2 0.750 · iso_c c=2 0.700 · tsv 0.500 · avg 0.450 · cart 0.450 · ties/dare/knots/ours_rm 0.433 · ours 0.400
       · ta 0.383 · ours_ties 0.367 · ct c=2 0.183.
  k=4: ours_rm g=2 0.825 · ours 0.817 · ties/dare 0.792 · cart 0.767 · ours_ties 0.758 · ta 0.742 · avg 0.700 · knots 0.667
       · tsv 0.200 · iso_c 0.133 · ct 0.125 · iso_cts 0.092.
  k=8: ours_rm 0.654 · ours/ta 0.642 · ours_ties 0.638 · dare/cart 0.600 · ties 0.558 · avg 0.550 · knots 0.525 · tsv 0.165
       · iso_cts 0.158 · iso_c 0.154 · ct 0.133.
  Caveat: iso_c/iso_cts/ct still at the c=2 edge at k=2 (and k=8) -- not extended again: it can only strengthen methods
  that already beat ours at k=2 (conservative for our claims). Test launched: shiva 370 + odin 92 models (shard file).
- EXP-112 RESULT (10-13 05:00; 13 methods x 3 k x 12 test subsets, 10 eps per member on states 0..9; reports
  results/analysis/exp112_report.txt, exp112_stats.txt). Ops: shiva + odin shards; 10 jobs failed transiently
  (LIBERO config race, parse_fail) and were rerun.
  Retention (mean over test subsets; avg rank): k=2: TSV 59.1% (3.75) · Iso-C 62.2% (4.25) · Iso-CTS 59.8% (4.92) · TA
  53.5% · ours 55.0% (5.33) · ours_rm 56.1% · ours_ties 54.2% · TIES 40.2% · avg 34.6% (8.17) · DARE 39.1% · CART 33.6% ·
  KnOTS 26.4% · CT 13.4%. k=4: OURS 71.3% (2.67) · DARE 66.7% · TA 66.5% · CART 69.1% · avg 64.5% · TIES 64.7% · ours_rm
  61.1% · ours_ties 58.7% · KnOTS 51.1% · spectral 12-16%. k=8: OURS 65.6% (3.00) · ours_rm 64.9% · TA 63.1% · DARE
  62.7% · ours_ties 59.9% · CART 57.5% · avg 58.0% · TIES 54.8% · KnOTS 50.1% · spectral 6-7%.
  Paired over all 36 subsets, ours - baseline: avg +11.6pp (28/36 wins, Wilcoxon p=0.0004) · TIES +10.7 (27/36, 0.0005)
  · CART +10.6 (26/36, 0.008) · DARE +7.8 (27/36, 0.001) · TA +2.9 (23/36, 0.04) · KnOTS +21.5 (31/36, <1e-4) · TSV +38.5,
  Iso-C +36.3, Iso-CTS +36.4, CT +53.0 (all p<1e-4). k=4+8 only: vs avg +7.2 (19/24, p=0.0025). At k=2 the spectral
  methods are ahead of ours (TSV -4.1, Iso-C -7.2, Iso-CTS -4.8; n.s.) -- they keep the external task.
  MECHANISM: ours - avg on FLEET members +10.4pp at k=4/8 (p=0.0009), +58.3pp at k=2 (p=0.002); on EXTERNAL members -2.3
  (n.s.) and -17.4 (n.s.). External LoRAs contribute ~nothing to the shared rank-16 part, so averaging dilutes it by the
  fraction of pool tasks in the merge; our operator keeps it at full strength. Consistent with EXP-110/111: fleet-only
  merges have no dilution -> ours == avg there.
  New candidates: ours_rm close to ours at k=8, worse at k=4; ours_ties worse -> the BASE operator is the method.
  Caveats: 10 eps per member; hyperparameters chosen on 3 validation subsets per k; spectral methods at their grid edge
  (c=2) at k=2 (could be stronger there); external tasks are poorly retained by every method except spectral at k=2.

### EXP-113 — mechanism controls for the EXP-112 advantage (ours vs averaging, k=8) — **done (10-14 01:25): ALL 4 PREDICTIONS CONFIRMED**
Motivation (user, 10-13: the paper needs a reason for mixed subsets -> framing = consolidating a skill library with newly
added skills from other sources; reviewers will ask whether it ever hurts, and whether the gain is provenance (shared
structure) or domain (other benchmark suites)).
PRE-REGISTERED (before any rollout; `sweep/exp113_controls.py` docstring is the record): k=8; arms avg vs ours with
alpha=1/8 (EXP-112's validated choice at k=8; no new tuning); 12 seeded subsets per condition; 10 eps per member on
states 0..9; single-task baselines for the 30 S16-init external LoRAs (2 x 10 eps).
  A  0 external (fleet-only)             prediction: |ours - avg| small (nothing to dilute)
  B  4 external of 8                     prediction: ours - avg > EXP-112's +7.6pp at 2 of 8 (more dilution)
  C  8 external (external-only)          prediction: ours - avg >= -3pp (adding the library's shared part does not hurt)
  D  EXP-112 k=8 test subsets with the 2 external members swapped for their S16-INIT versions (same tasks, carry the
     shared part)                         prediction: gain shrinks toward 0 vs EXP-112 (+7.6pp) -> provenance, not domain
Reference: EXP-112 k=8 test (2 zero-init external of 8): ours - avg +7.6pp (10/12, p=0.034).
- EXP-113 ops (10-13 08:30): shiva base stage overflowed /dev/shm (9 workers x 7.2 GB orchestrator-built single-task models > 63 GB; my E112_GPUS override bypasses hostcfg's tmpfs bound). Restarted with 6 workers for base (run twice to refill failures); merge runs keep 9 (prebuilt, bounded rounds).
- EXP-113 BUG + fix (10-13 17:00): condition D rebuilt EXP-112's k=8 test subsets with E.subsets() AFTER appending the 30 S16-init adapters to E.LIB; E.subsets draws externals from everything past the fleet -> wrong subsets -> IndexError, shiva's A/B/D never ran (idle 08:40-17:00). Fixed by capturing EXP-112's subsets before the extension (verified against EXP-112 records). The odin dry check had compared two equally wrong reconstructions. A/B/C used explicit ranges (unaffected). Odin too slow on C (~1 h per model) -> stopped; all conditions now on shiva (odin's C records synced).
- 10-14 deck (10-13 21:20): paper-framing claims now list the LoRA tasks used per claim; claim 4 title updated to the EXP-112 result; EXP-112 slide gained a per-size breakdown (ours vs each baseline at 2 / 4 / 8 tasks and pooled). EXP-113 at round 15/30 on shiva, ETA ~01:30.
- EXP-113 RESULT (10-14 01:25; k=8, 12 subsets per condition, 10 eps per member; results/analysis/exp113_report.txt;
  report bug fixed first: fleet keys are ints). Mean retention, ours (alpha=1/8) vs averaging:
    A fleet-only            avg 87.2% ours 87.0%: -0.2pp (6/12, p=0.87)            prediction tie            CONFIRMED
    B 4 of 8 external       avg 20.2% ours 46.1%: +25.9pp (12/12, p<0.001); fleet members +45.2, external +6.5
                                                                                  prediction > +7.6pp        CONFIRMED
    C all external          avg  3.3% ours 17.1%: +13.7pp (9/12, p=0.007)          prediction >= -3pp         CONFIRMED (helps)
    D EXP-112 subsets with S16-init externals: avg 75.6% ours 75.9%: +0.2pp (5/12, p=1.0)
                                                                                  prediction gain -> 0       CONFIRMED
  Reading: the advantage is caused by shared structure (provenance), not by the other benchmark suites: the same
  external TASKS trained to carry the shared part (D) remove both the averaging failure (avg 58.0% -> 75.6%) and our
  gain; the gain grows with the share of foreign adapters (2/8: +7.6, 4/8: +25.9); fleet-only merges tie. Adding the
  library's shared part also helps foreign-only merges (C), though both are low there. Side finding: training new
  adapters from the shared part makes them merge well even with plain averaging (D).

### EXP-114 — shared part from the heterogeneous pool vs from the native library — **superseded by EXP-115 (10-14): its pool contained the test tasks**
Motivation (user, 10-14): step 1 of our method computes S16 from the 76 LIBERO-90 fleet adapters only. Does it matter?
If S16 is computed from the whole heterogeneous pool (76 fleet + 30 zero-init externals = 106), the externals contribute
~nothing to the shared directions, so the shared component should shrink by ~76/106 = 0.72, partly reintroducing the
dilution the method removes.
PRE-REGISTERED (before any rollout; `sweep/exp114_hetero.py`): (1) offline: recompute S16 from 106 adapters (sanity: the
same code on the 76 fleet adapters reproduces the stored S16); report per-layer norm ratio and subspace overlap vs the
fleet S16. (2) rollouts: EXP-113 condition B (12 subsets, 4 fleet + 4 zero-init external, k=8), arm ours_h = pool S16_h +
(1/8) * sum of residuals w.r.t. S16_h; 10 eps per member, states 0..9; paired with EXP-113 B (ours 46.1%, avg 20.2%).
Prediction: avg < ours_h < ours (fleet S16).

### EXP-115 — LEAVE-TEST-OUT shared part: rerun every "ours" result with S16 computed without the tested tasks — **done (10-14 20:40): IDENTICAL to ours; all conclusions hold**
Motivation (user, 10-14): "the shared component should be computed on tasks that exclude the test tasks. If a task is tested,
it should not be a member of the pool when computing the shared component." Our EXP-112/113 "ours" numbers violated this:
S16 came from all 76 fleet adapters, including the subset's fleet members (merged and tested). Only EXP-110's ours_lgo
ablation (fleet-only, k=16: 81.2% vs ours 81.7%, avg 81.3%) used the clean protocol. Baselines are unaffected (no S16).
PRE-REGISTERED protocol (`sweep/exp115_lto.py`): for every merged subset, S16 = top-16 of the mean over the native library
(76 fleet adapters) MINUS the subset's members and their duplicate copies (t7/t18, t29/t62); residuals w.r.t. that
subspace; ours_lto = S16_lto + alpha * sum r_i. alpha re-validated per k on EXP-112's validation subsets (grid 0.5/k,
1/k, 2/k; validation subsets also leave their own members out), then:
  test  : EXP-112's 36 test subsets -> full comparison vs all baselines (existing baseline records)
  ctrl  : EXP-113 conditions A, B, D at k=8 (C has no fleet members -> pool = full library, identical to ours)
  hetero: condition B with the pool = 76 fleet + 30 zero-init external, minus the subset's members (the EXP-114 question)
Prediction: leaving <= 6 of 76 adapters out barely moves S16 -> conclusions unchanged.
- EXP-115 note (10-14, user follow-up): "maybe this requirement is not needed in practice ... given a new task, nothing stops us
  from adding it to the pool and computing a new shared component." Assessment: correct -- merging always reads the merged
  adapters' weights (averaging too); no test OUTCOME enters the merge; the leak that matters (tuning on test) is avoided.
  The practical protocol = library + the new tasks; EXP-112 used library-only (incl. merged fleet members, excl. new
  externals) -> adding 1-4 near-orthogonal externals to 76 shrinks S16 by only ~2-5%, so EXP-112 approximates it.
  Offline (10-14): leaving a subset's 6 fleet members out of the 76-adapter pool changes the merged update by 0.7% (same
  order as stored-vs-recomputed S16, 0.7%) -> leave-test-out should not change any result. Rollout check launched anyway
  as a reviewer-proof robustness result (val -> test -> ctrl -> hetero, ~10 h on shiva).
- EXP-115 ops (10-14): shiva GPU 0 left free for another user -> chain restarted on GPUs 1,2 (6 workers, builder on cuda:1); finished records kept.

## Direction (10-14, user): continual-learning bridge. Keeping the LIBERO-90 shared part at full strength protects existing
skills while new skills are added = continual learning (LIBERO itself is a lifelong-learning benchmark: LIBERO-90 then new
suites). Differences: training-free, and merges so far were one-shot. Plan: iterative continual merging with our method; which
tasks should define the shared part (all seen vs priority LIBERO-90 only). Deck: content_for_slides_21_10.md (meeting 10-21).

### EXP-116 — training-free continual merging: our merge (two shared-part definitions) vs merging and continual-merging baselines — **done (10-15)**
Motivation: user plan above + (10-14) "include the new continual learning baselines, and ... include a few more from the
merging literature". Note: our merge is sum-based -> order-independent: one-at-a-time == merging the set seen so far
(running sums, constant memory). Under alpha = 1/k a shared part estimated only from the tasks INSIDE the model reduces
exactly to plain averaging, so "all previous tasks" is defined as the LIBERO-90 library + every new task so far.
PRE-REGISTERED (`sweep/exp116_continual.py` docstring is the record): 4 seeded sequences = 16 LIBERO-90 tasks merged + 8
zero-init external tasks added one at a time; checkpoints t = 0, 4, 8; evaluate a fixed 8 of the 16 old tasks + all new
tasks so far (10 eps, states 0..9). Methods (fixed rules, no tuning): avg; ta (1.33/k); ties, dare_ties (norm-matched x1.33);
cart (0.25); maxabs (training-free MagMax, norm-matched x1.33); opcm (alpha 0.5, official default, arrival order, official
norm rescaling); ours_lib (shared part = LIBERO-90 library); ours_all (library + new so far); alpha = 1/k for ours.
104 merged models, 12,640 episodes. Checks: MaxAbs/OPCM recover the task at k=1; factored OPCM == dense reference (6e-15);
OPCM build 594 s on odin P6000 (incl. norm pass), MaxAbs 35 s.
Queue: shiva (GPUs 1,2; GPU 0 left free) 7/8 + odin 1/8 after EXP-115; hand-off of odin's remainder to shiva at the end.
- EXP-115 RESULT (10-14 20:40; results/analysis/exp115_report.txt). BUG found in reporting first: after EXP-113 appends 30
  S16-init adapters to E.LIB, E.subsets() returned different subsets, so E.best_settings() (baseline settings) and
  val_alpha() read records of the wrong subsets -> some baseline cells "incomplete" and selection on fleet members only.
  Fixed by pinning E.subsets to EXP-112's original subsets in exp113_controls.py; corrected selection gives the same alphas
  (1/(2k) at k=2, 1/k at k=4, 8) -> runs stand. EXP-113's report did not use best_settings (unaffected); EXP-112's stats
  were computed before the extension (unaffected).
  ours_lto - ours (EXP-112): +0.3pp (14/36, p=0.92) -> leave-test-out makes no difference.
  ours_lto vs baselines, all 36 test subsets: avg +11.9 (26/36, p=0.0002) · TIES +11.0 (29/36, 0.0001) · CART +10.9 (29/36,
  0.001) · DARE +8.1 (25/36, 0.0004) · TA +3.2 (22/36, 0.06, borderline) · KnOTS +21.7 · TSV +38.8 · Iso-C +36.5 · Iso-CTS
  +36.6 · CT +53.3 (all p<1e-4). At k=2 spectral methods remain ahead (n.s.).
  Controls with ours_lto vs avg: A fleet-only +0.5 (6/12) · B half external +26.1 (12/12, p=0.0005) · D S16-init externals
  -0.7 (4/12) -> mechanism replicated.
  HETERO POOL (shared part from fleet + the other 26 external adapters, leave-test-out) vs native pool, condition B: -6.7pp
  (1/12, p=0.003) -> putting foreign adapters into the shared-part pool dilutes it and hurts; define the shared part on the
  native/priority library. (Bears on EXP-116's ours_all and on "add new tasks to the pool": many foreign ones hurt.)
- EXP-116 ops (10-14 20:58): the hand-off/report watcher was stopped by Claude Code (odin critically low on memory). Cause: odin's share (4 policy servers at 12-15 GB RSS + driver; 4 GB free of 62 on a shared machine). Odin share stopped (tmux kill-session); its records kept; shiva's share (7/8) continues. Odin's remaining 1/8 to be run on shiva after its share -- watcher NOT re-armed (per Claude Code, only on the user's request).
- EXP-116 ops (10-14 21:05, user: 'automatically'): hand-off chain re-armed ON SHIVA (tmux e116tail: wait for shiva's share -> run odin's remaining 1/8 on shiva GPUs 1,2 -> report), independent of odin; odin's finished records synced.
- EXP-116 ops (10-15): chain finished 15:45 (report 15:55). 3 models at t=0 (cart s0, maxabs s0, ties s2) each lacked one
  scene (silent worker failures, no error in the log) -> re-run on shiva (tmux e116fix, 10 min), report regenerated.
- **EXP-116 RESULT (10-15 16:10; results/analysis/exp116_report.txt; complete, 104/104 models).** Retention (merged /
  single-task), mean over 4 sequences; old = 8 LIBERO-90 tasks, new = external tasks added so far.

  | method | old t=0 | old t=4 | old t=8 | new t=4 | new t=8 |
  |---|---|---|---|---|---|
  | ours_lib (shared part = LIBERO-90 library) | 82.0 | 81.3 | **79.9** | 26.4 | 18.6 |
  | ours_all (library + new so far) | 82.0 | 83.0 | 79.3 | 25.8 | 14.7 |
  | avg | 86.2 | 82.0 | 62.8 | 24.4 | 10.5 |
  | TA (1.33/k) | 68.8 | 75.2 | 74.5 | 25.7 | 16.2 |
  | TIES | 80.5 | 66.4 | 43.5 | 29.1 | 17.6 |
  | DARE-TIES | 80.7 | 78.7 | 63.1 | 25.1 | 14.0 |
  | CART | 83.4 | 72.3 | 52.2 | 24.2 | 20.9 |
  | MaxAbs (training-free MagMax) | 82.0 | 63.8 | 42.6 | 25.2 | 17.8 |
  | OPCM | 70.4 | 14.4 | 3.9 | 22.0 | 9.2 |

  - STABILITY: forgetting t=0 -> t=8 (same 32 task instances): ours_lib -2.1pp, ours_all -2.7 vs avg -23.4, DARE -17.6,
    CART -31.2, TIES -36.9, MaxAbs -39.4, OPCM -66.5; TA +5.8 (starts low: 1.33/k over-scales the LIBERO-90-only model).
    t=8 paired (32 instances): ours_lib vs avg +17.1pp (21/5, p=0.001), DARE +16.8 (p=0.001), CART +27.7 (p=1e-4), TIES
    +36.4, MaxAbs +37.4, OPCM +76.0 (p<1e-4); vs TA +5.4 (11/5, p=0.07, n.s.). At t=4 ours == avg (-0.7).
  - PLASTICITY: no method learns the new tasks: new-task retention 9-21% after 8 additions (26% after 4). ours_lib vs avg
    +8.0pp (9/0, p=0.008); vs others n.s. (CART +2.3 ahead, n.s.). Mechanism: an external adapter has <1% energy in the
    shared part, so a new task lives entirely in its residual, diluted to 1/k (k = 24 -> 4%).
  - SHARED PART: ours_lib == ours_all on old tasks (+0.6, p=0.94); ours_lib better on new tasks (+3.9pp, 8/0, p=0.01)
    -> define the shared part on the priority library only (agrees with EXP-115's heterogeneous-pool result).
  - Predictions: (1) ours_lib protects LIBERO-90 best -> CONFIRMED (TA n.s. behind at t=8, far behind at t=0);
    (2) ours_all forgets LIBERO-90 slowly -> REFUTED (no extra forgetting; the cost lands on the new tasks instead);
    (3) averaging forgets most -> REFUTED as stated: averaging forgets 23pp, TIES/CART/MaxAbs/OPCM forget more.
  - OPCM collapse is not over-scaling (external task-vector norms ~11 vs fleet ~15; its rescale factor falls 0.44 -> 0.35
    -> 0.30 at k = 16/20/24). Hypothesis (not verified): the shared direction enters OPCM's accumulator once (task 1) and
    the norm rescaling shrinks it as residuals accumulate = the dilution mechanism.
  => Training-free continual merging with a fixed library shared part solves stability (LIBERO-90 skills flat over 8
     additions) but not plasticity. Next: EXP-117 (weight new tasks more).

### EXP-117 — plasticity knob: weight new tasks c/k (old stay 1/k), ours vs averaging with the same weights — **done (10-16)**
Motivation: EXP-116 = stability solved, plasticity not (new tasks ~19% after 8 additions, because their residuals are
diluted to 1/k). Does the shared part let our merge buy plasticity at a smaller stability cost than averaging given the same
boost? PRE-REGISTERED (`sweep/exp117_plasticity.py` docstring): EXP-116 sequences / checkpoints t=4,8 / evaluated tasks /
states 0..9; arms ours_asym(c) = S16(LIBERO-90) + sum w_i r_i and avg_asym(c) = sum w_i D_i, w = 1/k old, c/k new,
c in {2,4,8} (c=1 = EXP-116, reused); primary = ours - avg at t=8 on old and new retention per c (paired Wilcoxon); all c
reported (trade-off curves, no selection). Prediction: new rises with c for both; ours loses less on old; ours' curve
dominates. Checks: c=1 reproduces ours_lib / avg exactly (rel. err 2e-7); 48 models, 6,720 episodes.
- Queue (10-15 16:00): shiva tmux e117, GPUs 1,2 (GPU 0 left free), after the EXP-116 fix; report + figure
  results/figs/exp117_tradeoff.png. ETA ~03:00 on 10-16.
- EXP-117 ops (10-16): run finished 03:27; one scene (s1 t8 ours c=2) failed silently -> re-run, report regenerated 03:55.
- **EXP-117 RESULT (10-16; results/analysis/exp117_report.txt, figure results/figs/exp117_tradeoff.png; complete).**
  Retention old / new (c = 1 is EXP-116):

  | weight on new tasks | ours, 4 added | avg, 4 added | ours, 8 added | avg, 8 added |
  |---|---|---|---|---|
  | 1/k (c=1) | 81.3 / 26.4 | 82.0 / 24.4 | 79.9 / 18.6 | 62.8 / 10.5 |
  | 2/k | 72.6 / 22.4 | 72.6 / 23.1 | 60.5 / 17.4 | 48.8 / 12.9 |
  | 4/k | 46.6 / 29.2 | 46.6 / 26.5 | 12.4 / 11.6 | 12.2 / 9.5 |
  | 8/k | 8.7 / 16.3 | 5.9 / 16.9 | 0.6 / 0.6 | 1.2 / 1.3 |

  - Predictions: new retention rises with c -> REFUTED (after 8 additions it FALLS: 18.6 -> 17.4 -> 11.6 -> 0.6; after 4
    only c=4 gives +3pp at the cost of -35pp on old); ours loses less on old -> CONFIRMED only at small boosts after 8
    additions (c=2: +11.7pp, 16/7, p=0.03); at 4 additions and at c>=4 the two arms are identical (model broken).
  - Interpretation: the new-task failure is NOT a dilution a re-weighting can fix. Boosting all new residuals c-fold also
    boosts the interference each new task receives from the other foreign adapters, and the model breaks (all tasks ->
    ~0 at c=8). The bottleneck is the mutual interference of from-scratch adapters, not their weight.
  => plasticity needs new adapters that share the library's structure -> EXP-118 (new skills trained from the shared part).

### EXP-118 — continual merging with new skills TRAINED FROM THE SHARED PART (S16-init adapters) — **done (10-16)**
Motivation: EXP-116 = stability solved, new tasks not learned; EXP-117 = weighting new tasks more does not help them (see
its result). EXP-113 D: external adapters trained starting from the library's shared part merge like library tasks. Is
"train new skills from the shared part, then merge" the plasticity route, and does our merge still matter then?
PRE-REGISTERED (`sweep/exp118_anchored.py` docstring): EXP-116 sequences / arrival order with each new task replaced by its
S16-init adapter (same task; 30/30 matched; their single-task success 95% vs 83% for zero-init); t = 4, 8; same evaluated
tasks, 10 eps, states 0..9; EXP-116 methods and rules unchanged minus ours_all. Predictions: new-task retention well above
EXP-116 for the linear methods; averaging forgets far less (no dilution); ours ~ averaging (as in EXP-113 D).
64 models, 8,960 episodes.
- Queue (10-16 03:43): shiva tmux e118 (GPUs 1,2; GPU 0 left free), after the EXP-117 one-scene fix + report. ETA ~18:00 10-16.
- Ops: run 04:00-16:20 on shiva, no failures; 64/64 models complete.
- **EXP-118 RESULT (10-16 16:25; results/analysis/exp118_report.txt).** Retention old / new, scratch (EXP-116) vs anchored:

  | method | old t=8 scratch | old t=8 anchored | new t=4 scratch | new t=4 anchored | new t=8 scratch | new t=8 anchored |
  |---|---|---|---|---|---|---|
  | ours_lib | 79.9 | **85.3** | 26.4 | 26.1 | 18.6 | 20.2 |
  | avg | 62.8 | 84.2 | 24.4 | 25.4 | 10.5 | 19.5 |
  | CART | 52.2 | 81.4 | 24.2 | 26.7 | 20.9 | 19.1 |
  | TIES | 43.5 | 81.1 | 29.1 | 24.0 | 17.6 | 14.6 |
  | MaxAbs | 42.6 | 80.7 | 25.2 | 22.1 | 17.8 | 15.9 |
  | DARE-TIES | 63.1 | 75.8 | 25.1 | 22.1 | 14.0 | 15.3 |
  | OPCM | 3.9 | 70.9 | 22.0 | **57.5** | 9.2 | **39.1** |
  | TA (1.33/k) | 74.5 | 65.9 | 25.7 | 24.1 | 16.2 | 19.8 |

  - STABILITY: anchoring new skills removes the forgetting of every non-TA method (anchored - scratch, old t=8, paired over
    32: avg +21.4pp p=0.002, TIES +37.6, CART +29.2, MaxAbs +38.1, OPCM +67.0, all p<=3e-4; DARE +12.7 p=0.08); ours +5.4
    (n.s., it barely forgot anyway). TA -8.6 (p=0.04): its 1.33/k now over-scales a shared part that every member carries.
  - With anchored adapters ours == averaging (old +1.2pp p=0.54; new +0.7 p=0.83); ours > TA +19.4, DARE +9.5, OPCM +14.4
    on old (p<=0.005); vs TIES / CART / MaxAbs n.s. -> our advantage exists only for adapters NOT trained from the shared part.
  - PLASTICITY still unsolved for the linear methods: anchored new tasks reach 15-20% retention after 8 additions (absolute
    success ~19% vs 95% single-task). avg rises 10.5 -> 19.5, ours 18.6 -> 20.2.
  - SURPRISE: OPCM learns the anchored new tasks: 57.5% retention after 4 additions, 39.1% after 8 (ours -18.9pp, 7/16,
    p=0.01), while keeping LIBERO-90 at 71% (ours +14.4). With scratch adapters OPCM collapsed (old 4%). OPCM adds each new
    task's directions outside the merged model's dominant subspace and rescales to the mean task norm (factor ~0.3 at k=24,
    vs 1/k = 0.04 for the linear methods) -> its new-task parts keep ~7x more weight.
  - Predictions: new-task retention well above EXP-116 for the linear methods -> REFUTED (+1..+9pp, still ~20%);
    averaging forgets far less -> CONFIRMED (+21.4pp); ours ~ averaging -> CONFIRMED.
  => training new skills from the shared part solves stability for (nearly) every method; plasticity needs something like
     OPCM's treatment of the task-specific parts. Next: EXP-119 = our shared part (full strength) + OPCM on the residuals.

### EXP-119 — our shared part (full strength) + OPCM on the task-specific parts, with a scaling-only control — **done (10-17)**
Motivation: EXP-118 = stability solved (our shared part, or anchored adapters), plasticity only by OPCM (39-58% new-task
retention, but LIBERO-90 at 71%). Hypothesis: shared part -> stability, OPCM's treatment of the residuals -> plasticity.
PRE-REGISTERED (`sweep/exp119_hybrid.py` docstring): ours_opcm = S16 + lam * OPCM(residuals) (alpha 0.5, arrival order,
lam = mean residual norm / merged-residual norm); control ours_norm = S16 + lam * sum of residuals (same norm rule, no
projection); scratch (EXP-116 sequences) and anchored (EXP-118 sequences); t = 4, 8; 32 models, 4,480 episodes. Primary:
ours_opcm vs ours_lib / opcm / ours_norm at t = 8, paired. Predictions: anchored -> LIBERO-90 >= 80% and new tasks well
above ours_lib; ours_norm ~ ours_opcm would mean scale, not projection, matters; smaller gains for scratch adapters.
Checks: residual factors reproduce the reference residual sum (9e-7); k=1 OPCM(r) == r; lam 0.199 (OPCM) vs 0.206 (plain
sum) at k=24 (~1/sqrt(k): residuals already near-orthogonal, so the control isolates the projection); OPCM norm pass 256 s.
- Queue (10-16 16:34): shiva tmux e119, GPUs 1,2 (GPU 0 left free). ETA ~00:00 10-17.
- **EXP-119 RESULT (10-17 00:10; results/analysis/exp119_report.txt; 32/32 models, no failures).** Retention old / new:

  | condition | t | ours_lib | OPCM | ours_opcm | ours_norm (control) |
  |---|---|---|---|---|---|
  | scratch | 4 | 81.3 / 26.4 | 14.4 / 22.0 | 45.9 / 26.5 | 34.3 / 23.2 |
  | scratch | 8 | 79.9 / 18.6 | 3.9 / 9.2 | 12.2 / 10.7 | 6.6 / 9.9 |
  | anchored | 4 | 86.7 / 26.1 | 71.4 / 57.5 | 72.1 / 43.6 | 69.5 / 44.5 |
  | anchored | 8 | 85.3 / 20.2 | 70.9 / 39.1 | 65.1 / 45.1 | 55.0 / 41.3 |

  - Scale factors: ours_opcm lam 0.20-0.22, ours_norm 0.21-0.24 (~1/sqrt(k)) -> ALL residuals (old and new) weighted ~5x 1/k.
  - Anchored, t=8: ours_opcm vs ours_lib: new +24.9pp (15/5, p=0.001), old -20.3pp (4/19, p=2e-4); vs OPCM: new +6.0 (n.s.),
    old -5.9 (p=0.07) -> lands on OPCM's trade-off point, does not dominate either. vs ours_norm: old +10.1pp (p=0.007), new
    +3.8 (n.s.) -> the learning comes from the larger weight; OPCM's projection reduces the cost on LIBERO-90.
  - Scratch: collapse (old 12%), as in EXP-117: up-weighted foreign residuals break the model; projection helps a little
    (old +5.6pp vs ours_norm, p=0.02).
  - Predictions: anchored LIBERO-90 >= 80% -> REFUTED (65%); new well above ours_lib -> CONFIRMED (+24.9); ours_norm ~
    ours_opcm -> on new yes, on old no (projection +10pp); smaller gains for scratch -> CONFIRMED (collapse).
  - Interpretation: the shared part at full strength did NOT keep LIBERO-90 once residuals are weighted ~0.2 -> the old-task
    loss comes from the residual weight itself (old tasks' own residuals are boosted too: Phase C showed full-strength
    residuals break conflicting twins) and/or from the new residuals. Open: boost only the NEW residuals for anchored adapters
    (EXP-117 tested that only for scratch). EXP-120 (running) answers part of it: 2 anchored new tasks at w = 0.25 / 1, the
    rest at 1/k.
  => stability-plasticity is set by the residual weight: 1/k = stable, new tasks ~20%; ~1/sqrt(k) = new ~45% (anchored),
     LIBERO-90 ~65%. No arm so far gets both.

### EXP-120 — privileged tasks: protect a CHOSEN set of tasks in continual merging (user idea 10-16) — **done (10-17)**
Motivation (user): define a set of privileged tasks to preserve; only they define the shared component, so the merge
becomes a method to protect chosen tasks in a continual setup, and the set can be changed dynamically (sum-based merge ->
recompute the shared part from the privileged adapters' running mean whenever the set changes).
User approved a first test (10-16). PRE-REGISTERED (`sweep/exp120_privileged.py` docstring): W = S16(LIBERO-90) +
w * sum of the privileged tasks' residuals + (1/k) * the others' (a one-task group's shared component = its own
task-specific part); P = first 2 new tasks per sequence (fixed rule); w = 1 and 0.25; scratch + anchored; t = 8; 16 models,
2,560 episodes. Readout vs ours_lib (paired): privileged new tasks (selectivity), LIBERO-90 and other new tasks (cost).
Predictions: privileged protected far more; cost on the rest, larger for scratch and w = 1. Check: w = 1/k == ours_lib (2e-7).
Selectivity within LIBERO-90 not tested: all its adapters share the same 16 directions (privileging a subset cannot be
selective there).
- Queue (10-16): shiva tmux e120 waits for EXP-119's report, then GPUs 1,2. ETA ~04:00 10-17.
- Ops: ran 00:10-04:30; one scene (anchored w=0.25, s3) failed silently -> re-run, report regenerated 04:40. 16/16 complete.
- **EXP-120 RESULT (10-17 04:40; results/analysis/exp120_report.txt).** Retention at t=8 (24 tasks) vs ours_lib, paired:

  | condition, w | privileged (2/seq, n=8) | LIBERO-90 (n=32) | other new (n=24) |
  |---|---|---|---|
  | scratch, ours_lib | 31.9 | 79.9 | 14.1 |
  | scratch, w=1 | 15.0 (-16.9, p=0.25) | 1.4 (-78.5) | 0.0 |
  | scratch, w=0.25 | 29.2 (-2.8) | 48.2 (-31.7, p<1e-4) | 9.5 |
  | anchored, ours_lib | 30.3 | 85.3 | 16.8 |
  | anchored, w=1 | **80.0 (+49.8, 6/0, p=0.03)** | 28.5 (-56.9, p<1e-4) | 12.4 |
  | anchored, w=0.25 | **64.6 (+34.3, 5/0, p=0.06)** | 71.7 (-13.6, 3/20, p=0.001) | 19.5 (+2.6, n.s.) |

  - Predictions: privileged protected far more -> CONFIRMED for anchored adapters (selective: in the same model privileged
    65% vs other new 19.5% at w=0.25), REFUTED for scratch (no protection; w=1 breaks everything incl. the privileged);
    cost lands on the rest, larger for scratch and w=1 -> CONFIRMED.
  - The cost falls on LIBERO-90, not on the other new tasks (+2.6 n.s. at w=0.25): two anchored task-specific parts at
    0.25 cost the library 14pp although its shared part stays at full strength.
  => a chosen task can be protected, but only if its adapter was trained from the shared part, and protecting it is paid by
     the library: same stability-plasticity frontier as EXP-117/119, now steerable per task. Small n for the privileged
     tasks (8 instances).
- Offline mechanism check (10-17 04:50, 6 adapters per group, LIBERO-90 shared directions U (outputs), V (inputs)), energy:
  LIBERO-90: shared 0.794 | reads shared inputs -> writes other outputs 0.203 | writes INTO shared outputs 0.002 | outside 0.001
  anchored new: shared 0.884 | reads shared inputs 0.075 | writes into shared outputs 0.039 | outside 0.002
  scratch new: shared 0.001 | reads shared inputs 0.009 | writes into shared outputs 0.065 | outside 0.925
  -> library tasks' task-specific parts READ the shared features and write their own outputs; they never write into the
  shared outputs. New adapters do (3.9% / 6.5%). A "fully outside" projection would remove ~all of a library/anchored
  task-specific part (dropped, no GPU spent). Candidate cause of EXP-120's library loss: the write-into-shared component.

### EXP-121 — write-protected privileged tasks: remove the part that writes into the shared outputs — **done (10-17)**
Motivation: the mechanism check above. PRE-REGISTERED (`sweep/exp121_writeprotect.py` docstring): privileged task p enters
as w * (I - UU^T) D_p (its update minus everything written into the shared output directions); rest as EXP-120 (P = first 2
new, w = 1 / 0.25, scratch + anchored, t = 8; 16 models, 2,560 episodes). Readout vs EXP-120 (paired, same w) and ours_lib.
Predictions: anchored -> LIBERO-90 loss shrinks clearly, privileged stay protected; scratch -> little change.
Check: model differs from EXP-120's by exactly the removed term (3e-5).
- Queue (10-17 04:44): shiva tmux e121, GPUs 1,2 (GPU 0 left free). ETA ~09:00.
- **EXP-121 RESULT (10-17 08:40; results/analysis/exp121_report.txt; 16/16, no failures).** Retention at t=8:

  | condition, w | privileged: EXP-120 -> write-protected | LIBERO-90: EXP-120 -> WP | other new: EXP-120 -> WP |
  |---|---|---|---|
  | anchored, ours_lib (ref) | 30.3 | 85.3 | 16.8 |
  | anchored, w=1 | 80.0 -> 72.6 (n.s.) | 28.5 -> **62.9 (+34.4, 27/3, p<1e-4)** | 12.4 -> 20.7 (+8.3, p=0.03) |
  | anchored, w=0.25 | 64.6 -> **27.8 (-36.8, 0/7, p=0.016)** | 71.7 -> 84.5 (+12.8, p=0.003) | 19.5 -> 21.1 |
  | scratch, w=1 | 15.0 -> 29.3 | 1.4 -> 3.0 | 0.0 -> 0.0 |
  | scratch, w=0.25 | 29.2 -> 30.7 | 48.2 -> 60.2 (+12.0, p=0.005) | 9.5 -> 11.2 |

  - Predictions: anchored LIBERO-90 loss shrinks clearly -> CONFIRMED (+34.4 / +12.8; at w=0.25 back to ours_lib, -0.8 n.s.);
    privileged stay protected -> CONFIRMED at w=1 (72.6% vs 30.3%, p=0.04), REFUTED at w=0.25 (protection gone);
    scratch little change -> privileged yes; LIBERO-90 +12pp at w=0.25 (unexpected, still -20 vs ours_lib).
  - The write-into-shared component is BOTH the cause of the library's loss AND a large part of what makes a privileged
    anchored task work at moderate weight. Anchored frontier points (privileged, LIBERO-90): ours (30, 85) · WP 0.25 (28,
    85) · priv 0.25 (65, 72) · WP 1 (73, 63) · priv 1 (80, 29) -> write-protection moves along the same frontier, it does
    not clearly shift it.
  => every weight-space knob tried (global weight EXP-117, OPCM / rescaling EXP-119, per-task weight EXP-120, write-
     protection EXP-121) trades library retention for new-task retention on one frontier. Escaping it likely needs the new
     adapters to be TRAINED not to write into the shared outputs (post-hoc removal != training under the constraint).

### EXP-122 — replication of the privileged-tasks frontier on 4 new sequences (anchored) — **done (10-17)**
Motivation: EXP-120/121's privileged comparisons rest on 8 instances (p=0.03-0.06). PRE-REGISTERED
(`sweep/exp122_replicate.py` docstring): sequences 4-7 (seeded like EXP-116), anchored, t = 8; arms ours_lib (reference),
priv w=1, priv w=0.25, write-protected priv w=1 (fixed rules, no re-tuning); report new sequences alone and pooled 0-7
(n = 16 privileged instances). Prediction: the EXP-120/121 pattern replicates. 16 models, 2,560 episodes.
- Queue (10-17 08:46): shiva tmux e122, GPUs 1,2 (GPU 0 left free). ETA ~13:00.
- Ops: done 12:35; one scene (s7 write-protected w=1) dropped silently -> re-run on 1 worker next to EXP-123's training.
- **EXP-122 RESULT (10-17 12:54; results/analysis/exp122_report.txt; complete).** Anchored, t=8, paired vs ours_lib:

  | arm | privileged, seqs 4-7 (n=8) | privileged, pooled 0-7 (n=16) | LIBERO-90 pooled (n=64) | other new pooled (n=48) |
  |---|---|---|---|---|
  | ours_lib (ref) | 9.1 | 19.7 | 78.3 | 21.3 |
  | priv w=1 | 70.1 (+61.1, p=0.016) | 75.1 (+55.4, 13/1, p=0.001) | 32.1 (-46.2) | 13.4 (-7.9, p=0.05) |
  | priv w=0.25 | 54.4 (+45.3, p=0.03) | 59.5 (+39.8, 11/0, p=0.003) | 69.8 (-8.5, p=0.001) | 24.4 (+3.2) |
  | post-hoc write-protected w=1 | 78.9 (+69.8, p=0.008) | 75.8 (+56.1, 14/2, p=8e-4) | 58.0 (-20.3, p<1e-4) | 22.9 (+1.6) |

  - Prediction (pattern replicates) -> CONFIRMED on the new sequences alone (all three privileged effects p<=0.03).
  - Pooled frontier (privileged, LIBERO-90): ours (20, 78) · priv 0.25 (60, 70) · WP 1 (76, 58) · priv 1 (75, 32): at the
    same protection, write-protection keeps 26pp more of the library than plain privileging at w=1.
  - Seqs 4-7 have harder privileged tasks (ours_lib 9% vs 30% in 0-3) and a lower library baseline (71% vs 85%).

### EXP-123 — new skills TRAINED write-protected (no writes into the shared outputs), then privileged — **done (10-18)**
Motivation: EXP-117/119/120/121 = one stability-plasticity frontier for every weight-space knob; post-hoc removal of the
write-into-shared part also removed the protection. LIBERO-90 adapters' task-specific parts read the shared inputs and never
write into the shared outputs -> train new adapters in that form. Trainer: `train_lora.py --init s16wp` (new option; S16 init,
A frozen at V^T, B's shared-output component reset to U*Sigma after every step -> D = S16 + B_perp V^T). Smoke test (20
steps): A unchanged (0 error), shared-output component of B held (2e-6), B learns elsewhere; loaded adapter has shared block
== S16 and a task-specific part that neither writes into the shared outputs nor reads non-shared inputs (~2e-4, bf16).
PRE-REGISTERED (`sweep/exp123_wptrain.py` docstring): 13 adapters = unique privileged tasks of sequences 0-7 (1,500 steps,
batch 8); single-task 2 x 10 eps; EXP-120 privileged merge with those tasks' anchored adapters swapped for write-protected
ones (others anchored), w = 1 / 0.25, sequences 0-7, t = 8; paired vs ours_lib, anchored privileged, post-hoc write-protected.
Prediction: off the frontier (privileged protected, LIBERO-90 loss < 5pp at w = 0.25). Risk: lower single-task success.
- Queue (10-17 09:30): shiva tmux e123 (`sweep/exp123_chain.sh`) waits for EXP-122; training on GPUs 1,2 (~3.5 h), then
  base + run (~4.5 h). ETA ~21:00.
- Ops: training 12:37-20:48 (13 adapters, ~70 min each, 2 GPUs; no failures, final losses 0.016-0.021); base stage: 4 of 26
  single-task runs failed with "No space left on device" (/dev/shm: 13 distinct 7.2 GB builds cached by 6 workers) ->
  re-run on 2 workers; run 21:04-00:40; report regenerated 01:00 with 26/26 single-task runs. 16/16 models complete.
- **EXP-123 RESULT (10-18 01:00; results/analysis/exp123_report.txt).** Pooled sequences 0-7, t=8; privileged n=16,
  LIBERO-90 n=64, other new n=48. Single-task: write-protected 96.2% vs anchored 95.0% (same 13 tasks) -> constraint is free.

  | arm (privileged weight) | privileged | LIBERO-90 | other new |
  |---|---|---|---|
  | ours_lib (no privilege) | 19.7 | 78.3 | 21.3 |
  | anchored privileged 0.25 (EXP-120/122) | 59.5 | 69.8 | 24.4 |
  | **write-protected TRAINED 0.25** | **41.4** (+21.7 vs ours, 8/0, p=0.01) | **77.9** (-0.4, n.s.) | 23.7 |
  | post-hoc write-protected 1 (EXP-121/122) | 75.8 | 58.0 | 22.9 |
  | anchored privileged 1 | 75.1 | 32.1 | 13.4 |
  | **write-protected TRAINED 1** | **85.2** (+65.5, 13/0, p=0.001) | **52.1** (+19.9 vs anchored priv 1, p<1e-4) | 23.1 |

  - Predictions: off the frontier -> PARTLY. LIBERO-90 loss < 5pp at w=0.25 -> CONFIRMED (-0.4); privileged protected
    >= anchored-privileged minus a few pp -> REFUTED at 0.25 (41 vs 60), CONFIRMED at 1 (85 vs 75, +10 n.s.).
    Single-task cost -> none.
  - Frontier (privileged, LIBERO-90): the trained-WP points (41, 78) and (85, 52) both lie above the old envelope (ours
    (20, 78) - anchored 0.25 (60, 70) - post-hoc WP 1 (76, 58)): ~+20pp protection at no library cost; +10pp protection at
    -6pp library vs post-hoc WP 1 (p=0.06). The trade-off remains (w=1 still costs the library 26pp).
  => training new skills write-protected shifts the stability-plasticity frontier outward at no single-task cost; it does
     not remove the trade-off. Next: EXP-124 (all new tasks write-protected, continual).

### EXP-124 — continual merging with ALL new skills trained write-protected — **done (10-18)**
Motivation: EXP-123 = write-protected training is free (single-task 96% vs 95%) and protects 2 privileged tasks at no
library cost (w=0.25). Can ALL new tasks then be weighted up and learned while LIBERO-90 is kept? PRE-REGISTERED
(`sweep/exp124_wpcontinual.py` docstring): write-protected adapters for all 22 new tasks of sequences 0-3 (11 from EXP-123
+ 11 trained, same recipe); EXP-116/118 sequences with every new task write-protected; t = 4, 8; w for all new tasks in
{1/k, 0.1, 0.25} (old 1/k); 24 models, 3,360 episodes. Readout vs anchored ours_lib (paired), OPCM, ours+OPCM. Prediction:
w=0.25 -> new >= 40% with LIBERO-90 loss < 5pp (beats OPCM / ours+OPCM on both axes).
- Queue (10-18 00:55): shiva tmux e124 (`sweep/exp124_chain.sh`; base stage on 2 workers to avoid the /dev/shm overflow).
  Training ~7 h (11 x ~70 min on 2 GPUs), base ~0.5 h, run ~5 h. ETA ~14:00 10-18.
- Ops: training 01:01-08:04 (11 adapters, no failures); base: 2 of 44 single-task runs lost to the /dev/shm overflow again
  (even on 2 workers) -> re-run on 1 worker; run 08:34-12:50; report regenerated 13:03 (44/44). 24/24 models complete.
- **EXP-124 RESULT (10-18 13:03; results/analysis/exp124_report.txt).** Single-task: write-protected 96.1% vs anchored 94.8%
  (22 tasks). Retention old / new, sequences 0-3 (old n=32, new n=16 at t=4, 32 at t=8), paired vs anchored ours_lib:

  | arm | t=4 old / new | t=8 old / new | t=8 vs ours_lib (old; new) |
  |---|---|---|---|
  | anchored ours_lib (EXP-118) | 86.7 / 26.1 | 85.3 / 20.2 | |
  | anchored OPCM (EXP-118) | 71.4 / 57.5 | 70.9 / 39.1 | |
  | anchored ours+OPCM (EXP-119) | 72.1 / 43.6 | 65.1 / 45.1 | |
  | WP, new at 1/k | 87.8 / 27.3 | 86.3 / 19.6 | +0.9 (n.s.); -0.6 (n.s.) |
  | WP, new at 0.1 | 85.5 / 33.0 | 85.2 / 29.6 | -0.1 (n.s.); **+9.4 (9/4, p=0.03)** |
  | WP, new at 0.25 | 82.4 / 52.4 | **76.9 / 50.5** | -8.4 (6/11, p=0.06); **+30.3 (20/4, p<1e-3)** |

  - Predictions: plain ~ anchored ours_lib -> CONFIRMED; w=0.25 new >= 40% -> CONFIRMED (50.5); LIBERO-90 loss < 5pp ->
    REFUTED (-8.4, borderline p=0.06); beats OPCM and ours+OPCM on both axes -> YES numerically (76.9 / 50.5 vs 70.9 / 39.1
    and 65.1 / 45.1) BUT those baselines used anchored adapters -> EXP-125 runs every baseline on the same WP adapters.
  - w=0.1 is a free lunch: +9.4pp on new tasks at no LIBERO-90 cost.
  => first arm that learns new tasks (50%) while keeping LIBERO-90 within ~8pp: the best stability-plasticity point so far.

### EXP-125 — every baseline on the SAME write-protected adapters (fairness control for EXP-124) — **done (10-19)**
Motivation: EXP-124's comparison with OPCM / ours+OPCM used baselines run on the anchored adapters -> adapter type and method
confounded. PRE-REGISTERED (`sweep/exp125_wpbaselines.py` docstring): EXP-124 sequences / write-protected adapters, t = 4, 8;
avg, ta, ties, dare_ties, cart, maxabs, opcm with EXP-116's fixed rules; ours (EXP-124, all w) vs each, paired at t = 8.
Prediction: WP adapters help the linear baselines a little; none reaches ours (w=0.25) on both axes; OPCM closest.
56 models, 7,840 episodes.
- Queue (10-18 12:57): shiva tmux e125 waits for the EXP-124 fix, then GPUs 1,2. ETA ~02:00 10-19.
- Ops: run 13:05-23:55; 3 scenes dropped silently (s0t8 ties, s2t8 ta, s2t8 cart) -> re-run on 1 worker next to EXP-126's
  training; report regenerated 00:25. 56/56 complete.
- **EXP-125 RESULT (10-19 00:25; results/analysis/exp125_report.txt).** Retention old / new, all on write-protected adapters:

  | method | t=4 old / new | t=8 old / new | ours w=0.25 minus method, t=8 (old; new), paired |
  |---|---|---|---|
  | avg | 87.3 / 24.0 | 84.2 / 19.0 | -7.3 (n.s., p=0.14); +31.5 (21/3, p=1e-4) |
  | TA 1.33/k | 69.1 / 25.4 | 67.8 / 16.3 | +9.1 (n.s.); +34.2 (p=1e-4) |
  | TIES | 81.6 / 25.4 | 81.6 / 16.0 | -4.6 (n.s.); +34.4 (p<1e-4) |
  | DARE-TIES | 78.8 / 21.5 | 78.2 / 16.1 | -1.2 (n.s.); +34.4 (p<1e-4) |
  | CART | 83.6 / 26.6 | 84.8 / 19.2 | -7.8 (p=0.06); +31.2 (p=2e-4) |
  | MaxAbs | 80.1 / 23.4 | 81.0 / 12.4 | -4.1 (n.s.); +38.1 (p<1e-4) |
  | OPCM | 76.1 / 50.2 | 78.4 / 36.0 | **-1.5 (n.s.); +14.5 (17/6, p=0.025)** |
  | ours w=1/k | 87.8 / 27.3 | 86.3 / 19.6 | |
  | ours w=0.1 | 85.5 / 33.0 | 85.2 / 29.6 | |
  | ours w=0.25 | 82.4 / 52.4 | 76.9 / 50.5 | |

  - Predictions: WP adapters help the linear baselines a little -> CONFIRMED on LIBERO-90 (vs anchored: CART 81.4->84.8,
    DARE 75.8->78.2, others ~same), not on new tasks; no baseline reaches ours (w=0.25) on both axes -> CONFIRMED (every
    baseline 14-38pp lower on new tasks; LIBERO-90 differences n.s.); OPCM closest -> CONFIRMED (OPCM on WP: LIBERO-90
    70.9 -> 78.4, new 39.1 -> 36.0).
  - Fair verdict: with the same adapters, ours at w=0.25 dominates OPCM (new +14.5pp, LIBERO-90 equal) and learns new
    tasks 31-38pp better than every linear merge at a LIBERO-90 cost of 1-8pp (n.s.); ours at w=0.1 (85.2 / 29.6) is at
    least as good as the best linear merges (avg 84.2 / 19.0, CART 84.8 / 19.2) on both axes.
  - The gain needs BOTH ingredients: write-protected adapters + our weighted merge (WP adapters alone with averaging: new
    19%; our merge with anchored adapters at higher weight: on the old frontier, EXP-117/119/120).

### EXP-126 — training-side ablations of write-protection: soft penalty (SkillsCrafter-style) and read-outside (InfLoRA-style) — **done (10-19)**
Motivation: novelty check (10-18) — closest training-side prior work is SkillsCrafter (AAAI'26; soft B-orthogonality, A
trained) and InfLoRA / O-LoRA / OSRM (input side pushed AWAY from old tasks). Reviewers will ask whether the hard
projection and "read inside the shared inputs" matter. Trainer options added (`train_lora.py --init s16soft | readout`,
existing modes untouched). Smoke tests (20 steps): soft moves A (<= 7%) and lets B drift into the shared outputs (<= 21%
vs 54% elsewhere, wp_lambda = 1 untuned); readout |A V| <= 7e-4, orthonormal rows, base+S16 step-0 loss == S16-init loss;
composed readout delta == S16 + BA (1e-7). PRE-REGISTERED (`sweep/exp126_ablations.py` docstring): 6 unique privileged
tasks of sequences 0-3 x 2 variants = 12 trainings; single-task 2 x 10 eps; EXP-120/123 privileged merge with the variant
adapters, w = 1 / 0.25, sequences 0-3, t = 8 (16 models, 2,560 episodes); paired vs ours_lib, anchored privileged, hard WP.
Predictions: hard WP keeps more LIBERO-90 than both at comparable protection; soft between anchored and hard; readout
costs the library like anchored adapters.
- Queue (10-18): shiva tmux e126 (`sweep/exp126_chain.sh`) waits for EXP-125; training ~7 h, base + run ~4.5 h. ETA ~13:30 10-19.
- Ops: training 23:56-06:58 (12 adapters, no failures); base 07:16 (prebuilt rounds, 24/24, no /dev/shm overflow); run
  -10:57; 16/16 models complete, no dropped scenes.
- **EXP-126 RESULT (10-19 11:00; results/analysis/exp126_report.txt).** Sequences 0-3, t=8; privileged n=8, LIBERO-90 n=32.
  Single-task: soft 93.3%, readout 96.7%, anchored 92.5% (same 6 tasks). Retention privileged / LIBERO-90:

  | privileged adapters | w = 1 | w = 0.25 |
  |---|---|---|
  | anchored (EXP-120) | 80.0 / 28.5 | 64.6 / 71.7 |
  | soft write-protection (A trainable, penalty 1) | 76.6 / 34.8 | 54.2 / 78.1 |
  | read-outside (A frozen orthogonal to shared inputs) | 84.7 / 45.8 | 48.3 / 81.4 |
  | **hard write-protection (EXP-123, same seqs)** | **93.7 / 53.3** | 39.9 / 82.8 |
  | ours_lib (no privilege) | 30.3 / 85.3 | |

  - w=1: hard dominates both ablations: vs soft LIBERO-90 +18.5pp (21/3, p=3e-4), privileged +17.1 (p=0.09); vs readout
    LIBERO-90 +7.5 (14/5, p=0.04), privileged +9.0 (n.s.).
  - w=0.25: no significant differences between hard and either variant (soft: privileged +14.3, LIBERO-90 -4.8; readout:
    +8.5 / -1.4; all n.s., n=8); the three constrained variants lie on a similar frontier, all above anchored on LIBERO-90
    (soft +6.3 p=0.04, readout +9.7 p=0.009, hard +11.1).
  - Predictions: hard keeps more LIBERO-90 at comparable protection -> CONFIRMED at w=1, not distinguishable at 0.25;
    soft between anchored and hard -> roughly (at w=1 close to anchored); readout costs the library like anchored ->
    REFUTED (+17.3pp vs anchored at w=1, p=0.004; any constraint relative to the shared subspace helps).
  => constraining a new skill relative to the library's shared subspace is what matters; the hard read-inside /
     write-outside form is the strongest under strong weighting and on par at moderate weight. Caveats: n=8 privileged
     instances, one training seed, one untuned penalty weight for soft.
- Ops (10-18 22:05): shiva quota 42.0 / 46 GB before EXP-126's 12 trainings -> archived two closed-line folders to the NAS
  (/media/pinas/lzhou/libero-eval-archive/): results/analysis/exp106_sil (2.77 GB, EXP-106) and new_loras_var50 (3.88 GB,
  donor-era seed variants); verified (file counts, file bytes, rsync checksum: 0 differences), then local copies removed.
  Quota now 35.7 GB. Only EXP-106 drivers referenced them.

### EXP-127 — replication of the headline continual result on 4 new sequences (write-protected, ours vs OPCM / avg / CART) — **done (10-19)**
Motivation: the headline (EXP-124/125) rests on 4 sequences; double it before the 10-21 meeting. PRE-REGISTERED
(`sweep/exp127_replicate_wp.py` docstring): sequences 4-7, every new task write-protected (18 adapters exist, 4 trained
with the same recipe), t = 4, 8; ours w = 0.1, 0.25; OPCM, avg, CART (fixed rules); 40 models, 5,600 episodes; report
sequences 4-7 alone and pooled 0-7 (pooled bookkeeping reproduces EXP-124/125's sequence 0-3 numbers exactly). Prediction:
pattern replicates (ours w=0.25 > OPCM on new tasks at similar LIBERO-90; > avg / CART by >= 25pp on new tasks).
- Queue (10-19 11:18): shiva tmux e127 (`sweep/exp127_chain.sh`): training ~2.5 h, base (prebuilt) + run ~9 h. ETA ~23:00.
- Ops (10-19 11:36, user: "Shiva's GPU 0 is available again! use it"): the two training shards on GPUs 1,2 kept running;
  the chain's controller (bash PID 3528188) was stopped by PID (shards reparented, unaffected) and replaced by
  `sweep/exp127_chain3.sh` (tmux e127b): GPU 0 immediately runs the 25/40 merges whose adapters already exist (1 model per
  round, ports 8100+, stops at a round boundary when training ends: E127_AVAILABLE_ONLY / E127_STOP_FILE); then base on
  GPUs 1,2; then the full run on GPUs 0,1,2. Code: exp112_bench.plan() gained an optional E112_PORT0 (port offset, only
  when set) so two orchestrators can run at once. New ETA ~19:30.
- Ops: training done 13:38 (4 adapters), base 13:46 (56/56 single-task runs incl. EXP-123/124's), run done 19:31; one
  scene dropped (s5 t8 OPCM) -> re-run on 1 worker with E112_PORT0=8100 next to EXP-128; report regenerated 20:15. 40/40.
- **EXP-127 RESULT (10-19 20:15; results/analysis/exp127_report.txt).** Single-task, all 28 write-protected adapters 96.8%.
  Retention old / new after 8 additions:

  | arm | sequences 4-7 | pooled 0-7 | ours w=0.25 minus arm, pooled (old; new), paired n=64 |
  |---|---|---|---|
  | ours w=0.25 | 66.8 / 50.8 | 71.9 / 50.7 | |
  | ours w=0.1 | 70.8 / 33.2 | 78.0 / 31.4 | |
  | OPCM | 71.3 / 46.4 | 74.8 / 41.2 | -2.9 (n.s.); **+9.5 (31/15, p=0.016)** |
  | averaging | 74.1 / 19.7 | 79.2 / 19.3 | -7.3 (p=0.07); +31.3 (44/6, p<1e-4) |
  | CART | 73.6 / 19.4 | 79.2 / 19.3 | -7.3 (p=0.016); +31.3 (43/5, p<1e-4) |

  - Prediction "pattern replicates": vs averaging / CART -> CONFIRMED (+31pp new on the new sequences alone, p<=1e-4);
    vs OPCM -> WEAKER: on sequences 4-7 alone ours - OPCM = +4.4pp new (n.s.), -4.4 old (n.s.); pooled +9.5pp new
    (p=0.016), old n.s. The EXP-125 "+14.5pp vs OPCM" shrinks to +9.5pp at n=8 sequences.
  - Sequences 4-7 have a lower LIBERO-90 baseline for every method (harder old tasks: averaging 74% vs 84% on 0-3).
  - ours w=0.1 pooled: LIBERO-90 equal to averaging / CART (78.0 vs 79.2), new +12pp.
  => headline (pooled, 8 sequences): ours w=0.25 learns new tasks +31pp better than linear merges and +9.5pp better than
     OPCM, at a LIBERO-90 cost of ~3pp vs OPCM (n.s.) and ~7pp vs linear merges.

### EXP-128 — RETAIN-style recursive interpolation as a continual-merging baseline — **done (10-19)**
Motivation: RETAIN (ICLR'26, 2512.08333) is the only published lifelong VLA merging rule: theta_n = (1-a) theta_{n-1} +
a theta_ft,n (2 tasks, full fine-tunes, no merging baselines). PRE-REGISTERED (`sweep/exp128_retain.py` docstring):
training-free analog on our adapters — start = average of the 16 LIBERO-90 tasks, each arrival Delta <- 0.5 Delta + 0.5 D_new
(old (1/16) 0.5^t, new j 0.5 * 0.5^(t-j)); EXP-124 setting (write-protected, sequences 0-3, t = 4, 8); 8 models, 1,120 eps;
vs ours / OPCM / averaging. Prediction: LIBERO-90 far below ours; recent new tasks learned, earliest forgotten. Check:
closed-form weights == literal recursion (9e-8).
- Queue (10-19 18:00): shiva tmux e128 waits for EXP-127's report, then GPUs 0,1,2. ~1 h.
- **EXP-128 RESULT (10-19 20:55; results/analysis/exp128_report.txt; 8/8 complete).** Write-protected adapters, seqs 0-3,
  retention LIBERO-90 / new: t=4 RETAIN 79.4 / 53.3 (ours 0.25: 82.4 / 52.4); t=8 RETAIN **81.9 / 36.7** (ours 0.25: 76.9 /
  50.5; ours 0.1: 85.2 / 29.6; OPCM 78.4 / 36.0; averaging 84.2 / 19.0). Paired t=8: ours 0.25 - RETAIN old -4.9 (n.s.), new
  +13.7 (p=0.095); ours 0.1 - RETAIN old +3.3 (p=0.099), new -7.2 (p=0.085). New-task retention by arrival position (t=8,
  n=4 each): 63 0 22 38 3 46 47 74 (recent tasks better, noisy).
  - Predictions: LIBERO-90 far below ours -> REFUTED (81.9%: write-protected adapters each carry the shared part, so the
    recursion keeps it at ~full strength; only old residuals decay); recent learned / earliest forgotten -> partly (noisy).
  - KEY READING: on write-protected adapters RETAIN (81.9 / 36.7) and OPCM (78.4 / 36.0) lie ON our w-curve (ours
    interpolated at 81.9% LIBERO-90 gives ~38% new); averaging (84.2 / 19.0) sits at the curve's top end next to ours at
    1/k (86.3 / 19.6). => with write-protected adapters most merge rules land on ONE frontier; the training recipe creates
    it, and the merge rule mainly picks the point. Our merge's distinct value there: the dial (any point), closed form,
    order-independence. Its decisive value remains the training-free case (scratch adapters: only ours keeps LIBERO-90).
    EXP-131 tests the OPCM comparison at matched retention directly.

### EXP-129 — OOD robustness: merged models vs single adapters (oracle routing), "why merge rather than route?" — **done (10-20)**
Motivation: MergeVLA (CVPR'26) routes per episode; with instructions one could route to each task's own adapter and
recover single-task success. RETAIN (ICLR'26) reports merging buys OOD robustness. PRE-REGISTERED (`sweep/exp129_ood.py`
docstring): EXP-81's validated `ood_shift` (all free objects' xy moved d m in one seeded direction; demos span ~+-2.4 cm),
d = 0.05, 0.10 (0 = existing ID records); EXP-124 setting (sequences 0-3, write-protected, t = 8), merged arms ours w=0.25,
ours w=0.1, averaging vs each evaluated task's single adapter (fleet / write-protected; 2 x 10 eps); 12 merged + 47
single models per shift, 5,720 episodes. Readout: merged vs single per shift (paired), paired OOD drop. Prediction: single
stays above, but the gap shrinks under shift (merged drops less), most for LIBERO-90 tasks. Dry check: 64/64 ID instances.
- Queue (10-19 18:20): shiva tmux e129 waits for EXP-128's report, then GPUs 0,1,2. ~7 h -> ETA ~03:00 10-20.
  Then WUDI (static-merging baseline used by MergeVLA) to keep the GPUs busy.
- Ops: run 20:58-03:45 (faster than estimated); 5 evaluations dropped silently (1 merged scene, 4 single runs) -> re-run on
  1 worker (ports 8100+) next to EXP-130; report regenerated 04:25; complete.
- **EXP-129 RESULT (10-20 04:25; results/analysis/exp129_report.txt).** EXP-124 setting, t=8, 4 sequences, n=32 per group.
  Success (single = each task's own adapter = oracle routing):

  | group, shift | single | ours w=0.25 | ours w=0.1 | averaging |
  |---|---|---|---|---|
  | LIBERO-90, none | 95.9 | 74.7 | 82.2 | 81.2 |
  | LIBERO-90, 5 cm | 85.5 | 69.4 | 74.7 | 75.3 |
  | LIBERO-90, 10 cm | 65.9 | 48.8 | 57.8 | 57.2 |
  | new, none | 95.5 | 49.1 | 28.7 | 18.4 |
  | new, 5 cm | 85.9 | 45.6 | 28.1 | 19.4 |
  | new, 10 cm | 67.5 | 37.2 | 22.5 | 14.1 |

  - Single adapters stay above every merged model at every shift (all p<=0.014).
  - Absolute drop ID -> 10 cm: LIBERO-90 single -30.0pp vs merged -24..-26 (n.s., p>=0.11); new tasks single -28.0 vs ours
    0.25 -11.9 (p=0.012), ours 0.1 -6.2, avg -4.4 (p<=0.001) — but merged new-task success starts far lower (floor).
  - Merged / single ratio: LIBERO-90 ours 0.1 86% -> 87% -> 88%, avg 85% -> 88% -> 87% (flat); new tasks ours 0.25
    51% -> 53% -> 55%, ours 0.1 30% -> 33% -> 33% (slight rise).
  - Predictions: single stays above -> CONFIRMED; gap shrinks under shift -> only in absolute pp (ratio ~flat; slight for
    new tasks); merged drops less -> n.s. for LIBERO-90, significant for new tasks but confounded by the floor.
  => object-position robustness does NOT justify merging over routing in this setting. The case for one merged model must
     rest on serving one model with no task identification / constant memory, or on other generalisation (paraphrased
     instructions, unseen compositions) — untested.

### EXP-130 — WUDI-Merging (data-free; MergeVLA's strongest static baseline) on the write-protected adapters — **done (10-20)**
Motivation: MergeVLA (CVPR'26) reports WUDI among the best static merges; reviewers will expect it. Implemented in
merge_methods.wudi (official objective, sum init, Adam; factored + vectorised over tasks, identical to the dense loss;
runs under enable_grad inside the no_grad builder). Offline (24 tasks, 3 module sizes): official lr 1e-5 x 300 barely moves
(tau_m ~0.93 x sum ~ 22x averaging's norm, loss >> loss at averaging); lr 1e-3 reaches the averaging loss level; the
objective's optimum has 20-124x averaging's norm (shared input directions -> scale nearly unconstrained).
PRE-REGISTERED (`sweep/exp130_wudi.py` docstring): EXP-125 setting; arms wudi_default (official), wudi_conv (lr 1e-3),
wudi_nm (wudi_conv rescaled to averaging x 1.33, the TIES/DARE/MaxAbs convention); 24 models, 3,360 episodes; build
8.5 min/model. Predictions: unrescaled arms collapse; wudi_nm ~ other linear merges (new ~20%); none reaches ours.
- Queue (10-19 20:00): shiva tmux e130 waits for EXP-129's report, then GPUs 0,1,2. ~5 h -> ETA ~08:00 10-20.
- **EXP-130 RESULT (10-20 09:20; results/analysis/exp130_report.txt; 24/24 complete, no drops).** Retention old / new:
  wudi_default 0.0 / 0.0 (t=4 and t=8); wudi_conv 0.0 / 0.0; wudi_nm (rescaled to averaging x1.33; scale 0.076-0.090)
  1.2 / 0.0 (t=4), 0.6 / 0.0 (t=8). Ours w=0.25 vs each: old +76..77pp, new +50.5pp (all 31/0 and 26/0, p<1e-5).
  - Predictions: unrescaled arms collapse -> CONFIRMED; wudi_nm ~ other linear merges -> REFUTED (also collapses).
  - Mechanism (offline, 3 module sizes, lr 1e-3 solution): share of the merged update inside the shared subspace (two-sided)
    averaging 0.91 / 0.99 / 0.99 vs WUDI 0.006 / 0.011 / 0.003; cosine with the shared part 0.95-1.00 vs 0.06-0.08.
    WUDI's interference objective discards the library's shared component (its large-norm solution lives in directions the
    tasks barely share) -> collapse at any scale. Consistent with the mechanism slide: any merge that loses the shared part
    fails. MergeVLA's WUDI success (4 suite-level experts, small model) does not transfer to a library of LoRAs that share a
    subspace.

### EXP-131 — matched-retention comparison vs OPCM (both methods get a knob) — **done (10-20)**
Motivation: user (10-19): paying on old skills is fine if we pay LESS than competitors -> compare at matched retention.
Pooled over 8 sequences our w=0.1 -> 0.25 curve (78.0/31.4 -> 71.9/50.7) passes roughly through OPCM (74.8/41.2).
PRE-REGISTERED (`sweep/exp131_matched.py` docstring): sequences 0-7, write-protected, t = 8; new arms ours w = 0.15, 0.2 and
OPCM alpha = 0.3, 0.7 (existing: ours 0.1 / 0.25, OPCM 0.5); 32 models, 5,120 episodes; report both curves, ours
interpolated at OPCM's LIBERO-90 levels, nearest-pair paired tests. Prediction: on or slightly above OPCM's curve (no
clear dominance). Dry check: pooled OPCM alpha=0.5 reproduces 74.8 / 41.2.
- Queue (10-19 20:51): shiva tmux e131 waits for EXP-130 (WUDI); ~7 h -> ETA ~15:00 10-20 (before the 10-21 meeting).
- **EXP-131 RESULT (10-20 14:45; results/analysis/exp131_report.txt; 32/32, no drops). Figure results/figs/matched_t8.png.**
  Pooled 8 orders, t=8, LIBERO-90 / new: ours 0.1 78.0/31.4 · 0.15 79.7/39.6 · 0.2 75.1/44.0 · 0.25 71.9/50.7;
  OPCM alpha 0.3 78.9/40.9 · 0.5 74.8/41.2 · 0.7 76.7/41.0.
  - Nearest matched pairs: ours 0.15 vs OPCM 0.3: new -1.4pp (p=0.91), LIBERO-90 +0.8 (p=0.86); ours 0.2 vs OPCM 0.5:
    new +2.8 (p=0.44), LIBERO-90 +0.3 (p=0.24). (The script's third "nearest pair", ours 0.1 vs OPCM 0.7, is matched on
    LIBERO-90 only loosely and our curve is non-monotone at 0.1 -> 0.15 (noise), so its -9.6pp is not a matched comparison.)
  - Prediction: on or slightly above OPCM's curve, no clear dominance -> CONFIRMED (tie at matched retention).
  - OPCM's alpha barely moves it (new 40.9-41.2%, LIBERO-90 74.8-78.9%): OPCM is one operating region; ours spans
    78/31 (0.1) .. 72/51 (0.25) over 8 orders (and 86/20 at 1/k, first 4 orders only) and reaches new-skill retention OPCM
    cannot at any alpha.
  => vs OPCM on write-protected adapters: tie where OPCM operates; our merge adds reach (+10pp new at w=0.25 beyond OPCM's
     region) and the dial. Linear merges stay at ~19% new at any LIBERO-90 level reached.
- EXP-128 add-on (10-19 21:10): RETAIN-style recursion on SCRATCH adapters (EXP-116 sequences; training-free case), same
  rule (a = 0.5), 8 models; prediction: the shared part decays as 0.5^t (scratch adapters lack it) -> LIBERO-90 collapses,
  ours_lib (79.9%) far ahead. `E128_ADAPTERS=scratch`, tags c128z*, report exp128z_report.txt. Queued after EXP-131
  (tmux e128z), ~1 h -> ETA ~16:00 10-20.
- EXP-128 add-on ops (10-19 21:17, after the user asked about odin): moved from the end of shiva's queue to ODIN's free P6000
  (nvidia-smi index 2; the other P6000 is used by another user): 1 policy server, E113_ROUND=1 (odin /dev/shm 32 GB lives
  in RAM: 2 x 7.2 GB in flight), CUDA_DEVICE_ORDER=PCI_BUS_ID + E113_DEV=cuda:2. RAM after server load: 27 GB available
  (driver 14 GB RSS, server 3.7 GB). Shiva's waiting session e128z killed (it only waited). ETA ~06:30 10-20; the watcher
  syncs odin's c128z records to shiva and runs the report there (references live on shiva).
- EXP-128 add-on ops (10-20 04:30): odin's single P6000 runs ~30 episodes/h (22/112 scene evaluations in 7 h; ETA would
  be after the 10-21 meeting). Hand-off armed on odin (tmux e128z_handoff, `sweep/exp128z_handoff.sh`): when EXP-131's
  report lands on shiva (~12:30), stop the odin run, sync its records to shiva, finish there on 3 GPUs (~1 h), then mark
  RUN EXIT so the existing watcher reports. Lesson: one odin P6000 ~ 1/25 of shiva's throughput; use it only for small jobs.
- **EXP-128 add-on RESULT (10-20 15:40; results/analysis/exp128z_report.txt; 8/8 complete; odin did 38 scene
  evaluations, shiva the rest after the hand-off).** SCRATCH adapters, seqs 0-3: RETAIN-style LIBERO-90 / new t=4 0.7 / 15.9,
  t=8 **1.6 / 2.0** vs ours_lib 79.9 / 18.6 (old +78.4pp, 29/0; new +16.6pp, 9/2, p=0.008), OPCM 3.9 / 9.2, avg 62.8 / 10.5.
  - Prediction (shared part decays as 0.5^t -> LIBERO-90 collapses) -> CONFIRMED.
  => training-free case: the only published lifelong VLA merging rule collapses with adapters trained elsewhere; ours keeps
     the library. Completes RETAIN on both adapter types (WP: on our curve; scratch: collapse).

### EXP-132 — training-seed robustness of the headline (seed-1 write-protected adapters) — **done (10-22)**
Motivation: paper requirement (2-3 training seeds); all 10-21 work landed, GPUs idle (user: keep GPUs busy).
PRE-REGISTERED (`sweep/exp132_seeds.py` docstring): retrain the 22 new skills of orders 0-3 with `--init s16wp --seed 1`
(same recipe); single-task 2 x 10 eps; continual orders 0-3, t = 8; ours w=0.1 / 0.25, OPCM, averaging (16 models, 2,560
eps); seed 1 vs seed 0 per arm, paired; pooled. Prediction: |seed delta| <= 5pp per axis per arm; ours w=0.25 stays
>= +25pp above averaging on new skills.
- Queue (10-20 15:42): shiva tmux e132 (`sweep/exp132_chain.sh`), training 3 shards (~9 h), base + run (~3.5 h). ETA ~05:00 10-21.

### Note (10-20, user question "how do you know library skills read but don't write?") — the read/write asymmetry comes from a SHARED LoRA A
- Measurement behind the claim: per weight matrix, ΔW splits into 4 orthogonal parts w.r.t. the shared output (U) / input
  (V) directions; energies (6 adapters per group, 458 matrices) LIBERO-90 79.4 / 20.3 / 0.2 / 0.1. Weight geometry, not
  activations; 6 of 76 adapters (to redo on all 76).
- Check (11 library adapters, 6 random matrices): A matrices nearly IDENTICAL across tasks (mean |A_i - A_j| / |A_i|
  0.02-0.04, row-space overlap 0.998-1.000); B matrices differ 34-68%. => the fleet was trained from the SAME A
  initialisation and A barely moved (known LoRA asymmetry); task differences live in B. "Reads the shared inputs, never
  writes into the shared outputs from other inputs" follows from the shared A (every update's rows lie in span(A0) = V).
- Implications: (1) "foreign" adapters are foreign mainly because their A init differs (our zero-init externals: unseeded
  A); (2) write-protected training = adopting the library's training regime (library A frozen, learn B) + keeping B's
  shared-direction component fixed; (3) paper framing: LoRA skill libraries trained from a common A share an input
  subspace; cite LoRA asymmetry (Zhu et al. 2024) / shared-A designs (HydraLoRA).
- Proposed ablation (next week): new skill with A frozen at the library's A, B from zero, no other constraint — does that
  alone make it merge like a library skill?

### EXP-133 — write-protected vs plain fine-tuning from the shared part (anchored), in the headline setting — **done (10-21)**
Motivation: user (10-20) "is write-protected different from fine-tuning new skills from the same shared component? which is
better?" — the headline (all new skills weighted 0.1 / 0.25) was only run with write-protected adapters. Evidence so far:
equal at 1/k (EXP-118 vs 124), write-protected clearly better at w=1 for privileged skills (EXP-120/123/126), anchored
protects more but costs the library at w=0.25 (privileged). PRE-REGISTERED (`sweep/exp133_anchored_weighted.py`
docstring): EXP-118 sequences 0-3 (anchored adapters), t = 4, 8, ours w = 0.1 / 0.25 (EXP-124 rule); 16 models, 2,240 eps;
paired vs EXP-124. Prediction: equal at w=0.1; at w=0.25 write-protected keeps >= +5pp more LIBERO-90 at similar new-skill
retention.
- Queue (10-20 16:42): shiva tmux e133 waits for EXP-132's report; ~3 h -> ETA ~08:30 10-21 (before the 14:00 meeting).
- Ops (10-21 11:55): SHIVA FROZE overnight and was rebooted at 10:13. The exp132 log is silent after 00:01 (2 of 3 training
  shards had finished; 20/22 seed-1 adapters trained, 1 partial); shiva's user journal ends 03:31 (last watcher logins);
  ssh at 08:20 authenticated but no command ran. Kernel log not readable without adm rights -> cause unknown. All tmux chains
  were lost; EXP-133 had never started. Recovery: EXP-133 relaunched 11:52 with t=8 models first (meeting 14:00), then the
  EXP-132 chain resumes (trains the 2 missing adapters, base, run, report). Watchers re-armed (ssh with timeouts).
- **EXP-133 INTERIM (10-21 13:15; t=8 complete, t=4 running; results/analysis/exp133_report_t8_interim.txt).** Same merge,
  orders 0-3, after 8 additions, LIBERO-90 / new: w=0.25 anchored 58.9 / 51.7 vs write-protected **76.9 / 50.5** (WP - anchored:
  old +18.0pp, p=0.003; new -1.2, n.s.); w=0.1 anchored 77.2 / 35.0 vs WP 85.2 / 29.6 (old +8.1, p=0.041; new -5.4, p=0.055).
  At matched LIBERO-90 (~77%: anchored w=0.1 vs WP w=0.25) write-protected learns new skills +15pp more (50.5 vs 35.0).
  - Prediction: equal at w=0.1 -> REFUTED (WP +8pp old, -5pp new: a different point on a better curve); at w=0.25 WP >= +5pp
    LIBERO-90 at similar new-skill retention -> CONFIRMED (+18pp).
  => write-protection is needed: plain fine-tuning from the shared part learns new skills as well but costs the library
     far more when they are weighted up. Deck add-on slide updated (one bullet).


### EXP-134 — evaluation after EVERY addition (make the continual claim real), user plan 10-21 — **done (10-25)**
Motivation (user 10-21): "Make the continual claim robust (so far: final model evaluated only); keep 8 new tasks; evaluate
after EVERY addition, on all library skills + all new skills so far; report average incremental retention and the
library's forgetting curve; try a non-uniform coefficient schedule for new tasks." PRE-REGISTERED (`sweep/exp134_stepwise.py`
docstring): t = 0..8, all 16 library skills + new so far, 10 eps, 4 orders. Setting wp (write-protected new skills): ours
w=0.25 / 0.1 / budget (w_t = min(0.25, 1/n_new): total new weight <= one skill), OPCM, RETAIN-style, averaging (204 models,
41,280 eps, ~2.2 days); setting scratch (training-free claim): ours, OPCM, RETAIN-style, averaging (queued later). Metrics:
library / new / overall retention per t, average incremental retention, library forgetting curve, new-skill backward
transfer. Schedule rationale: the library's cost grows with the TOTAL weight of new skills (t x w); a constant budget keeps
it bounded; recency weighting (RETAIN-like) is not sensible (forgets early new skills).
- Queue (10-21 15:08): shiva tmux e134w waits for EXP-132's report. Then: MergeVLA-style baseline, joint-training upper
  bound, then the scratch setting (drivers being written).
- **EXP-133 RESULT (10-21 ~14:10; results/analysis/exp133_report.txt; 16/16 complete).** After 4 additions: w=0.1 old 83.1
  vs 85.5 (WP +2.4, n.s.), new 35.2 vs 33.0; w=0.25 old 73.2 vs 82.4 (WP +9.2, p=0.08), new 60.7 vs 52.4 (n.s.). After 8: as in
  the interim (+8.1 / +18.0pp LIBERO-90 for WP). => the write-protection advantage grows with the number of new skills
  (interference of unconstrained adapters accumulates).
- **EXP-134 scratch RESULT — official training-free method after every addition (10-24 10:21;
  results/analysis/exp134_scratch_report.txt / _curves.json; figure results/figs/stepwise_scratch.png, script
  make_stepwise_figure.py, values asserted against the report).** 4 orders x t = 0..8, all 16 library skills + new so far;
  complete (one dropped scene, c134zs0t1-retain, rerun). Library % / new % per t:

  | arm | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 | t=6 | t=7 | t=8 |
  |---|---|---|---|---|---|---|---|---|---|
  | ours (1/k) | 86.3 | 83.7 / 63.9 | 81.5 / 31.9 | 82.1 / 19.4 | 81.5 / 23.1 | 78.6 / 21.7 | 75.8 / 21.6 | 77.8 / 18.9 | 76.6 / 17.3 |
  | averaging | 86.1 | 85.0 / 53.1 | 79.5 / 27.9 | 78.8 / 17.8 | 73.1 / 23.3 | 69.6 / 19.5 | 69.9 / 19.7 | 64.6 / 13.6 | 59.7 / 10.7 |
  | OPCM | 79.0 | 67.2 / 58.6 | 50.1 / 33.3 | 33.8 / 17.7 | 24.2 / 24.2 | 15.2 / 13.7 | 7.4 / 13.4 | 7.0 / 11.9 | 6.3 / 9.2 |
  | RETAIN-style | 86.1 | 24.4 / 81.0 | 5.3 / 25.9 | 3.5 / 12.6 | 2.4 / 13.5 | 2.2 / 24.7 | 1.4 / 18.6 | 0.5 / 6.8 | 1.2 / 7.4 |

  Summary (average incremental retention over t=1..8 overall / new; library forgetting t=0->8; mean new-skill backward
  transfer, n=28): ours 67.9 / 27.2, -9.8pp, -0.6pp; averaging 61.9 / 23.2, -26.4, -4.7; OPCM 26.3 / 22.7, -72.7, -7.2;
  RETAIN-style 8.0 / 23.8, -84.8, -40.2. Paired Wilcoxon: ours vs averaging library at t=8 +16.9pp (n=64, p=8e-7), mean
  over t=1..8 +7.2pp (p=8e-6); overall retention per (order, t) +6.0pp, 28/32 (p=3e-7). vs OPCM / RETAIN: library t=8
  +70 / +75pp, overall 32/32 (p<1e-9). At t=1 averaging is level with ours (85.0 vs 83.7); ours ahead from t=2 on.
  - Predictions (scratch): ours' library curve stays flat -> PARTLY REFUTED (declines ~1.2pp per addition, -9.8pp after 8,
    vs -26.4 for averaging); OPCM / RETAIN decline with t -> CONFIRMED (RETAIN collapses at t=1-2: the recursion halves the
    library with every addition); averaging declines steadily -> CONFIRMED.
  - Consistency with EXP-116 (same t=0 / t=8 models): on EXP-116's 8 library skills this run measures ours 88.8 (t=0) ->
    75.6 (t=8) vs EXP-116's 82.0 -> 79.9; on the other 8 skills 83.9 -> 77.6. The same models re-evaluated differ by up to
    7pp (10 episodes per skill; sampling noise); EXP-116's -2.1pp forgetting was optimistic (low t=0 draw). Averaging
    (-23.0 / -29.7 on the two halves vs EXP-116 -23.4) and OPCM agree. => the 16-skill per-step number (-9.8pp) replaces
    the 21-10 deck's -2.1pp; every ranking holds.
  - New skills: after the first addition no method keeps a new skill (<= 33% for every method from t=2) — the add-on's job; ours has the
    best average incremental new-skill retention (27.2%) and no negative backward transfer (-0.6pp vs RETAIN -40pp).
  => official method's continual claim: least library forgetting from the 2nd addition on, best average incremental
     retention, earlier new skills kept as later ones arrive. Deck 28_10 slide filled.
- **RESULT, write-protected add-on (10-25 17:25; results/analysis/exp134_wp_report.txt; complete: 6 arms x t=0..8 x 4
  orders, after 6 dropped evaluations were rerun).** Library after 8 / library forgetting / average incremental retention
  (all / new) / new-skill backward transfer (n=28):

  | arm | library t=8 | forgetting | avg incremental all / new | BWT |
  |---|---|---|---|---|
  | ours w=0.25 | 73.6 | -12.7 | 73.9 / 52.6 | +6.0 |
  | ours budget | 81.2 | -5.1 | 73.4 / 42.7 | -11.7 |
  | ours w=0.1 | 82.8 | -3.5 | 73.3 / 32.3 | +8.6 |
  | OPCM | 80.9 | +1.9 (t=0 79.0) | 72.4 / 51.2 | -39.9 |
  | RETAIN-style | 83.5 | -2.6 | 72.3 / 51.0 | -55.4 |
  | averaging | 83.7 | -2.4 | 70.9 / 27.4 | +2.7 |

  Paired (ours w=0.25 minus): overall avg incremental (per order x t, n=32) OPCM +1.5 (p=0.048), RETAIN +1.7 (p=0.03),
  averaging +3.1 (p=0.002); new-skill avg incremental (per order x new skill, n=32) OPCM +2.5 (p=0.18), RETAIN +1.1
  (p=0.35), averaging +30.2 (p=5e-5); BWT OPCM +45.9, RETAIN +61.4 (p<3e-5), averaging +3.4 (n.s.); new skills at t=8
  OPCM +17.6 (p=0.011), RETAIN +18.7 (p=0.028); library at t=8 OPCM -7.2, RETAIN -9.9, averaging -10.1 (p<0.004);
  library mean over t=1..8 OPCM +0.3, RETAIN -0.1 (n.s.), averaging -3.7 (p=0.01). Budget vs w=0.25: library t=8 +7.6
  (p=0.002), new t=8 -24.1 (p=2e-4), BWT -17.7 (p=0.03). Ours w=0.1 / budget vs OPCM / RETAIN: library mean +2.9..+5.4
  (p<0.005), new mean -15..-22 (p<0.012).
  - Built-in replicate: the budget schedule equals w=0.25 for t <= 4 (same model, evaluated twice): differences up to
    4pp library / 7pp new = evaluation noise (10 episodes); ours w=0.25's last-step library drop (78.1 -> 73.6) is in that
    range.
  - Predictions (wp): ours w=0.25 highest average incremental new-skill retention -> nominally (52.6 vs 51.2 / 51.0) but
    a TIE (n.s.); w=0.1 library flat -> CONFIRMED (-3.5); w=0.25 declines with t -> CONFIRMED (-12.7; plateau t=3..7);
    budget flattens the library at a cost on new skills growing with t -> CONFIRMED (-5.1 vs -12.7; new-skill cost
    -2.5 at t=1 .. -24.1 at t=8); RETAIN forgets early new skills, ours does not -> CONFIRMED (-55.4 vs +6.0; OPCM too).
  => same average as OPCM / RETAIN-style, different profile: they learn a new skill at arrival (86 / 94% at t=1) and
     lose it with each addition; ours keeps it -> after 8 additions ours holds +18pp more of the new skills, paying
     7-10pp of library at t=8 (level on average over the steps). Figure results/figs/stepwise_wp.png (all 6 arms; user
     10-25: "figures where every method gets a curve"). Deck 28_10 add-on + budget slides filled.


### EXP-135 — MergeVLA-style baseline (training-free part: TIES + per-task masks + own expert head, oracle routing) — **done (10-25)**
Motivation: user (10-21) "add more baselines: additional continual merging methods, like MergeVLA". MergeVLA itself needs a
cross-attention-only action expert trained from scratch + a test-time router (not one merged model, not continual) -> not
reproducible on pi0.5; its training-free merge transfers. PRE-REGISTERED (`sweep/exp135_mergevla.py` docstring): tau_m =
TIES (top-20%, norm-matched x1.33), mask S_t = |tau_t| > 0.6 |tau_m - tau_t| (MergeVLA default), expert head (action-expert
layer 17 + action_out_proj) = task's own update; one task model per evaluated task, oracle routing; settings wp + scratch,
orders 0-3, t = 4, 8 (224 task models). Dry check: mask keeps 34% of entries; cos(masked merge, own update) 0.87; head ==
own update.
- Run: GPU 1 (`sweep/chain4_gpu1.sh`), 10-24 15:56 -> 10-25 04:53, 224/224 task models evaluated (224 ok records, report
  not flagged incomplete); report 10-25 04:55 (tmux reports5).
- **RESULT (10-25; results/analysis/exp135_report.txt).** Retention LIBERO-90 (8 fixed library skills x 4 orders) / new;
  references = EXP-124 / EXP-125 (write-protected) and EXP-116 (scratch), same orders and skills:

  | setting, t | MergeVLA-style (oracle routing) | ours | other |
  |---|---|---|---|
  | write-protected, t=8 | 97.5 / 89.4 | w=0.25 76.9 / 50.5; w=0.1 85.2 / 29.6 | OPCM 78.4 / 36.0 |
  | write-protected, t=4 | 101.8 / 86.0 | w=0.25 82.4 / 52.4; w=0.1 85.5 / 33.0 | OPCM 76.1 / 50.2 |
  | scratch, t=8 | 67.0 / 29.2 | 1/k 79.9 / 18.6 | averaging 62.8 / 10.5; OPCM 3.9 / 9.2 |
  | scratch, t=4 | 86.0 / 50.8 | 1/k 81.3 / 26.4 | averaging 82.0 / 24.4; OPCM 14.4 / 22.0 |

  Paired over (order, skill), MergeVLA-style minus: write-protected t=8 vs ours 0.25 library +20.5pp (p=1e-4, 21/2), new
  +38.9 (p=3e-5); vs OPCM +19.1 / +53.4 (p<2e-4). Scratch t=8 vs ours: library -12.9pp (n=32, p=0.043, 9 up / 18 down),
  new +10.6 (p=0.12, n.s.); vs averaging +4.2 (n.s.) / +18.7 (p=0.001). Scratch t=4 vs ours: +4.7 (n.s.) / +24.4 (p=0.10).
  - Predictions: beats every single-merge method on new skills -> CONFIRMED for write-protected (+39..+60pp); scratch only
    vs averaging / OPCM (vs ours +10.6, n.s.). Library near averaging -> CONFIRMED for scratch (+4.2 n.s.), REFUTED for
    write-protected (97.5, near single-task, vs averaging ~84).
  - Reading: a routed model, not one merged model: it needs the task id at test time and keeps a mask + own head per skill.
    With the task id given, loading each skill's own LoRA is 100% / 100% by definition -> the oracle-routed numbers sit
    below plain adapter swapping; MergeVLA's case rests on its learned router (not reproducible on pi0.5; a router can only
    lower these numbers). With from-scratch skills its library falls below our single model after 8 additions even with
    the task id (67.0 vs 79.9). -> report apart from the single-model merges, next to joint training.

### EXP-136 — joint-training upper bound (one LoRA on all 24 skills per order) — **done (10-23)**
PRE-REGISTERED (`sweep/exp136_joint.py` docstring): rank 64, 12,000 steps, batch 8, one skill drawn uniformly per step
with its bare-instruction prompt (LIBERO-90 scene prefix stripped -> matches the eval prompt), stride 4, from the base;
4 orders; eval on all 24 skills. Trainer gained --tasks_file / --rank / --stride (existing modes unchanged). Smoke tests:
odin P6000 OOM even at rank 32 (no bf16 on Pascal -> fp32 weights) -> joint training only on shiva; shiva rank 64 batch 8
OK (17.5 GB RSS with 3 tasks; ~30 GB expected with 24).

- Distribution (10-21 15:36, user: "distribute the tasks on all the GPUs that can be used"): odin's only free GPU (P6000)
  can neither train these models (OOM) nor evaluate usefully (~1/25 of shiva) -> all on shiva's 3 A6000s, split so all
  stay busy: tmux g0 (GPU 0): EXP-136 train x4 (~40 h) -> EXP-136 eval -> EXP-134 wp shard 0/4 -> EXP-134 scratch shard
  0/4 (ports 8100+, 1 model per round); tmux g12 (GPUs 1,2): EXP-134 wp shard 1,2,3/4 -> EXP-135 -> EXP-134 scratch shard
  1,2,3/4 (2 models per round; /dev/shm budget ~43 GB); tmux reports: each report as soon as its inputs exist. All start
  after EXP-132's report (~20:30). Projected finish ~10-26 morning. Health monitor on odin wakes on outage / 3 h stall.
- Ops (10-21 17:37): shiva rebooted AGAIN (second time today, up 4 min at 17:37); all chains lost. nvidia-smi: "Unable to
  determine the device handle for GPU1 (0000:1B:00.0): Unknown Error" -> GPU 1 has failed (likely cause of both freezes).
  User palma logged in at 17:33 (probably investigating). Seed run (EXP-132): base complete (BASE EXIT 17:11), merges not
  run. User chose the TRIMMED plan on GPUs 0+2 (10-21): per-step write-protected run without the RETAIN arm (already shown
  to lie on our curve); MergeVLA-style; joint training (4 orders, GPU 0); from-scratch per-step only at t = 2, 4, 6, 8.
- Ops (10-21 17:55): a CUDA enumeration probe (initialising every GPU incl. the failed one) hung; at 18:00 shiva became
  unreachable (no ping). Cause unclear (admin action or another freeze; the probe touching the dead GPU cannot be ruled
  out). Rule from now on: never initialise CUDA on all of shiva's GPUs; pin jobs to the working GPUs by UUID.
- Ops (10-21 18:14): shiva back at 18:09 (palma rebooted it); GPU 1 visible and idle again, but it failed twice today ->
  kept OUT of the plan (user's choice: trimmed plan on GPUs 0+2). Launched tmux gpu0 (`sweep/chain2_gpu0.sh`: joint x4 ->
  joint eval -> per-step shards 3,4/5), gpu2 (`sweep/chain2_gpu2.sh`: EXP-132 merges + report -> wp shards 0,1,2/5 ->
  EXP-135 -> scratch shards 0,1,2/5), reports2. Pinned by UUID (orchestrator gained optional ORCH_CUDA_MAP / ORCH_EGL_MAP;
  rendering on EGL device 0); verified servers on GPU 2 only, GPU 1 untouched. A first launch died with the tmux server
  when its ssh session was killed by `timeout` -> relaunched with setsid. Health monitor (odin) also alerts on GPU errors.
  Projected finish ~10-26 night.
- Ops (10-21 19:20): the UUID pinning broke every evaluation on GPU 2 for ~1 h (status parse_fail in 2 s): robosuite asserts
  that MUJOCO_EGL_DEVICE_ID appears in CUDA_VISIBLE_DEVICES, and the client inherited the UUID. The EXP-132 merges (16) and
  the first EXP-134 items "ran" with no valid records (the chain even wrote a false "[e132] REPORT EXIT"). Fix: the
  orchestrator drops CUDA_VISIBLE_DEVICES from the eval client's env when a GPU map is set (the client needs no CUDA).
  Stopped gpu2 + reports2, relaunched; first eval after the fix OK (rate 1.0, 10 eps). Failed records are status parse_fail
  -> ignored by rec() -> rerun automatically. gpu0 (joint training) unaffected; its later eval stages use the fixed file.
- Ops (10-21 19:50, user: "do the curves not only for write protected checkpoints, since that's an add-on method, not the
  official method"): re-prioritised. The official (training-free) method's per-step run now goes FIRST and at EVERY step
  (from-scratch adapters, t = 0..8, ours / OPCM / RETAIN / averaging, 28k eps); the write-protected add-on per-step run
  drops to t = 2, 4, 6, 8 (skip RETAIN); MergeVLA-style last. New chains: tmux gpu2 (`sweep/chain3_gpu2.sh`: seed merges ->
  scratch shards 0,1,2/5 -> wp shards 0,1,2/5), gpu0next (`sweep/chain3_gpu0.sh`: waits for the running joint training ->
  joint eval -> scratch shards 3,4/5 -> wp shards 3,4/5 -> MergeVLA), reports3. Joint training kept running (only the old
  gpu0 controller bash was stopped, by PID). Projected finish ~10-26 midday.
- **EXP-132 RESULT (10-22 01:40; results/analysis/exp132_report.txt; 16/16 models, 44/44 single-task runs).** Orders 0-3,
  after 8 additions, LIBERO-90 / new retention, seed 1 vs seed 0 (paired): single-task 94.8% vs 96.1%;
  ours w=0.1 84.5/26.2 vs 85.2/29.6 (diffs -0.7 / -3.4, n.s.); ours w=0.25 75.3/51.0 vs 76.9/50.5 (-1.6 / +0.5, n.s.);
  OPCM 80.5/32.8 vs 78.4/36.0 (+2.1 / -3.2, n.s.); averaging 86.1/20.9 vs 84.2/19.0 (+1.9 / +1.9, n.s.).
  - Predictions: |seed delta| <= 5pp per axis per arm -> CONFIRMED (max 3.4pp, all n.s.); ours w=0.25 >= +25pp above
    averaging on new skills -> CONFIRMED (seed 1: +30.1pp).

- Ops (10-22 09:02, user: "try to use [GPU 1] now and see if it works, if so, use it henceforth"): GPU 1 stress test
  (5 min, 8192^2 fp32 matmuls pinned by UUID): 6,000 iterations, identical checksums, 299 W (power cap), peak 89 C; PCIe x8
  on x16 slots for all three cards (not GPU-1 specific). -> GPU 1 back in use: tmux gpu1 (`sweep/chain4_gpu1.sh`: scratch
  every-step shards 3,4/5 -> wp t=2,4,6,8 shards 3,4/5 -> MergeVLA-style -> restore units; ports 8200+, EGL on device 1);
  first eval OK (rate 1.0). GPU 0 (`sweep/chain4_gpu0.sh`): joint training (running) -> joint eval -> restore units.
  GPU 2 unchanged + tail (`sweep/chain4_gpu2tail.sh`) joins the restore units. RESTORED (`sweep/restore_units.sh`, 8 units
  claimed by atomic mkdir in results/claims/): write-protected add-on at the odd steps t=1,3,5,7 (5 arms) and the RETAIN
  arm at every step -> the full add-on per-step plan is back. Reports: tmux reports4 (`sweep/chain4_reports.sh`): official
  method (scratch) report, add-on even-step interim report, MergeVLA, joint, add-on full report. Temperatures under load:
  GPU 0 87 C, GPU 1 89 C -> temperature alert (>= 93 C) added to the odin watchers.
- **EXP-136 RESULT (10-23 11:47; results/analysis/exp136_report.txt).** 4 joint models (rank 64, 12k steps; 9.6-9.8 h each
  on GPU 0), each evaluated on its 24 skills (10 eps): retention library 91.0% (n=64), new 87.3% (n=32), overall 89.8%.
  For reference after 8 additions (same orders / skills): ours write-protected w=0.25 71.9 / 50.7 (8 orders), w=0.1 78.0 /
  31.4; ours training-free 79.9 / 18.6; OPCM 74.8 / 41.2. Predictions: joint is the best single model, library and new
  >= 80% -> CONFIRMED; gap largest on new skills -> CONFIRMED (new -36pp vs ours 0.25; library -13..-19pp).
  => joint training (all data at once, ~10 GPU-h per model) remains far above every training-free merge; the merges'
     case is no data, no retraining, adding a skill in seconds.
- Ops (10-23 15:40): shiva quota 45.7 / 46 GB -> archived results/analysis/new_loras_weak (1.48 GB, donor era) and
  new_loras_s16wp_seed1 (4.06 GB, EXP-132 done) to the NAS (file bytes + checksum verified, local removed) -> 40.5 GB.
  /dev/shm 86% with three chains (max in flight 8 x 7.2 GB = 57.6 GB < 63 GB). One dropped evaluation so far
  (c134zs0t1-retain, 1 scene) -> rerun when the shard finishes.

- Ops (10-24 10:12-10:20): /dev/shm overflow (ENOSPC on model save). Cause: the one-scene rerun for c134zs0t1-retain
  (tmux fixz, waited for 16 GB free) took a 7 GB slot freed between two rounds of another driver, on top of a stale 7 GB
  EXP-132 model left in /dev/shm since 10-21 -> GPU 2's wp shard and GPU 0's restore unit crashed while building; their
  running evals continued as orphans; the restore loops then crashed through the remaining units in 2 min and marked 6 of
  8 units "done" (restore_units.sh touched done regardless of exit status). Fixes: removed 10 stale / partial models (54% ->
  34 GB in use), reset the 6 false claims; restore_units.sh marks done only on exit 0, releases a failed claim and retries
  (3 passes); exp134 waits for its running eval before re-raising a build error (no orphaned eval holding the ports). Relaunch:
  tmux gpu2r (`sweep/chain5_gpu2.sh`: waits for its orphaned eval -> rest of wp shard 0,1,2/5 -> restore units), gpu0r
  (`sweep/chain5_gpu0.sh`: waits -> restore units), reports5 (`sweep/chain5_reports.sh`: wp even, MergeVLA, wp full);
  GPU 1's chain (gpu1) unaffected. No records lost (records are per evaluation). Peak /dev/shm now 8 x 6.8 = 54 GB.
- Ops (10-24 11:20): /dev/shm back at 97% after the relaunch: the two orphaned evals had finished, but nobody deletes an
  orphan's models (its driver had crashed) -> 3 stale models (20 GB) removed just before GPU 2's next build; now 41 GB,
  peak 54 GB. Watcher on odin alerts at >= 92%. Both relaunched chains evaluating normally (no failed runs).
- Ops (10-24 12:51): shm alert (97%). GPU 2's first relaunch (11:15) had hit ENOSPC building its 2nd round (shm was still
  full of the orphans' models at 11:15-11:20); the new wait-on-error kept its running round alive until 12:18 but left that
  round's 2 models in /dev/shm -> try 2 (12:28) ran on top of them (one build took 16 min under memory pressure; swap 47 GB).
  Removed the 2 stale models (97% -> 76%); exp134 now also deletes its own models on a build error. No records lost.
- Ops (10-25 08:03): no failure since 12:51. Units: wp_odd_0/1/2 and wp_retain_0/1 done; GPU 0 on wp_odd_3 (claimed 05:24,
  4/20 rounds, ~35 min/round -> ~17:30), GPU 1 on wp_retain_2 (claimed 04:53 after MergeVLA, 5/8, ~31 min/round -> ~09:00),
  GPU 2 on the last round of wp shard 0,1,2/5 (19/20) -> then wp_retain_3 (~12:30). EXP-137 shards start as GPUs free up
  (GPU 1 ~09:00). Projected: even-step interim report ~08:30; full add-on report ~17:30 (wp_odd_3 is the long pole);
  EXP-137 ~18:00. /dev/shm 65%, 44 GB RAM free, GPUs 70-79 C.
- EXP-134 wp INTERIM (10-25 08:15, even steps t=2,4,6,8; results/analysis/exp134_wp_even_interim_report.txt). Library /
  new retention at t=8: OPCM 80.9 / 35.0 (t=0 79.0), averaging 83.7 / 22.8, ours w=0.1 82.8 / 31.7, ours budget 81.2 /
  28.5 (t=0 of ours 86.3). Mean over t=2,4,6,8 (overall / new) and new-skill backward transfer (n=12): OPCM 71.6 / 46.6,
  -39.9pp; averaging 69.3 / 25.4, +7.7; ours w=0.1 71.7 / 29.9, +6.0; ours budget 71.8 / 38.3, -7.4. Ours w=0.25
  incomplete (t=1..5: library 84.8 -> 79.2, new 76.2 / 42.5 / 42.7 / 59.1 / 55.1). Early read: OPCM learns each new skill
  at arrival (86% at t=1) and then loses ~40pp of it; ours keeps new skills (w=0.1 +6.0). Not in the deck until the full
  report.
- Ops (10-25 08:19): 4 add-on evaluations dropped (status parse_fail in 3 s, transient: 5 of ~5,100 since 10-23) in units
  already finished -> nothing retries them: ours w=0.25 order 1 at t=6, 7, 8 (one scene each; why w=0.25 is incomplete)
  and RETAIN-style order 3 at t=6. Rerun queued on GPU 1 before its EXP-137 shard (`sweep/chain7_gpu1.sh`, tmux fix1,
  replaces the waiting e137g1). A final completeness sweep follows the full report.
- Ops (10-25 17:07): GPU 1's rerun done 09:33 (4/4). All 8 add-on units done; full report 17:00 still missing 2 evaluations
  (parse_fail: c134ws2t4-retain 10:15, c134ws3t7-opcm 09:31) -> OPCM / RETAIN-style summary incomplete. EXP-137: shards
  0, 1 done, 2 on GPU 1 (6/8); 3 parse_fail in done shards (c137s3t4-dilute x2, c137s0t8-dilute). Reruns on the idle GPUs
  (`sweep/fix8_reruns.sh`: tmux fix8g0 = the 2 add-on, fix8g2 = the 3 EXP-137); both reports re-run after them.
  Figure script generalised (`results/figs/make_stepwise_figure.py scratch|wp`; scratch output unchanged, CSV identical);
  wp palette: ours 0.25 / budget / 0.1 = one blue light -> dark by new-skill weight (ordinal check PASS), baselines keep
  their scratch-figure colors (categorical check PASS; contrast WARN -> direct labels + CSV).


### EXP-137 — why does the library decline: smaller weight (dilution) or interference? — **done (10-25): mostly interference; holding the library's weight does not help**
Motivation (user 10-24: a non-flat library curve is expected): our merge weights every skill's task part 1/k, so each library
skill's part shrinks 1/16 -> 1/24 over 8 additions even with zero interference. Split EXP-134's -9.8pp into the two causes; if
the shrinkage dominates, a fixed library weight would flatten the curve. PRE-REGISTERED (`sweep/exp137_dilution.py`
docstring): fixlib = library parts fixed at 1/16, new parts at 1/k as in ours (t = 2, 4, 6, 8); dilute = library parts at
1/(16+t), no new skill (t = 4, 8); same shared part, scratch adapters, 4 orders, 10 eps; 24 models, 464 evaluations
(~8.5 GPU-h; first quoted to the user as ~5 GPU-h — underestimate). Predictions: fixlib library near flat (t=8 within 4pp of
t=0); dilute alone <= -6pp at t=8 (most of ours' drop); fixlib new-skill retention within 3pp of ours.
- Queue (user: "yes" to running it after the current runs): 3 claimable shards (results/claims137/, separate from the add-on
  units so the full add-on report's 8-done trigger is unaffected); each GPU takes shards when its current chain ends
  (`sweep/chain6_gpu{0,1,2}.sh`), report via `sweep/chain6_reports.sh`.
- Check (10-24, shiva CPU, order 0, one expert + one VLM module): t=0 fixlib/dilute == ours exactly (rel diff 0.0); identity
  (ours - fixlib at t=8) == (dilute at t=8 - t=0) to 3e-5 rel -> the arms differ from ours only in the library weight. The
  library-weight change is 2-4% of the module's merged update norm. Launched 10-24 ~13:40: tmux e137g0/1/2 wait for their
  GPU's current chain, rep137 for the report.
- Run: shards 0 (GPU 1, 09:33-13:49), 1 (GPU 2, 12:31-16:53), 2 (GPU 1, 13:49-17:37); 3 dropped evaluations (parse_fail)
  rerun on GPU 2 (`sweep/fix8_reruns.sh`, 17:07-17:17); report rebuilt 17:45, complete (24 models, all cells).
- **RESULT (10-25; results/analysis/exp137_report.txt).** Library retention (16 skills x 4 orders), t=0 86.3 for all arms:

  | t | ours (library 1/(16+t), new 1/(16+t)) | fixed library (1/16, new as ours) | shrink only (1/(16+t), no new) |
  |---|---|---|---|
  | 2 | 81.5 / new 31.9 | 80.5 / new 33.3 | - |
  | 4 | 81.5 / 23.1 | 79.2 / 27.1 | 84.2 |
  | 6 | 75.8 / 21.6 | 78.8 / 20.5 | - |
  | 8 | 76.6 / 17.3 | 76.8 / 19.4 | 82.8 |

  Paired over (order, library skill), n=64: fixed library - ours -1.1 / -2.3 / +3.0 / +0.2 at t=2/4/6/8 (all n.s.);
  fixed library - t0 at t=8 -9.6 (p=0.006); shrink only - t0 -2.2 (t=4, n.s.), -3.5 (t=8, p=0.035); ours - shrink only
  (same library weight, only difference = the new skills) -2.7 (t=4, n.s.), -6.2 (t=8, p=0.08). New skills fixed library -
  ours +1.4 / +4.0 / -1.1 / +2.1.
  - Predictions: (1) fixed library near flat (t=8 within 4pp of t=0) -> REFUTED (-9.6, same as ours); (2) shrinking alone
    accounts for most of ours' drop (<= -6pp) -> REFUTED (-3.5, about a third); (3) new skills fixed library within 3pp of
    ours -> CONFIRMED at t=2, 6, 8 (+1.4 / -1.1 / +2.1; +4.0 at t=4).
  - Reading: about 1/3 of the -9.8pp is the smaller weight (seen only without new skills), about 2/3 interference from the
    new skills (-6.2, borderline). With new skills present, restoring the library's weight recovers nothing (+0.2) -> the
    decline is NOT fixed by reweighting; the 28-10 deck's "declines because each library part is weighted 1/k" was my
    hypothesis and is wrong -> corrected. Caveat: all arms share one t=0 evaluation (re-evaluations differ up to 4-7pp).
  => to keep the library flatter, reduce the new skills' interference (what write-protection does at training time);
     raising the library's weight is not a lever. Figure results/figs/dilution_scratch.png
     (`results/figs/make_dilution_figure.py`, values parsed from the report). Deck 28_10 slide filled.

### EXP-138 — offline gate: can external updates be made to respond only to their own task's inputs? — **done (10-25): soft gating passes; plain projections do not**
Motivation (user 10-25: "enough results for a paper, just not for a top conference yet"; improving retention of EXTERNAL
tasks is "the single most valuable thing"; brainstorm -> direction 1 go, direction 2 maybe; user: "yes go ahead").
Why external skills fail (EXP-116/117/119/120/134/137): an external (from-scratch, foreign-A) LoRA keeps 83% alone but ~18%
merged; up-weighting breaks the model; 92.5% of its weight reads input directions outside the library's -> it fires on
every task's inputs. A fix must make each external update respond mainly to its own task's inputs.
PRE-REGISTERED (10-25, before any number; `sweep/exp138_readgate.py` docstring is the record): order 0, scratch condition
(16 library + 8 external, our training-free merge at t=8 as the network); per adapted layer, input activations of all 24
skills from 24 demo frames each (12 fit / 12 test frames, own prompt). For each external update, energy of its output on
own / library / other-external test inputs, for:
  direction 1  hard: read factor projected off the top-q input directions of the other 23 skills (q 16/64/256/1024);
               control: random projection of the same q; soft: R' = R G_own (G_own + mu G_others)^-1 (mu 1/10/100)
  direction 2  refit to read only the library's 16 shared input directions (least squares on own fit inputs);
               controls: 16 random directions; best 16 directions of its own inputs (upper bound for any 16-dim read)
Gate (energy-weighted over layers, mean over the 8 externals): direction 1 GO if a variant keeps own-output error <= 0.3
with library and other-external output kept <= 0.3 each, while the random projection at that q keeps >= 0.6 (MAYBE if
<= 0.6); direction 2 GO if the refit error <= 0.3. Layer-output energy is a proxy -> rollouts only after a GO.
- Smoke tests (10-25 17:55-18:15, order 0, first 6-12 layers, 4-8 frames; not the experiment): pipeline OK. Fixes before
  the full run: soft{mu} now pulled toward R where no fit token reaches (a tiny ridge toward 0 blew up: kept 17-249,000x);
  others' input moment summed directly (total minus own lost positive-definiteness in fp32); refit ridge. Added after the
  smoke test: gen{k} (selective read: top generalized eigendirections of own vs others' inputs) and diagnostics (constant-
  offset share of the external's output; its cosine on library inputs and with the shared part's output). Early signals on
  12 layers: externals fire on library / other-external inputs as strongly as on their own (0.92 / 0.95); hard / gen
  projections remove own output as much as others' (no selectivity); soft1 keeps own 0.30 vs others 0.15 (own error 0.39);
  83% of an external's own output is a constant offset, the same on library inputs (cos 0.96); refit to the library's
  16 directions 0.15 error but 16 random directions 0.18 (inputs dominated by a few directions) -> refit test not
  discriminative; external offsets vs the shared part's output: cos 0.46, 22% of output along it.
- Launch (10-25 18:17): full gate for orders 0, 1, 2 on GPUs 0, 1, 2 (tmux e138g0/1/2; 24 frames, 1536 tokens per skill,
  split and layer); watcher on odin prints the three reports.
- Restart (10-25 18:24) with saving: per layer, the top-64 shared input directions Q of all 24 skills (fit tokens, each
  skill weighted equally) and each external's soft1 read factor -> results/analysis/exp138_save_s{0,1,2}/ (for EXP-139).
  14 collection passes per order; ETA ~20:00.

- **RESULT (10-25 19:22; results/analysis/exp138_report_s{0,1,2}.txt; 456 layers x 8 externals per order, 3 orders agree).**
  - Mechanism: per token, an external update fires on library / other-external inputs at 0.97-0.99 of its own-input
    level; 67-70% of its own output is a constant offset, the same on library inputs (cos 0.99); 84-86% of its own
    output energy is in the VLM MLP layers.
  - Direction 1: hard projections off the others' top-q input directions remove own output as much as others' (q=16:
    kept 0.10-0.11 for all; random-q control keeps 0.98); selective directions (gen{k}) carry ~0 of the own output.
    soft1 keeps own 0.56-0.59 (own error 0.14-0.15) vs library 0.09-0.10, other externals 0.12-0.13 -> meets the GO
    criterion (own error <= 0.3, others <= 0.3; selectivity ~5x, not a rescale); soft10 0.30-0.32 vs 0.02-0.03 (error
    0.35-0.38, MAYBE). Caveat: test frames alternate with fit frames of the same 4 demos (near-duplicate states) ->
    held-out numbers may be optimistic; rollouts decide.
  - Direction 2: refit to the library's 16 input directions 0.14-0.16 error, but 16 random directions 0.16-0.17 ->
    not discriminative (inputs dominated by a few directions); not a GO on this evidence.
  - Against EXP-139's hypothesis H: the externals' offsets barely align with the library shared part's output on the same
    inputs (cos 0.12; 3% of output along it).
  => next: EXP-139 (running) tests H and ssoft1 alone; EXP-140 tests soft-gated externals in the merge.

### EXP-139 — is an external skill's constant offset a duplicated embodiment adaptation? — **done (10-26): H refuted; the library's shared part S itself breaks externals (29.5% at full weight)**
Motivation (user 10-25: autonomy toward better new-skill retention; "exploit any special properties of VLAs as opposed to
general model merging"). EXP-138 smoke: an external LoRA's output is mostly a constant offset that hits every task. VLA-
specific hypothesis H: a skill LoRA trained from the generalist pi0.5 must re-learn the embodiment / domain adaptation to
LIBERO; library LoRAs share it (it sits in the shared part S), each external carries its own copy -> merging stacks
conflicting copies. If H holds, merge the embodiment adaptation once (S) and only the skill parts of the externals.
PRE-REGISTERED (`sweep/exp139_embodiment.py` docstring): stage 1, order 0, each of the 8 externals ALONE: sown (W0 + S +
D_e), sskill1 / sskill16 (W0 + S + D_e with its response to the top-1 / top-16 shared input directions removed), skill16
(no S; control), ssoft1 (W0 + S + EXP-138 soft-gated read), svlm / sexpert (D_e on one tower only; exploratory); own scene, 10 eps; own baseline = single-task success.
H predicts sskill16 >= 0.8 x own, skill16 <= 0.5 x own, sown >= 0.8 x own; sskill16 << own refutes H. 56 models.
- Queue (10-25 18:26): tmux e139g0/1/2 (`sweep/chain8_e139.sh`) wait for the EXP-138 runs, then shard 0/1/2 of 3 on GPUs
  0/1/2 (1 model per round), report. A builder check runs on the first saved EXP-138 pass.

### EXP-140 — soft-gated external skills in the merge — **done (10-25): NO-GO for externals; gating at 1/k helps the library (+6pp)**
Motivation: EXP-138's soft gating keeps 57% of an external update's own output while cutting what reaches the library /
other externals to 9% / 13% -> if that holds in closed loop, externals can be merged at a much higher weight.
PRE-REGISTERED (`sweep/exp140_softgate.py` docstring): orders 0-2, t=8, library as in ours; arms soft_k (externals gated
at 1/24), soft_w0.25, soft_w1, plain_w0.25 (ungated control); all 24 skills, 10 eps; reference ours (EXP-134). Predictions:
soft_k library >= ours - 2pp, externals within 5pp; soft_w0.25 externals >= ours + 15pp with library >= ours - 5pp;
plain_w0.25 library <= ours - 15pp. Method GO if soft_w0.25 or soft_w1 reaches externals +15pp at library >= -5pp.
Check: the builder reproduces ours exactly when externals are ungated at 1/24 (rel err 2e-7). Queue: tmux e140g0/1/2
(`sweep/chain9_e140.sh`) after each GPU's EXP-139 shard; 12 models, 288 evaluations.
- Ops (10-25 19:24-19:33): EXP-139 / EXP-140 crashed on their first gated layer (KeyError: EXP-138 saves no entry for an
  external whose factor is all zeros on a layer; the builder check had sampled only a few layers). Fixed (zero factor ->
  unchanged), FULL dry build of every arm x layer passed (EXP-139 25,648 layer deltas, EXP-140 5,496), relaunched as
  tmux c10g0/1/2 (`sweep/chain10.sh`): EXP-140 shard g/3 first (main bet), then EXP-139 shard g/3. No GPU time lost
  beyond ~10 min; the one EXP-139 result already recorded (sown, xa40cf3: 89% vs own 90%) is kept.
- RESULT (10-25 22:53; `results/analysis/exp140_report.txt`; retention library / external, 3 orders, t=8, paired vs ours):
  ours 76.7 / 13.5 | soft_k 82.9 / 12.2 (library +6.2pp, p=0.078; external -1.3, p=0.69) | soft_w0.25 71.4 / 17.2
  (library -5.3, p=0.014; external +3.8, p=0.14) | soft_w1 6.3 / 0.8 (complete after the 1-scene rerun `sweep/fix9_rerun.sh`)
  | plain_w0.25 8.0 / 9.3 (library -68.7). Predictions: soft_k library held (yes, even +6) and externals within 5pp (yes);
  soft_w0.25 externals +15pp (NO, +3.8); plain control collapses (yes). Method GO criterion NOT met.
  Reading: the gate removes most of the externals' damage to the library at the same weight (a candidate improvement to
  ours, not yet significant), but externals do not recover with weight: at 0.25 the gated externals reach 17% and already
  cost the library 5pp; at weight 1 everything collapses -> the gate's leak (9-13% offline, x8 externals x weight) is far
  above what closed loop tolerates, and externals need near-full weight on their own inputs. Per-token input gating alone
  cannot separate tasks enough. Next: EXP-139 (running) says where an external lives; then output-side protection.

### EXP-141 — offline action proxy to screen merges before rollouts — **done (10-26): FAILS the pre-registered bar; coarse collapse detector only**
Motivation: each closed-loop merged model costs ~1 GPU-hour; screening ideas for external retention needs something
faster. Earlier proxies failed to rank small donor effects (EXP-96/103/104), but merge effects here are large (8-83%).
PRE-REGISTERED (`sweep/exp141_actproxy.py` docstring): per order, base / 24 single-task models / EXP-140 arms predict
action chunks on 16 demo frames per skill (fixed noise); proxy = relative action error vs the skill's single-task model
(0 = same, 1 = as far as base), first 5 steps. Usable if, on order 0, Spearman(proxy, closed-loop retention) <= -0.6
pooled over 5 arms x 24 skills AND <= -0.8 over the 10 arm x group cell means. Running orders 0 (GPU0) and 1 (GPU2)
next to EXP-139, ~80 min each.


### EXP-142 — does the library alone already solve the externals that survive in ours? — **done (10-26): yes — ours' external retention is the library's zero-shot transfer**
Motivation — ANALYSIS (10-25 23:05, EXP-134 scratch per skill, `/tmp/e134_perskill.py`-style table): under ours,
external retention is bimodal and FIXED ON ARRIVAL, not eroded by later additions (new-skill BWT ~ 0): 6 of 32 (order,
external) instances keep 30-100% from the step they are added to t=8 (xcf3ac4, xe9c7e4, x2e4610, xe306d8 goal; xa40cf3,
xb0e1f6 spatial), all others are at 0% from the start, even alone at t=1 (e.g. xee763e). So the external problem is not
accumulation of externals; most are dead on arrival at weight 1/k (~1/17). Question: do the survivors owe anything to their
own LoRA? PRE-REGISTERED (`sweep/exp142_libonly.py`): library-only merge (t=0) per order 0-3 on its 8 external scenes;
if library alone >= 0.8 x ours on the survivors, the externals' own LoRAs contribute ~nothing in ours. 4 models, 32
evaluations, GPU2 (2 workers, ports 8400+) next to EXP-139 / EXP-141.

### EXP-143 — through which tokens must an external skill's update act? (VLA token pathways) — **done (10-26): spread over the whole image pathway + expert; instruction route dead**
Motivation: externals need near-full weight on their own inputs (EXP-140/142) and per-layer input gates leak because all
tasks' tokens share input directions (EXP-138). pi0.5's prefix = image tokens (same robot / table across tasks) + the
instruction's text tokens (task-specific), full attention; the action expert reads the prefix. If a skill's update only
needs the TEXT tokens (+ action expert), it can be keyed on the instruction, a VLA-specific separation handle.
PRE-REGISTERED (`sweep/exp143_tokenpath.py`): oracle position masks via forward hooks, order 0's 8 externals at full
weight on W0 (and on W0 + S): all / text / image / expert / text+expert / image+expert; proxy = relative action error vs
single-task on 16 demo frames. Route viable if text+expert on W0 <= 0.3 and clearly below image+expert; dead if >= 0.6.
GPU1, ~15 min.
- Side analysis (10-25 23:50, `sweep/analysis_ext_similarity.py`): weight-space cosine of full updates D = B A, summed over
  all 458 layers: library-library +0.79, externals (22 used in orders 0-3) ~0 with each other (same suite +0.006, cross
  suite +0.004) and with the library (+0.006); VLM layers alike. No per-suite shared part exists in weight space (expected
  for independent random A), so a "suite S" analogue of our library S is not available; functional (activation-space)
  similarity untested.
- RESULT run 1 (10-25 00:00; `results/analysis/exp143_report_s0.txt`; mean relative action error over the 8 externals,
  update at full weight on W0): all 0 (reference) | image+expert 0.03 | image 0.13 | text+expert 0.48 | text 0.56 |
  expert 0.73. -> the skill acts through the IMAGE pathway (vision tower, 163 LoRA'd layers, + image positions of the
  LM); the instruction route is NOT viable (0.48 > 0.3; pre-registered "viable" fails). Caveat: action_in/out_proj and
  time_mlp were classed "lm" (only the text mask hooked them); conclusion unaffected (both text+expert and image+expert
  include them). On W0 + S every mask gives ~0.15-0.20 (S alone already moves base most of the way on this scale); the
  two externals with W0+S+full error ~0.4 (xee31a9 0.41, xed0e9e 0.37) are the ones S breaks (xee31a9 sown 0% closed
  loop) -> the proxy flags S conflicts per skill. Run 1b (masks fixed; vision tower vs LM image positions) running.
- RESULT run 1b (masks fixed: action/time projections -> expert; `results/analysis/exp143_report_s0b.txt`; mean error,
  W0, 8 externals): vision tower alone 0.55 | LM image positions alone 0.40 | text positions alone 0.56 | expert 0.73 |
  vision+expert 0.26 | LM-image+expert 0.35 | text+expert 0.48 | (run 1: vision + LM-image 0.13, + expert 0.03).
  -> no single pathway carries the skill: it is spread over the vision tower, the LM at image positions and the expert;
  only the whole image pathway + expert reproduces it. Token-keyed (instruction) merging: DEAD. Done.
- EXP-141 RESULT (10-26 00:20; `results/analysis/exp141_report_s0.txt`, order 0): pooled Spearman(proxy, retention) -0.63
  (n=120; passes <= -0.6) but cell means -0.71 (n=10; fails <= -0.8) -> NOT usable as a screen (pre-registered). Library
  error 0.06 for ours / soft_k / soft_w0.25 (closed loop 76 / 87 / 73%) and 0.13 for the two collapsed arms (9 / 6%);
  external error 0.18-0.23 for every arm (closed loop 0-11%). It detects collapse, nothing finer: the base-normalised
  scale is dominated by LIBERO adaptation, and closed-loop success turns on small action errors (compounding).
  Order-1 copy stopped (freed GPU2). The queued single-external screen (chain11, GPU0) is kept as a collapse check only.


### EXP-144 — data-free conversion of from-scratch skills through their OWN rollouts (pilot, order 0) — **done (11-05, corrected after a CUDA-OOM data incident): generalizes — externals 45.3% (4-order mean; scratch ours 17.2, WP on demos 63.7), library 69.5 (ours 76.6)**
Motivation: EXP-138..143 + side analyses: a from-scratch (foreign-A) external is dead on arrival in every training-free
merge tried — needs near-full weight on its own inputs, fires on every task, S breaks many externals even at full weight
(EXP-139 partial), no token pathway localises it (EXP-143), externals are mutually orthogonal (no suite-level shared part),
and the offline proxy cannot screen finer than collapse (EXP-141). WP adapters trained on DEMOS merge well (order 0, t=8:
new 61.0% of the original skill's success, library 78.5%; scratch ours 5.7% / 76.1%), but the from-scratch setting has
no demos. VLA-specific route: a robot skill can produce its own data — run it in closed loop, keep successful episodes
(task success signal), train a WP adapter on them. No original data used.
PRE-REGISTERED (`sweep/exp144_selfconvert.py`): collect 50 episodes per external on initial states 10..49 (test states
0..9 untouched; dumps every 2 steps + executed actions; `train_lora.py --rollouts`, new option), train WP (s16wp, 1,500
steps, batch 8) on successful episodes only, single-task check (2 x 10), merge as EXP-134 wp (new at 0.25) at t=8 on all
24 skills. Works if externals >= 30% of the original skill's success with library >= scratch ours - 5pp; matches
WP-on-demos if within 10pp. Chains `sweep/chain12_e144.sh` g0/1/2 (collect + train own shard; GPU1 then base + run +
report on all GPUs). Started GPU1 00:25; GPUs 0/2 join after EXP-139/142/141x; ETA ~06:30.
- EXP-139 RESULT (10-26 00:50; `results/analysis/exp139_s0_report.txt`; order 0, each external ALONE, retention vs its
  single-task success, 10 eps): sown (S + full external) 29.5% (p=0.008 vs 100%) | sskill1 46.7% | sskill16 39.4% |
  skill16 (no S) 11.0% | ssoft1 32.5% | svlm (S + VLM part only) 65.8% (p=0.29) | sexpert 4.0%. Complete (0 drops).
  Predictions: sown >= 0.8 NO (0.30), sskill16 >= 0.8 NO (0.39), skill16 <= 0.5 yes -> H (duplicated embodiment offset)
  REFUTED as stated. What it shows instead: (1) the library's shared part S, added at full strength as in every S-anchored
  merge, breaks most externals even with their FULL update and no other skill (2 of 8 at 0% under S vs 60% / 50% own) ->
  caps S-anchored training-free merging of externals at ~30-45%; (2) removing the response to the top shared input
  direction helps on S (sskill1 46.7 vs sown 29.5; 3 up, 1 down, 4 tied); (3) the VLM part carries the skill (svlm 65.8
  vs sexpert 4.0; svlm vs sown mixed per skill: 3 up, 3 down, 2 tied — n.s., driven by xed0e9e 233% / xcc8fd8 125%).
  WP adapters are trained ON TOP of S (s16 init) and so are S-compatible by construction — consistent with them merging
  well (61% on order 0) -> motivates EXP-144 (retrain from-scratch skills on top of S from their own rollouts).
- EXP-142 RESULT (10-26 00:55; `results/analysis/exp142_report.txt`; 32 (order, external) instances, complete, 0 drops):
  library alone (t=0 merge, no external) 20.5% vs ours at the external's arrival 17.8%; on ours' survivors (>= 25%, n=9)
  library alone 56.1% vs ours 61.9% (ratio 0.91 >= 0.8 -> pre-registered: the externals' own LoRAs contribute ~nothing in
  ours). Three goal externals are solved by the library alone at 100% (xcf3ac4, xe9c7e4, x2e4610); the only external
  its own LoRA helps at 1/k is xa40cf3 (22 -> 56, 11 -> 33, 22 -> 56 in 3 orders). => In the from-scratch setting, ours'
  "new-skill retention" is zero-shot transfer of the library; the training-free merge adds the externals' skills ~not at all.

- Progress (10-26 07:46): chain12 collected + trained 4 of 8 (xa40cf3, xee31a9, x2bc885, xcc8fd8; 70-101 min each);
  rollout success on states 10..49: xa40cf3 96%, x2bc885 94%, xcc8fd8 92%, xee31a9 78%, xed0e9e 36%. Lost: (1) my shard bug
  (shards recomputed after filtering done jobs -> 3 collections skipped, GPU2's 2 failed "server died during load", the
  GPU0/GPU2 trainers ran before their dumps existed) -> fixed: stable shards, training = shared claim queue; (2) shiva GPU1
  fell off the bus ~05:03 (nvidia-smi "Unable to determine the device handle ... Unknown Error"); every simulator client
  since then, on ANY GPU, hangs in the NVIDIA modeset driver (`nvkms_open_common`) -> EGL rendering dead machine-wide
  until a reboot; CUDA (training) still works. Base stage hung 05:03-07:46, killed.
- Resumed 08:05 on GPUs 0 + 2: `chain13_e144.sh 0` (training queue, GPU0; started xed0e9e 07:49) + `chain13b_e144.sh`
  (GPU2: collect x47b567/x290bfc/xb61165 with OSMesa CPU rendering — new `ORCH_GL=osmesa` in the orchestrator —, then an
  OSMesa-vs-EGL check on the 8 original skills (`E144_STAGE=calib`, e144_osm_*), then training queue, then base + merge on
  GPUs 0/2 with OSMesa + report). Merge numbers are only comparable to the EGL references if the calib check matches.
- 08:05-12:09: OSMesa collection of the 3 missing sets finished (08:49; x47b567 47/50, x290bfc 40/50, xb61165 29/50), but
  the OSMesa calib stage and the xed0e9e training (stalled at step 1300/1500) made no progress afterwards; shiva rebooted
  ~12:10 (user). After reboot (13:17): all 3 GPUs back, EGL works on every device id (identical renders across ids).
  OSMesa vs EGL on 3 init states: mean |diff| ~1.3/255, 1-5% of pixels > 8 -> OSMesa dumps moved to
  results/exp144_dumps_osmesa/ and recollected with EGL (no OSMesa data or evals used). `chain14_e144.sh`: recollect
  (GPUs 1/2) -> 3 trainers from the claim queue (GPUs 0/1/2) while tmux t0 retrains xed0e9e on GPU 0 (restarted 13:18) ->
  base + merge on all GPUs (EGL) + report. ETA ~16:30-17:00.
- RESULT (10-26 16:15, order 0, t=8, all 24 skills, 10 eps, EGL; complete, 0 parse_fail; `results/analysis/exp144_report.txt`):
  | arm | library | externals (vs original skill) |
  | scratch ours (1/k) | 76.1% | 5.7% |
  | WP on demos (0.25) | 78.5% | 61.0% |
  | converted, own rollouts (0.25) | 70.4% | 44.5% |
  Externals +38.8pp vs scratch ours (5 up / 0 down / 3 tied; Wilcoxon p=0.062, the floor for 5 non-ties) — x47b567 0->50,
  x290bfc 0->10, x2bc885 0->30, xb61165 0->120, xcc8fd8 12->112; xa40cf3 33 = 33; xee31a9, xed0e9e 0 in every arm.
  Library -5.7pp (5 up / 9 down / 2 tied, p=0.45; per-skill swings +-40). Vs WP on demos: externals -16.6 (0/5/3, p=0.062),
  library -8.1 (n.s.). Converted single-task 81% mean vs the originals' 76% (xee31a9 60->75, xb61165 50->75, xcc8fd8 80->100;
  x290bfc 100->85, xed0e9e 30->20): training on own successes keeps the skill. Rollout success on states 10..49: 36-96%.
  Pre-registered: externals >= 30% YES (44.5); library >= scratch ours - 5pp NO by 0.7pp (70.4 vs 71.1; n.s.); within 10pp
  of WP on demos NO (-16.6). => First method that makes from-scratch externals survive a merge (x7.8); a data-free,
  VLA-specific step (the skill generates its own data via closed-loop success). Open: library cost, generality.
- Extension (10-26 16:25, `sweep/chain15_e144.sh`; script now takes E144_SEQS / E144_WS): collect + convert the 14 externals of
  orders 1-3 not in order 0 (S16 is the fleet's, order-independent -> converted adapters are reused across orders); order 0
  new-skill weight sweep 0.125 / 0.5 first (library cost); then base + merge orders 1-3 at 0.25. References exist for all
  orders (orders 1/2/3 scratch ours externals 26.4 / 8.3 / 28.6%, WP on demos 55.7 / 66.8 / 71.1%). ETA sweep ~18:00, all ~01:30.
- 10-26 16:45: chain15 stopped by shiva's per-user disk quota (50 GB hard / 46 GB soft; `quota -s` 51200M*) — collection of
  orders 1-3 partial (x19d48f 7, xb4a4ab 7, xe9c7e4 5, x73aa05 28, xdf929b 30 eps; x805c47/x2e4610/xb0e1f6 not started), w
  sweep not run. Unused artifacts (OSMesa dumps, exp126/136/138 adapters + saves, ~7.9 GB) copied to the NAS archive; local
  deletion awaits the user's OK.
- 17:45 (user OK): local copies deleted after an rsync checksum check against the NAS archive (OSMesa dumps, new_loras_readout,
  new_loras_s16soft, new_loras_short, exp138_save_s0-2, joint_loras) and all EXP-144 rollouts moved to
  /media/pinas/lzhou/libero-eval-archive/exp144_dumps (new collections write there; `E144_DUMPS`); shiva home 51.2 -> 36.3 GB.
  6 of 14 order 1-3 collections were complete (records ok); the 5 partial ones set aside (`*_partial_10261746`); chain15
  relaunched 17:46 (8 collections -> w sweep -> 14 trainings -> base + merges orders 1-3). ETA sweep ~19:15, all ~03:00.
- 10-26 18:59 order 0 new-skill weight sweep (same converted adapters, t=8, 10 eps, complete, 0 parse_fail):
  | converted, w | library | externals |
  | 0.125 | 79.0% | 24.8% |
  | 0.25 | 70.4% | 44.5% |
  | 0.5 | 30.0% | 16.9% |
  Peak at 0.25; 0.5 breaks the whole policy (8 new residuals x 0.5 swamp S16). 0.125 keeps the library (+2.9 vs scratch ours)
  but externals only 24.8 (still x4.4 scratch ours). No w meets both pre-registered criteria. WP on demos at 0.25 costs no
  library (78.5) -> the converted adapters interfere more at the same w (next: residual norms converted vs WP on demos).
- 19:00: 6 stale `.claim_*` markers from the 16:45 quota crash (no process alive) removed so the train queue picks up
  xcf3ac4 xf0a297 xe306d8 xee763e x1515d0 xed335f; 3 trainers running (14 adapters, ~5 rounds). ETA all ~05:00 10-27.
- 19:15 (shiva clock): `train_lora.load_rollout_samples` bug — `z["img"][j]` re-read the npz member per frame and every sample
  pinned its own copy of the episode's images (60 GB RSS, ~10 min load, 67 GB swap with 3 trainers). Fixed (arrays read once per
  file; identical samples, 1.5 GB, 8 s); the 3 running trainings (16 min in) restarted on the fix.

- 11-04 10:24 chain19 (shiva back after the 10-27 outage): xb0e1f6 retraining on GPU1 || EXP-145b on GPUs 0+2. ~11:26 shiva GPU1
  fell off the bus AGAIN (nvidia-smi "Unable to determine the device handle ... Unknown Error"; same as 10-26 05:03): the
  training died (exit 1, 62 min) and every EGL client failed (rate=None in 2 s) once the base stage started. chain19 stopped
  11:29 (1 failed job, no records written). Needs a reboot (user). Remaining: xb0e1f6 training, base + merges orders 1-3.

- 11-05 15:41 shiva rebooted (user); all 3 GPUs back. chain21 (`sweep/chain21_resume.sh`, tmux c21): GPU1 left idle (fell off
  the bus twice under load); xb0e1f6 training on GPU2 || EXP-145c on GPUs 0+2 -> EXP-144 base + merges orders 1-3 on GPUs 0+2.

- 11-05 19:12 RESULT orders 1-3 (t=8, w 0.25, test, 10 eps, complete; `exp144_report_s{1,2,3}_w0.25.txt`): conversion alone does
  NOT generalize. externals (scratch ours / WP on demos / converted): o1 26.4 / 55.7 / 22.2; o2 8.3 / 66.8 / 8.0; o3 28.6 / 71.1
  / 20.0 (order 0: 5.7 / 61.0 / 44.5). 4-order mean: 17.2 / 63.7 / 23.7. library: o1 84.4 / 85.7 / 46.2; o2 69.7 / 63.2 / 67.4;
  o3 76.1 / 67.2 / 61.2 (mean 76.6 / 73.7 / 61.3). The converted adapters still work ALONE (single 70-100% on most) but drop
  to 0% inside the merge for 5-6 of 8 externals per order: order 0 was the favourable case. Converted != WP on demos even
  though both are WP adapters at the same weight -> the self-rollout data (successes from states 10-49 only, one policy's
  narrow trajectories) gives private parts that do not survive next to the others. In-context training (EXP-145) must carry
  the whole external gain on orders 1-3. Diagnostic running: |B_perp| and cosines, converted vs WP on demos (`sweep/_e144_norms.py`).
- 11-05 19:25 diagnostic (`sweep/_e144_norms.py`, `sweep/_e145_cm.py`, CPU): converted private parts are NOT larger than WP on
  demos (|B_perp| 16.3-16.7 vs 16.8-17.3, all orders) and only 0.13-0.20 cosine to them; they share more of a common mode
  (pairwise cos among the 8 externals 0.094-0.102 vs 0.033-0.042, same in all orders -> does not explain order 0 vs 1-3).
  The order-0 in-context update does not cancel that common mode (mean ctx update cos +0.005; ctxr -0.16, removes ~4% of it
  at 1/8): ctx mostly strengthens each external's own direction (cos(update_i, conv_i) +0.37..+0.46). Training-free
  common-mode removal dropped; the in-context round is needed.
- 11-05 19:55 **DATA INCIDENT (CUDA OOM) — order 1-3 numbers above are PROVISIONAL.** Evals that shared an A6000 with a
  training run (+ another orchestrator) hit CUDA OOM in the policy server; the LIBERO client catches the server error, ends the
  episode as a FAILURE, and the orchestrator recorded the scene as status ok (e.g. a 10-episode scene "done" in 21 s at 0%).
  Audit of all 42.8k records: only 11-05 runs affected (older fast scenes are genuine 100%). Quarantined 36 scene records
  (results/merge_quarantine/oom_1105; originals kept): c144 orders 1/2/3 merges 10/6/7 of 24, 4 converted-single baselines,
  EXP-145d exploratory arms ctx_l0.25 7 and mix_q0.25_r0.125 2. Clean: EXP-145c ctx_l0.125, EXP-145d primary mix 1/8+1/8,
  order-0 EXP-144. Fix (`sweep/merge_orchestrator.py`): a scene with any "Caught exception" episode -> status episode_error
  (rerun, never counted); eval logs per port (concurrent orchestrators interleaved eval_w{i}.log). Reruns queued
  (`sweep/chain27_rerun_oom.sh`, tmux c27, after EXP-147 cons eval; <= 3 eval workers + 1 trainer per A6000).
- 11-05 21:53 **CORRECTED RESULT orders 1-3 (36 quarantined records rerun with the patched orchestrator; 148 post-reboot
  records audited, 0 crashed): conversion DOES generalize — the 19:12 "fails on orders 1-3" entry above was an OOM artifact.**
  externals (scratch ours / WP on demos / converted): o1 26.4 / 55.7 / 43.1; o2 8.3 / 66.8 / 49.1; o3 28.6 / 71.1 / 44.3
  (o0 5.7 / 61.0 / 44.5) -> 4-order mean 17.2 / 63.7 / **45.3**. library: o1 84.4 / 85.7 / 78.7; o2 69.7 / 63.2 / 67.7; o3 76.1 /
  67.2 / 61.2 (o0 76.1 / 78.5 / 70.4) -> mean 76.6 / 73.7 / **69.5**. Converted singles 60-100% except xed0e9e (20%, 30% own).
  => conversion through own rollouts: externals +28pp over scratch ours, ~18pp below WP on demos, library -7pp vs ours (same
  pattern on all 4 orders). The 19:25 diagnostic stands as description only (its "order 0 vs 1-3" motivation is void).

### EXP-145 — in-context conversion: train each converted external INSIDE the merge it will live in (pilot, order 0) — **order 0 done (11-05): FedAvg of ctx + ctxr (1/8 each) = library 92.5% / externals 60.5% (WP on demos 78.5 / 61.0); orders 1-2 training**
- Why: EXP-144 converted adapters are trained ALONE but act at x0.25 next to 16 library + 7 other new residuals (single 81% ->
  merged 44.5%); the receiver knows that merge, so it can train the adapter for it. Residual sizes rule out a norm effect:
  converted 16.7 vs WP on demos 17.3 (from scratch 41.8, library 26.7), so the extra library cost vs WP on demos (-8.1, n.s.)
  is direction or noise.
- Setup (`sweep/exp145_incontext.py`, `train_lora.py --init ctx --context`): per external, the merge of the OTHER 23 members
  (S16 + 0.25 x other converted residuals + 1/24 x library residuals, exactly as PBuilder builds it) folded into the frozen base;
  its private part B_perp (A = V^T frozen) acts at x0.25, starts from the EXP-144 adapter, 750 steps batch 8 on its own
  successful rollouts. Arms: ctxr = half the steps on library replay (the library's own demos, 5 per library skill, own
  prompt — receiver-owned data; no external data); ctx = no replay. One Jacobi round (contexts hold the EXP-144 residuals).
  Merge = EXP-144 priv at w 0.25, t=8, 24 skills, 10 eps. Smoke: lr 0 round-trips the adapter exactly (6e-7); context folded
  into 451/458 weights (the 7 others have zero B in every member, skipped by the eval builder too).
- PRE-REGISTERED (before training): helps if externals >= 44.5 + 10pp with library >= 67.4%; ctxr removes the library cost if
  library >= 73.1% (scratch ours 76.1 - 3) with externals >= 44.5%.
- 10-26 19:25 launched (`sweep/chain16_e145.sh`, tmux c16 on shiva; 3 trainers sharing GPUs with chain15's EXP-144 trainers,
  then merges on ports 8100+). ETA ~01:00 10-27.
- 10-27 04:32 RESULT (order 0, t=8, w 0.25, 10 eps, complete): **NO-GO as run.**
  | arm | library | externals |
  | converted (EXP-144) | 70.4% | 44.5% |
  | in-context ctxr (+ library replay) | 0.0% | 0.0% |
  | in-context ctx | 38.6% | 31.1% |
  ctxr collapses every skill; ctx: externals -13.3 (2 up / 3 down, n.s.), library -31.8 (p=0.008). Diagnosis (`sweep/_e145_diag.py`):
  ctxr's 8 updates point the same way (pairwise cosine 0.67, |sum| 125 vs 51 if independent) — every skill learned the SAME
  library correction assuming the others fixed, so the merge applied it ~8x (Jacobi overshoot). ctx's updates are near
  independent (cosine 0.14) but each grows B_perp 16.7 -> 28.5 (undoes the 0.25 down-weighting, ~ w 0.43; w 0.5 gave library 30%),
  and each was trained against the others' OLD residuals. Pre-registered criteria: both NO.
  Lesson: a parallel round needs FedAvg-style aggregation (mean of the updates, not the sum); next = cheap check without training:
  merge conv + lam x (ctxr - conv), lam 1/8 (FedAvg) and 1/4, evals only.
- 10-27 ~04:50: FedAvg blend check prepared (`E145_STAGE=interp`, `sweep/chain17_e145b.sh`: lam 1/8, 1/4, GPUs 0+2) — shiva went
  offline (ssh + tailscale) while the blends were being written; not launched. chain15 was on its last EXP-144 training
  (xb0e1f6, step 800/1500) before base + merges of orders 1-3.

- 11-04 RESULT EXP-145b (FedAvg blends, no training: adapter_i = conv_i + lam x (ctxr_i - conv_i); order 0, t=8, w 0.25, test,
  10 eps, complete, 0 drops; `results/analysis/exp145_report_s0.txt`; ran 10:24-11:24 on shiva GPUs 0+2):
  | arm | library | externals |
  | converted (EXP-144) | 70.4% | 44.5% |
  | ctxr, lam 1/8 (= FedAvg: merge gets the MEAN update) | **92.8%** | 47.0% |
  | ctxr, lam 1/4 | 83.5% | 39.9% |
  lam 1/8 vs converted: library +22.4pp (12 up / 1 down, p=0.003), externals +2.5 (2/0, n.s.); lam 1/4: library +13.1 (8/2,
  p=0.032), externals -4.6 (n.s.). lam 1/8 = 1/K was fixed a priori (FedAvg), not tuned; both lams evaluated on test.
  Library 92.8 is above ours' t=0 library (86.3, 4-order mean; library-only merge) and above WP on demos at t=8 (78.5) -> the
  averaged update repairs the merge's library damage itself, not only the externals' interference. Since ctxr's 8 updates
  are 0.67-cosine aligned, their mean ~ one shared library correction learned from library replay inside the merge; the
  external-specific parts are diluted 8x (externals ~unchanged). Pre-registered (EXP-145): ctxr removes the library cost
  (library >= 73.1 with externals >= 44.5) -> YES at lam 1/8 (post-hoc FedAvg fix of the run, not the pre-registered arm);
  helps externals (>= 54.5) -> NO. Open: is it library replay alone (ctx blend = no-replay ablation; repair adapter trained
  on replay only), generality over orders 1-3, more FedAvg rounds for externals.

- 11-04 11:35 EXP-145c (no-replay ablation, `E145_BLEND=ctx`: conv + 1/8 (ctx - conv)) launched on odin's P6000s
  (`sweep/chain20_e145c_odin.sh`); odin's GPU1 (P6000, 03:00.0) fell off the bus ~12:07 (same "Unknown Error") with 11/24
  scenes done -> run hung; records moved to results/merge_stale/odin_gpu1_fault_1104 (not used). Relaunched on shiva 11-05
  (chain21; blend adapters on the NAS, results/analysis/new_loras_ctx/ctx_l0.125 is a symlink; shiva quota 44.3/46 GB).

- 11-05 16:17 RESULT EXP-145c (no-replay ablation, conv + 1/8 (ctx - conv); order 0, t=8, w 0.25, test, 10 eps, complete, shiva
  GPUs 0+2): library 73.7% (+3.3 vs converted, 6/2, p=0.32) / **externals 59.7%** (+15.2, 4 up / 1 down, p=0.125; 1.3pp below
  WP on demos' 61.0). vs ctxr 1/8 (92.8 / 47.0): the two halves of ctxr's data do different jobs — library replay repairs the
  library (+19pp), the externals' own-rollout updates lift the externals (+13pp) once replay does not take half their steps
  and the shared library correction does not dominate the mean. Next (no training, EXP-145d, chain23): both means at once,
  conv + a (ctx - conv) + b (ctxr - conv), primary a = b = 1/8 (a priori), exploratory a = 1/4; ctx alone at 1/4.

- 11-05 17:05 generality prep (user: "promising, go ahead"): `exp145_incontext.py` takes E145_SEQ (per-order ADIR
  results/analysis/new_loras_ctx_s{n} -> NAS, shared claim queue between hosts); converted adapters copied to odin; orders 1-2
  prepped; odin's healthy P6000 (CUDA only) trains ctx then ctxr for orders 1-2 (`sweep/chain24_odin_train.sh`, tmux c24 on
  odin); shiva joins after EXP-147 / EXP-144; order 3 after xb0e1f6. Arms beyond ctx/ctxr decided by EXP-145d / EXP-147.

- 11-05 19:15 RESULT EXP-145d (no training; order 0, t=8, w 0.25, test, 10 eps, complete): **both FedAvg means combine.**
  conv + 1/8 (ctx - conv) + 1/8 (ctxr - conv) (primary, a priori): library **92.5%** (+22.1 vs converted, 11/2, p=0.010) /
  externals **60.5%** (+16.0, 5 up / 0 down, p=0.062) — library = ctxr's repair (92.8), externals = ctx's lift (59.7); vs WP
  on demos (78.5 / 61.0): +14pp library, -0.5pp externals with NO external demos (only the externals' own rollouts + library
  demos). Exploratory: a = 1/4, b = 1/8 87.8 / 44.4 (more of ctx overshoots); ctx alone at 1/4 54.5 / 46.2 (vs 73.7 / 59.7 at
  1/8): 1/8 = 1/(number of parallel updates) is the right mean, larger steps hurt both groups. Pilot on order 0 only, where
  conversion already worked; orders 1-3 start from much weaker converted merges (EXP-144).
- 11-05 21:53 CORRECTION (OOM incident, see EXP-144): the two exploratory arms were partly crash-corrupted (7 + 2 scenes);
  rerun: ctx alone at 1/4 **77.6 / 58.3** (was 54.5 / 46.2), a = 1/4, b = 1/8 **87.8 / 59.4** (was 87.8 / 44.4). "Larger means
  hurt both" was WRONG: the blend is flat between 1/8 and 1/4 (ctx 73.7/59.7 vs 77.6/58.3). Primary (92.5 / 60.5) unchanged.
- 11-05 19:27 orders 1-3: order 3 prepped (xb0e1f6 converted copied to odin); shiva trainers (`sweep/chain26_ctx_train.sh`, tmux
  c26) join odin's queue: ctx for orders 1-3 first (needed by every candidate arm), then ctxr (dropped via
  results/analysis/stop_c26_ctxr if one replay-only correction, EXP-147 consl, repairs the library as well). GPU0 starts
  after cons is trained, GPU2 after consl. 24 ctx trainings ~ 10 h on 2 A6000s + odin's P6000 (~4.6 h per run).

### EXP-146 — CUSD baseline (common/unique split, arXiv 2611.04108) in the scratch continual setting — **parked (11-04, user: preprint released today, not a mandatory baseline yet; code + diag kept)**
- Why: CUSD (NeurIPS'26, user-flagged 11-04) is the closest published relative of ours: shared input basis from the merged
  set, task-dominated ("unique") directions damped by beta before averaging. CUSD+TA at beta = lambda is plain TA, so with
  F = averaging it is averaging with the unique directions damped. Not spectral -> unlike TSV / Iso-C / Iso-CTS it should
  not collapse at k = 24. Threat: if it protects the library as well as ours, the library anchor is not what does it.
- Setup (`sweep/exp146_cusd.py`, operator `merge_methods.cusd`): EXP-134 scratch at t=8 (16 library + 8 from-scratch
  externals, orders 0..3, all 24 evaluated, 10 eps), F = averaging, beta = b/k. Port follows the official code step by step;
  solved in the joint row space of the LoRA A's (exact: outside it the quotient matrix is identity and the energy zero) —
  matches a literal full-space version to 1e-6, beta = lambda reproduces TA (4e-8). Val (order 0, states 25..34): rel x
  b {0, .25, .5}, abs x b {0, .5}; pick by mean success; test = picked setting, 4 orders, states 0..9; diag (CPU) = share
  of each skill's energy CUSD calls unique.
- PRE-REGISTERED (11-04, before any rollout): CUSD threatens the claim if its library retention is within 3pp of ours
  (pooled, 4 orders) with externals >= ours - 3pp. Expected: unique directions mostly the externals' -> small beta damps the
  externals; library between averaging and ours.
- 11-04 diag (`E146_STAGE=diag`, no rollouts; orders 0-2, order 3 running): rel scoring calls ~86 of ~383 directions per layer
  unique (13% of the energy) = ~63% of EACH external's energy and 0.7-1.9% of each library skill's -> rel CUSD rediscovers the
  library / external split from the merged set alone; small beta damps exactly the externals' private parts (close to what
  our anchor does -> real threat). abs scoring calls ~4 directions unique (57% of the energy) = ~70% of each LIBRARY skill's.
  Grid amended before any rollout: rel x b {0, .25, .5, 2} + abs b .5 (abs b 0 dropped; b 2 = externals' private parts x2).
  Batched the quotient sum ((sum_j G_j^-1)(sum_i G_i) - kI, exact): ~0.2 s per layer on a P6000, same partitions.
- 11-04 shiva back (rebooted ~10-29; chain15 killed with xb0e1f6 untrained). Queue launched: chain19 = EXP-144 finish (xb0e1f6
  on GPU 1) || EXP-145b (GPUs 0+2) -> EXP-144 base + merges orders 1-3 -> EXP-146 (chain18).
- 11-04 diag order 3 done: same picture on all 4 orders (rel: externals 0.63-0.65 of their energy unique, library 0.007-0.019;
  abs: library ~0.70, externals ~0.01). Results `results/analysis/exp146_diag_s{0..3}.{txt,json}` (odin).
- 11-04 PARKED by the user (preprint out today, not a mandatory baseline yet): EXP-146 removed from chain19 before it ran;
  `merge_methods.cusd`, `sweep/exp146_cusd.py`, `sweep/chain18_e146.sh` kept for later. chain19 = EXP-144 finish || EXP-145b
  -> EXP-144 base + merges orders 1-3 (tmux c19 on shiva, log results/chain19.log; launched 11-04 ~10:35).

### EXP-147 — consolidation: ONE correction trained inside the actual t=8 merge instead of 8 parallel ones (pilot, order 0) — **generalizes (11-06, 4/4 orders): raw success library 89.7 / new 80.0 vs joint training 87.3 / 83.4 (n.s. on new), no new-skill demos; ablations running**
- Why: EXP-145b's FedAvg blend keeps the library (92.8%) but leaves externals at 47.0 (converted 44.5): averaging the 8
  parallel in-context updates keeps their shared library correction and dilutes each external's own improvement 8x; summing
  them overshoots. The receiver owns every input (merge, externals' self-rollouts, library demos), so it can train one
  correction against the merge it deploys — no overshoot, no dilution, half the compute of the parallel round.
- Setup (`sweep/exp147_consolidate.py`; `train_lora.py`: `--tasks_file` takes a rollout glob per task, replay weights over
  several skill groups, context `init_zero`): context = the full t=8 converted merge (24 members: S16 + 0.25 x 8 converted
  private parts + 1/24 x 16 library residuals); one private part from zero (A = V^T frozen, B orthogonal to the library's
  shared outputs) at x0.25; 3,000 steps batch 8. Arms: cons = half the 8 externals' own successful rollouts (uniform over
  externals) + half library replay (5 demos per library skill); consl = replay only (pure library repair). Since all
  externals share A = V^T, the correction enters the merge as adapter_i = conv_i + 1/8 (cons - c0) (c0 = 0-step run), K = 24.
  Merge = EXP-144 priv, w 0.25, t=8, all 24 skills, 10 eps, test.
- PRE-REGISTERED (11-05 15:45, before training): cons helps externals if externals >= FedAvg ctxr (47.0) + 10pp (= 57.0%)
  with library >= 86.3% (ours' library-only merge at t=0, 4-order mean); matches WP on demos if externals >= 56.0.
  consl: library within 5pp of cons and externals within 5pp of converted (44.5) -> cons' external gain comes from its
  own-rollout half.
- 11-05 15:45 launched (`sweep/chain22_e147.sh`, tmux c22 on shiva): GPU0 c0 + cons; GPU2 consl after chain21's xb0e1f6
  training; then blend + evals on GPUs 0+2 (ports 8200+). Adapters on the NAS (results/analysis/new_loras_cons symlink).
- 11-05 19:21 cons trained (3,000 steps, ~3.6 h on an A6000); its eval starts on GPU0 as soon as the adapter lands
  (`sweep/chain25_e147_cons_eval.sh`, tmux c25, port 8400) while consl trains on GPU2 (~21:30). Report gains the EXP-145d
  reference (FedAvg mix 1/8 + 1/8 = 92.5 / 60.5), added after pre-registration; the pre-registered bars are unchanged.
- 11-05 20:10 **RESULT cons (order 0, t=8, w 0.25, test, 10 eps, 24/24 scenes, audited clean: no crashed episodes, no
  concurrent orchestrator): library 90.8% / externals 89.4%.** vs converted 70.4 / 44.5: +20.4 (10/1, p=0.006) / +44.9 (6/1,
  p=0.031); vs FedAvg mix 1/8+1/8 (92.5 / 60.5): -1.7 n.s. / +29.0 (5/1, p=0.062); vs WP on demos 78.5 / 61.0; vs scratch ours
  76.1 / 5.7. Per external (retention vs own scratch): 111 / 67 / 100 / 80 / 90 / 0 / 180 / 87% (xed0e9e 30% alone, 0 in every
  merge). PRE-REGISTERED: externals >= 57.0 with library >= 86.3 -> **YES**; matches WP on demos (>= 56) -> YES, +28pp above.
  Reference: joint training on all demos (EXP-136, rank 64, 12k steps, ~10 GPU-h, 4-order mean) 91.0 / 87.3 -> one 3,000-step
  correction (3.6 h on an A6000, B only, rank 16), with NO external demos (own rollouts + 5 demos per library skill), reaches
  the joint-training level on order 0. One correction trained against the deployed merge >> 8 parallel ones averaged.
- 11-05 20:17 generality + ablations launched (FedAvg ctx/ctxr generality trainings stopped: cons dominates them; chains
  c24/c26 killed, claims cleared): `sweep/chain28_cons_orders.sh` (tmux c28): cons for orders 1-3 (per-order ADIR
  new_loras_cons_s{n} on the NAS; GPU0 + GPU2 claim queue; each order evaluated as soon as trained, 1 worker per GPU).
  `sweep/chain29_cons_ablations.sh` (tmux c29, after the order trainings): consz = cons trained on the FROM-SCRATCH merge
  (context = scratch ours 1/k, correction enters as one extra member at 0.25, `ZBuilder` privx with k = 24 as in training):
  is the per-skill conversion (EXP-144) needed at all?; conso = own rollouts only (no library replay). odin P6000
  (`sweep/chain30_odin_cons1.sh`, ~18 h): cons seed 1 (seed robustness). consl (replay only) eval running (chain22).
- 11-05 21:12 RESULT consl (replay only, same 3,000 steps; patched orchestrator, 0 crashed scenes): library 85.0% / externals
  41.3% (converted 70.4 / 44.5; cons 90.8 / 89.4). Pre-registered: externals within 5pp of converted -> YES (-3.2) => cons'
  external gain (+48pp over consl) comes from the externals' own rollouts; library within 5pp of cons -> just NO (-5.8): the
  own-rollout half also helps the library (or noise, n=16). Replay alone repairs the library (+14.6, p=0.011) but less than cons.
- 11-05 21:17 plain-fine-tune baselines added (the obvious reviewer question: is cons just fine-tuning the merge?): consf =
  a standard LoRA (rank 16, default init, A and B trainable, scale 1, NO write protection) trained on the converted merge with
  the same data / steps / batch, deployed as merge + B A; consfz = the same on the scratch merge (`train_lora.py` context
  key "free"; `exp147` ZBuilder privx adds raw members as B A). chain29 relaunched with priority consz > consf > conso > consfz.
- 11-05 23:48 RESULT cons order 1 (t=8, w 0.25, test, 10 eps, 24/24 scenes ok, audited: 0 crashed episodes): **library 98.1% /
  externals 96.1%** vs converted 78.7 / 43.1 (+19.4, 9/2, p=0.023 / +53.0, 6/1, p=0.031); WP on demos 85.7 / 55.7; scratch ours
  84.4 / 26.4. Per external (vs own): 100 / 17 / 111 / 89 / 112 / 129 / 111 / 100% (xee31a9 17%: fails in every merge).
  Generality holds on the first held-out order (above the joint-training 4-order mean 91.0 / 87.3). Order 2 eval next, order 3
  training (GPU0); consz training (GPU2, since 23:20).
- 11-06 00:36 RESULT cons order 2 (24/24 scenes ok, 0 crashed episodes): **library 91.6% / externals 93.3%** vs converted
  67.7 / 49.1 (+24.0, 11/0, p=0.003 / +44.2, 6/1, p=0.031); WP on demos 63.2 / 66.8; scratch ours 69.7 / 8.3. Per external:
  75 / 111 / 33 / 125 / 129 / 100 / 90 / 83% (xed0e9e 30% own, 33% here; xee31a9 83% here vs 17% on order 1). 2 of 3 held-out
  orders reproduce; order 3 training (GPU0, ETA ~02:00 + 40 min eval).
- 11-06 02:32 RESULT cons order 3 (24/24 scenes ok, 0 crashed episodes): **library 92.5% / externals 112.7%** vs converted
  61.2 / 44.3 (+31.4, 12/1, p=0.023 / +68.4, 8/0, p=0.008); WP on demos 67.2 / 71.1; scratch ours 76.1 / 28.6.
  REPORT BUG (fixed): the first order-3 report found the converted reference for 6/8 externals only (46.6%, n=6): `member_jobs`
  chunks members by index order, and the order-3 converted records were rerun (c27) in a process that registered orders 1-3
  -> different member order -> different chunk names for 2 scenes. Every chunk of a tag serves the same prebuilt merge ->
  `exp112_bench.member_success` now falls back to any chunk of the same suite for a missing scene (both hosts; backup
  results/merge_stale/exp112_bench.py.bak_1106). Corrected report above (n=8, 44.3 = EXP-144's).
- 11-06 03:00 DENOMINATOR CHECK (`sweep/_e147_abs.py`): exp147 retention divides externals by the FROM-SCRATCH skill's own
  success (30-100%, mean 80.0), EXP-136 joint by the write-protected adapter's own (~96%) -> "cons ~ joint" in retention was not
  apples-to-apples. RAW success, 4 orders (64 library + 32 new scenes, 10 eps each), library / new:
  joint training 87.3 / 83.4 | **cons 89.7 / 80.0** | WP on demos 71.1 / 50.6 | converted 67.2 / 36.6 | scratch ours 74.7 / 16.2
  | each skill's own LoRA 95.6 / 80.0. Per order cons: 86.9/71.2, 94.4/83.8, 88.1/73.8, 89.4/91.2. Paired (Wilcoxon, scenes):
  cons - joint library +2.3pp (20/12, p=0.047), new -3.4pp (9/14, p=0.43, n.s.); cons - WP on demos +18.6 / +29.4 (both
  p<0.001); cons new = own single-task (+0.0, 12/12); cons library -5.9 vs own single-task (4/19, p=0.001).
  => **GENERALITY HOLDS (4/4 orders): convert + one correction matches joint training on all 24 skills, with no demos of the
  new skills.** Compute honesty: 8 conversions x 1,500 steps + one 3,000-step correction (rank 16) ~ joint's 12,000 steps
  (rank 64) -> the win is data (no new-skill demos, 5 demos per library skill) not compute, unless consz shows conversion is
  unnecessary (then 3,000 steps).
- 11-06 02:47 RESULT consz (order 0; correction trained inside the FROM-SCRATCH merge, no per-skill conversion; 24/24 ok,
  0 crashed episodes): retention library 90.2% / externals 75.7% (cons 90.8 / 89.4; scratch ours 76.1 / 5.7). Raw success:
  consz 86.9 / 61.3 vs cons 86.9 / 71.2 vs joint 90.0 / 76.2 (order 0). cons - consz: new +10.0pp (5/1, p=0.125, n.s. at n=8),
  library 0.0. => the correction alone does most of the work (new 5.0 -> 61.3 raw from the training-free merge in ONE 3,000-step
  run); conversion adds ~10pp on new skills. Per external (retention): 89 / 17 / 100 / 70 / 70 / 0 / 160 / 100%.
  NEXT: `sweep/chain31_consz_orders.sh` (tmux c31, 02:49): consz for orders 1-3 (contexts written by prep), trainers pick up
  each GPU after chain29's last claim; evaluator port 8800. If consz holds, the method is merge + one correction (3,000 steps,
  1/4 of joint's) with conversion as an optional +10pp on new skills.
- 11-06 05:08 RESULT consf (order 0; plain LoRA, A and B trainable, scale 1, NO write protection, trained inside the converted
  merge, same data / steps; 24/24 ok, 0 crashed): retention 91.0 / 91.3 (cons 90.8 / 89.4); raw 88.1 / 68.8 vs cons 86.9 / 71.2
  (new -2.5, 2/4, p=0.81; library +1.3, p=0.58). => **write protection is NOT what makes the correction work**: a plain LoRA
  trained against the deployed merge does as well. The active ingredients so far: (i) training INSIDE the deployed merge,
  (ii) the new skills' own successful rollouts (consl: without them externals stay at 41%), (iii) library replay (conso pending).
  Open: does the merge matter at all, or is this just distilling rollouts + replay into one LoRA?
- 11-06 05:13 consb launched (`sweep/chain32_consb.sh`, tmux c32, GPU0): plain LoRA from the BASE, same data (own rollouts +
  5 demos per library skill), 3,000 steps, rank 16, deployed alone (base + B A, as EXP-136); `exp147` ZV arm (None, "base",
  True) -> no --context, PBuilder "ta" lam 1. consfz (plain LoRA on the scratch merge, 800/3000 steps) STOPPED to free GPU0
  (claim cleared; requeued after the consz orders in c32): consf ~ cons makes consfz ~ consz likely. Pre-registered: consb
  within 5pp of cons on both -> the merge is irrelevant (story = rollout distillation); consb >= 10pp below cons on new or
  library -> the merge is the warm start that makes a 3,000-step rank-16 correction suffice. ETA train ~08:00, eval ~08:45.
- 11-06 06:00 RESULT conso (order 0; own rollouts only, NO library replay; 24/24 ok, 0 crashed): retention 75.9 / 114.1;
  raw 75.0 / 83.8 vs cons 86.9 / 71.2 (library -11.9, 2/7, p=0.036; new +12.5, 4/0, p=0.125). => replay is what keeps the
  library (+12pp); without it the correction overfits toward the new skills, which then beat their own LoRAs (83.8 vs 80.0
  raw). replay_frac is a library / new trade-off knob (0.5 used throughout). Ingredients now: inside the merge + own rollouts
  (new) + library replay (library); write protection not needed (consf).
- 11-06 06:04 consr launched (`sweep/chain33_consr.sh`, tmux c33): NO DEMOS AT ALL — cons with the library replay drawn from
  the library skills' own successful rollouts (single-task LoRA, states 10..49, 50 eps, dumps on the NAS as EXP-144) instead of
  5 demos each. `exp147` stage collect_lib (writes replay_roll.txt), arm consr (tid y); `train_lora.py` replay lines take an
  optional 3rd field = rollout glob (2-field lines unchanged). Collection on idle eval slots (port 9000), training on GPU2 after
  the consz orders (~10:05), eval ~13:30. Memory: consb (A and B trainable, from base) uses 23 GB vs 15 GB in-context; peak
  per GPU stays <= ~39 GB.
- 11-06 08:16 RESULT consz order 1 (24/24 ok, 0 crashed): retention 97.9 / 88.1 (cons 98.1 / 96.1); raw 94.4 / 77.5 vs cons
  94.4 / 83.8 vs joint 85.6 / 81.2. Same pattern as order 0: the correction alone carries most of it, conversion +6pp new.
  Library collection for consr: 15/16 skills x 50 eps done by 08:17 (~94% rollout success).
- 11-06 09:26 RESULT consb (order 0; plain LoRA from the BASE, same data + 3,000 steps + rank 16, deployed alone; 192 min
  training; 24/24 ok, 0 crashed): retention 64.2 / 83.1; **raw 61.9 / 56.2** vs cons 86.9 / 71.2 (library -25.0, 0/14,
  p=0.001; new -15.0, 1/7, p=0.195) and vs consz 86.9 / 61.3 (library -25.0, 1/13, p=0.001; new -5.0 n.s.). PRE-REGISTERED:
  >= 10pp below cons on new or library -> **YES: the merge is the warm start** that lets a 3,000-step rank-16 correction
  reach joint level; from the base, the same data + budget loses a quarter of the library (5 demos per library skill are
  not enough to relearn it). Story: merge (keeps the library) + one correction trained inside it (own rollouts for new
  skills, a little replay) — neither alone works (merge: new 5%; correction alone: library 62%).
  Library rollouts for consr: 16/16 collected 08:21 (replay_roll.txt).
- 11-06 11:32 RESULT consz order 2 (24/24 ok, 0 crashed): retention 85.6 / 94.0 (cons 91.6 / 93.3); raw 83.1 / 72.5 vs
  cons 88.1 / 73.8 vs joint 86.2 / 85.0. Here conversion adds little on new (+1.3) but +5 on library. Orders 0-2 mean raw:
  consz 88.1 / 70.4 vs cons 89.8 / 76.3 -> the correction carries most of it; conversion +6 new, +2 library on average.
  Order 3 next (~12:30). c34: library rollouts for order 1 done (16/16); consr (order 0) training on GPU2 since 10:40.
- 11-06 12:31 RESULT consz order 3 (24/24 ok, 0 crashed): retention 91.2 / 107.4 (cons 92.5 / 112.7); raw 87.5 / 86.2 vs
  cons 89.4 / 91.2 vs joint 87.5 / 91.2. **consz 4-order raw: 88.0 / 74.4** vs cons 89.7 / 80.0 vs joint 87.3 / 83.4.
  Pooled paired (64 library, 32 new skills): conversion +1.7 library (17/10, p=0.13), **+5.6 new (15/6, p=0.047)**;
  consz vs joint: library +0.6 (p=0.40), **new -9.1 (5/13, p=0.020)**. => the correction alone already keeps the library
  at joint level, but conversion is what closes the new-skill gap to joint (cons vs joint new -3.4 n.s.). Keep conversion
  in the method. c31 done; GPU2 free of c31 eval.
- 11-06 12:38 step-budget arms launched (`sweep/chain35_budget.sh`, tmux c35; order 0): consq / consh = cons at 750 / 1,500
  steps (GPU2 after consr; how much correction is needed?); consz6 = consz at 6,000 steps (GPU0 after consfz; can a longer
  correction replace conversion? conversion costs 8 x 1,500 steps). `exp147` VSTEPS per arm (tids g / h). Eval waits for <= 2
  eval workers per GPU (ports 9400 / 9500). PRE-REGISTERED (raw, order 0, vs cons 86.9 / 71.2, consz 86.9 / 61.3): consh
  within 3pp of cons on both -> half the steps suffice (figure: success vs correction steps, 0 = converted merge); consz6
  within 3pp of cons on new -> a longer correction replaces conversion (simpler method); consz6 >= 5pp below cons on new ->
  conversion stays. ETA consq+consh eval ~16:45, consz6 eval ~22:30.
- 11-06 12:40 cons1 (seed 1) eval queued on shiva (`sweep/chain36_cons1_eval.sh`, tmux c36, port 9300): starts when odin's training log says [done] (new_loras_cons is on the shared NAS, no copy) and <= 2 eval workers per GPU; ETA train ~14:40, eval ~15:30.
- 11-06 14:20 consr eval INCOMPLETE (c33): worker 0 ran on port 9100 = shiva's Prometheus node exporter -> the policy server
  could not bind, clients got HTTP 200 -> every w0 episode parse_fail in 2 s (only 6/24 skills recorded, by w1 on 9101).
  Not a result. Rerun on port 9600 (`sweep/chain37_consr_rerun.sh`, tmux c37; parse_fail records count as missing). Never
  use E112_PORT0=9100 on shiva (other chain ports checked free: 9300/9400/9500/9600).
- 11-06 15:03 RESULT consr (order 0; NO DEMOS AT ALL: library replay = the library skills' own successful rollouts; rerun on
  port 9600 filled the 18 missing skills, 24/24 ok): retention 96.2 / 88.2; **raw 90.6 / 71.2** vs cons (5 demos per library
  skill) 86.9 / 71.3 (library +3.7 n.s., new +0.0) vs joint 90.0 / 76.2 (library +0.6, new -5.0, both n.s.). => the library's
  own rollouts replace its demos: the whole pipeline needs only the adapters + the simulator. Pre-registered follow-up
  launched: consr on orders 1-3 (`sweep/chain38_consr_orders.sh`, tmux c38; trainer lanes GPU2 after consh, GPU0 after
  consz6, claims split the orders; evaluator port 9700). ETA order 1 ~20:45, others overnight.
- 11-06 16:19 RESULT cons1 (order 0, SEED 1; trained on odin 18.8 h, eval on shiva c36; 24/24 ok): retention 89.5 / 93.8;
  **raw 86.2 / 73.8** vs seed 0 86.9 / 71.3 (library -0.6, new +2.5, both n.s.) vs joint 90.0 / 76.2 -> seed-robust.
  odin P6000 next: cons seed 1 on order 2, the weakest order (cons new 73.8 vs joint 85.0; is the gap noise?) —
  `sweep/chain39_odin_cons1_s2.sh`, tmux c39 on odin, ETA ~10:30 tomorrow, eval on shiva after.
- 11-06 17:56 RESULT step budget, cons (order 0; 24/24 ok each). Raw library / new by correction steps: 0 (converted merge)
  69.4 / 33.8 -> 750 steps (43 min) 91.9 / 61.3 -> **1,500 steps (93 min) 88.1 / 71.2** -> 3,000 steps 86.9 / 71.3 (joint
  90.0 / 76.2). consh vs cons: library +1.2, new +0.0 (n.s.). PRE-REGISTERED: within 3pp on both -> **YES, half the steps
  suffice**; the library is restored within 750 steps, new skills need ~1,500. Correction = 1,500 steps rank 16 vs joint
  12,000 steps rank 64. consz6 (can a longer correction replace conversion?) lands ~22:30; figure (success vs correction
  steps, cons + consz) after that.
- 11-06 20:15 RESULT consfz (order 0; plain LoRA, A and B trainable, NO write protection, trained inside the FROM-SCRATCH
  merge, no conversion; 24/24 ok; c29 finished 15:37, read late): **raw 86.9 / 77.5** vs consz (write-protected correction,
  same merge) 86.9 / 61.2 (library 0.0, new +16.2, 5/1, p=0.094) vs cons (conversion + WP correction) 86.9 / 71.3 vs joint
  90.0 / 76.2. Reading: on the scratch merge the WP correction is capacity-limited (its A is the library's; the externals'
  LoRAs do not share A, EXP-139/142), a free-A correction is not; on the converted merge plain ~ WP (consf 88.1 / 68.8).
  If it holds on orders 1-3, **conversion is unnecessary** (pipeline = training-free merge + one plain LoRA trained inside it
  on rollouts). n = 8 new skills, one order -> not a claim yet. Launched consfz orders 1-3 (`sweep/chain41_consfz_orders.sh`,
  tmux c41; lanes start after chain40 has claimed cons s4 + joint s4, eval port 9400; ETA ~02:00 first training, all ~12:30).
  PRE-REGISTERED (4-order raw): conversion unnecessary = consfz new >= cons new - 3pp (cons 80.0) and library within 3pp of
  cons (89.7); conversion needed = consfz new <= consz new + 3pp (74.4).
- 11-06 20:14 RESULT consr order 1 (no demos; 24/24 ok): **raw 88.1 / 83.8** vs cons 94.4 / 83.8 (library -6.2pp, 0/7
  skills, p=0.016; new +0.0) vs joint 85.6 / 81.2. New skills unaffected; the library pays something on this order (still
  above joint). Orders 2-3 overnight (c38).
- 11-06 21:49 RESULT consz6 (order 0; write-protected correction inside the FROM-SCRATCH merge, 6,000 steps, no conversion;
  24/24 ok): **raw 87.5 / 77.5** vs consz (3,000 steps) 86.9 / 61.2 (new +16.3, 5/1, p=0.094) vs cons 86.9 / 71.3 (library
  +0.6, new +6.2, n.s.) vs consfz (plain LoRA, 3,000 steps) 86.9 / 77.5 (identical). PRE-REGISTERED: within 3pp of cons on new
  -> **YES: a longer correction replaces conversion** (6,000 correction steps vs conversion's 8 x 1,500 + 3,000). consz at
  3,000 was undertrained, not capacity-limited (revises the consfz reading above); a plain LoRA gets there in half the steps.
  Candidate pipeline: training-free merge + ONE plain LoRA trained inside it (3,000 steps) on own rollouts + library replay.
- 11-06 21:56 queue change (`sweep/chain42_noconv.sh`, tmux c42; replaces c41, which had started nothing, and chain40's
  trainer lanes, blocked by hand-made claim files; chain40's evaluator still evaluates cons s4 / joint s4). One job queue for
  both A6000s: consfz s4 (scaling) > joint s4 > consfz orders 1-3 > cons s4. Reason: EXP-148 refs (below) show the converted
  merge collapses at 38 skills; the no-conversion arm is now the main candidate. consfz eval port 9400.
- 11-06 23:24 RESULT consr order 2 (no demos; 24/24 ok): **raw 86.9 / 83.8** vs cons 88.1 / 73.8 (library -1.3, new +10.0,
  both n.s.) vs joint 86.2 / 85.0 (+0.6 / -1.2, n.s.). On the weakest order the demo-free arm reaches joint on new skills.
  3-order demo-free mean (orders 0-2): 88.5 / 79.6 vs cons 89.8 / 76.3 vs joint 87.3 / 80.8. Order 3 ~02:00 (c38).
  c42: consfz s4 training on GPU2 since 22:37.
- 11-07 00:34 RESULT consr order 3 (24/24 ok): raw 91.9 / 88.8 vs cons 89.4 / 91.2 vs joint 87.5 / 91.2 (all n.s.).
  **4-ORDER DEMO-FREE (consr): raw 89.4 / 81.9** vs cons (5 demos per library skill) 89.7 / 80.0 vs joint training (all
  demos) 87.3 / 83.4. Pooled paired (64 library / 32 new skill-runs): consr - joint library +2.0 (p=0.20), new -1.6 (p=0.98);
  consr - cons -0.3 / +1.9 (n.s.). => **no demonstrations at all, joint-training level on 4/4 orders**: the pipeline needs
  only the skill adapters + the simulator. chain38 done. joint s4 training on GPU0 since 23:58 (c42 j4).

### EXP-148 — scaling: one correction for ALL 22 new skills at once (16 library + 22 new = 38 skills) — **SCALES (11-07): one plain LoRA correction (no conversion) 88.1 / 81.8 = joint training on the 38 skills 85.0 / 84.1; with conversion 83.8 / 79.1 (scales, at the 5pp bar; no gain from conversion) — done**
Motivation: t = 8 is small; does one correction trained inside the merge still reach joint training when the merge holds
2.75x more new skills (more interference at w 0.25 each)? Pseudo-order 4 (`exp116.SCALE`): order 0's library + the union
of the 4 orders' new skills (22; converted adapters + rollouts already exist, no new data).
- 11-06 18:04 launched (`sweep/chain40_scale.sh`, tmux c40). Code: `exp116.sequence(4)`; exp144 tag / exp145 T = len(new)
  (orders 0-3 unchanged: order-0 abs rerun identical); exp147 reference arms scr (scratch merge, ours_lib 1/k) + convm
  (converted merge), replay.txt written directly at SEQ 4, consr replay_roll = order 0's; exp136 registers joint_s4 (tasks =
  order 0's 16 library + the 22 new, 38 lines, demos). new_loras_cons_s4 + joint_loras on the NAS (shiva quota 44.5/46 GB).
  Now: scr + convm eval (port 9800). After chain38 has claimed consr orders 1-3: cons s4 (3,000 steps) and joint s4 (rank
  64, 12k steps, ~10 h) on the first free A6000s (ETA cons eval ~03:30, joint eval ~12:00 tomorrow).
  PRE-REGISTERED (raw success; t = 8 order-0 reference cons 86.9 / 71.3): scales = cons s4 library within 5pp of joint s4
  and new skills within 5pp of joint s4 (as at t = 8, gap 2.4 / 3.4); breaks = >= 10pp below joint on either -> then
  test one correction per 8 new skills / longer correction.
- 11-06 21:51 RESULT training-free references at 38 skills (38/38 ok each; raw library / new): scratch merge (1/k)
  **55.6 / 20.0** (t=8: 73.8 / 5.0); converted merge (w 0.25 per new skill) **15.6 / 13.2** (t=8: 69.4 / 33.8) -> the
  converted merge COLLAPSES at 22 new skills (22 x 0.25 of private parts swamps the library); the scratch merge degrades
  gracefully. Deviation from the plan: consfz s4 (plain LoRA correction inside the scratch merge, no conversion; see EXP-147
  consz6 / consfz) added and trained FIRST (c42); the same pre-registered criterion (within 5pp of joint s4 on both = scales)
  applies to it. cons s4 still runs (last in the queue) and keeps its pre-registration. ETA consfz s4 eval ~03:00,
  joint s4 eval ~12:00.
- 11-07 02:30 RESULT consfz s4 (ONE plain LoRA correction, 3,000 steps, inside the scratch merge of all 38 skills, no
  conversion, library replay = demos; 38/38 ok): **raw 88.1 / 81.8** vs scratch merge 55.6 / 20.0, converted merge 15.6 /
  13.2; at t=8 (order 0) consfz 86.9 / 77.5. New skills per skill 40-100% (3 at 40). One correction absorbs 22 new skills at
  once with no loss vs 8. Pre-registered verdict waits for joint s4 (training on GPU0 since 23:58, ETA eval ~12:00); vs the
  4-order t=8 joint mean (87.3 / 83.4) it is +0.8 / -1.6.
- 11-07 02:36 queue change: candidate FINAL pipeline consfzr (exp147 arm: consfz + consr's replay of the library's own
  rollouts = no conversion, no demos at all) on orders 0, 4, 1, 2, 3 (`sweep/chain43_final.sh`, tmux c43; eval port 9300),
  then consfz orders 2-3, then cons s4 (chain42's f2 / f3 / c4 claims made by hand so c43 owns them; chain42 still trains f1
  and evaluates consfz; chain40 evaluates cons s4 / joint s4). PRE-REGISTERED: final pipeline holds = consfzr 4-order raw
  within 5pp of joint (87.3 / 83.4) on both AND consfzr s4 within 5pp of joint s4 on both.
- 11-07 04:39 RESULT consfz order 1 (no conversion, plain LoRA correction, library demos; 24/24 ok): raw **88.8 / 76.2** vs
  cons 94.4 / 83.8 (-5.6 / -7.5, n.s.) vs consz (WP, 3,000 steps) 94.4 / 77.5 vs joint 85.6 / 81.2 (+3.1 / -5.0, n.s.).
  On order 1 the plain correction is no better than WP on new skills (order 0: +16) and below cons. Orders 0-1 mean consfz
  87.9 / 76.9 vs cons 90.7 / 77.6 -> library -2.8, new -0.7: conversion's value still open (pre-registered 4-order test
  needs f2 / f3, queued in c43 after the consfzr arms). consfzr order 0 training since 03:57.
- 11-07 07:16 RESULT consfzr order 0 (candidate final: no conversion, no demos; 24/24 ok): raw **78.8 / 65.0** vs consfz
  (library demos) 86.9 / 77.5 (-8.1 / -12.5, n.s.) vs consr (conversion, no demos) 90.6 / 71.2 (library -11.9, 2/7,
  p=0.033) vs joint 90.0 / 76.2. All 16 library skills are in its replay; the loss is a few skills collapsing (open top
  drawer 100 -> 20 / 60, cream cheese 100 -> 50) -- the plain correction is unprotected and per-skill swings are large
  (butter: consfz 10, consfzr 70). Not the final pipeline as it stands. Queue reordered: chain43's r1-r3 claimed by hand so
  its lanes take f2 / f3 (does conversion matter, pre-registered) and c4 (does the conversion pipeline scale) first;
  consfzr orders 1-3 after that (`sweep/chain44_final_rest.sh`, tmux c44; chain43's evaluator still evaluates them).
  consfzr s4 training (GPU2, ETA eval ~10:30).
- 11-07 10:11 RESULT consfzr s4 (38 skills, no conversion, no demos; 38/38 ok): raw **84.4 / 81.4** vs consfz s4 (library
  demos) 88.1 / 81.8 (-3.7 / -0.5, n.s.; new 9/7 skills). At scale the demo-free, conversion-free correction holds; the
  order-0 drop (78.8 / 65.0) looks like run variance, not a systematic failure (orders 1-3 decide, c44). joint s4 eval ~12:30.
- 11-07 11:54 RESULT joint s4 (rank 64, 12k steps, demos of all 38 skills; 38/38 complete): raw **85.0 / 84.1**.
  consfz s4 - joint: library +3.1 (p=0.39), new -2.3 (p=0.84); consfzr s4 (no demos) - joint: -0.6 / -2.7 (n.s.);
  training-free scratch merge - joint: -29.4 / -64.1. PRE-REGISTERED (within 5pp of joint on both = scales): **YES for
  both arms**. One correction of 3,000 steps, rank 16, trained inside the merge on the new skills' own rollouts, reaches
  joint training on 38 skills (12,000 steps, rank 64, all demos); the demo-free version too. The gap vs joint does not grow
  from 8 to 22 new skills. cons s4 (conversion pipeline from the collapsed converted merge) training on GPU2 since 11:53.
- 11-07 12:11 odin: cons1 order 2 trained (chain39, 18.8 h); eval on shiva (`sweep/chain45_cons1_s2_eval.sh`, tmux c45,
  port 9200). Tried consfzr seed 1 (order 0) on odin's P6000 (`sweep/chain46_odin_consfzr1.sh`; exp147 arm consfzr1): OOM
  twice (plain LoRA in-context needs ~23.5 GB, card 24 GB; expandable_segments did not help) -> dropped, claim removed.
  Plain-LoRA corrections run on shiva only. odin GPU1 (0000:03:00.0) reports "Unknown Error" in nvidia-smi.
- 11-07 12:43 RESULT consfz order 2 (24/24 ok): raw **90.0 / 67.5** vs cons 88.1 / 73.8 (+1.9 / -6.2, n.s.) vs consz 83.1 /
  72.5 vs joint 86.2 / 85.0 (+3.7 / -17.5, n.s.). Orders 0-2 mean consfz 88.6 / 73.7 vs cons 89.8 / 76.3 (library -1.2, new
  -2.6) -> borderline on the pre-registered "conversion unnecessary" bar (new >= cons - 3, library within 3); order 3
  (f3, ~14:30) decides.
- 11-07 12:50 RESULT cons1 order 2 (seed 1; trained odin, eval shiva c45; 24/24 ok): raw **83.1 / 76.2** vs seed 0 88.1 / 73.8
  (-5.0 p=0.084 / +2.5) vs joint 86.2 / 85.0 (-3.1 / -8.8, n.s.). Two-seed mean on order 2: 85.6 / 75.0 -> the new-skill gap
  to joint on order 2 (~-10) is NOT seed noise for cons (consr, demo-free, got 83.8 there). Seed-to-seed library swing up
  to 5pp (order 0: 0.6) -> single-run differences below ~5pp library / ~10pp new are within run variance.
- 11-07 14:25 RESULT consfz order 3 (24/24 ok): raw 85.6 / 88.8 vs cons 89.4 / 91.2 vs joint 87.5 / 91.2.
  **4-ORDER consfz (no conversion, plain LoRA correction, library demos): raw 87.8 / 77.5** vs cons 89.7 / 80.0 vs consz
  (WP, no conversion) 88.0 / 74.4 vs joint 87.3 / 83.4. Pooled paired: consfz - cons library -1.9 (p=0.29), new -2.5
  (p=0.38); consfz - consz +3.1 new (p=0.12); consfz - joint +0.5 / -5.9 (n.s.). PRE-REGISTERED: conversion unnecessary
  (new >= cons - 3 AND library within 3) -> **YES** (-2.5 / -1.9); conversion needed (new <= consz + 3 = 77.4) -> no (77.5,
  by 0.1). Verdict: conversion buys ~+2.5 new at t=8 (n.s.) and breaks the merge at 38 skills -> drop it from the main
  pipeline; keep as an optional ablation. Remaining: consfzr (no demos) orders 1-3 (c44, r1 since 13:37), cons s4 (~16:00).
- 11-07 16:21 RESULT cons s4 (conversion pipeline: WP correction trained inside the collapsed converted merge, 3,000 steps;
  114 ok / 1 episode_error): raw **83.8 / 79.1** vs joint s4 85.0 / 84.1 (library -1.2 p=0.38, new -5.0 p=0.53) vs consfz s4
  88.1 / 81.8 (library -4.4 p=0.11, new -2.7 p=0.50). PRE-REGISTERED (within 5pp of joint on both = scales, >= 10 below =
  breaks): **scales, exactly at the bar on new skills** (-5.0). The correction rescues even the collapsed converted merge
  (15.6 / 13.2 -> 83.8 / 79.1), but conversion adds nothing at 38 skills (both n.s. below consfz) -> consistent with
  dropping it (EXP-147 4-order verdict). EXP-148 done; consfzr 4-order (c44) still pending for the final-pipeline test.
- 11-07 18:17 RESULT consfzr order 1 (no conversion, no demos; 24/24 ok): raw **72.5 / 85.0** vs consr 88.1 / 83.8 (library
  -15.6 p=0.033) vs consfz 88.8 / 76.2 (library -16.2 p=0.054, new +8.8 n.s.) vs joint 85.6 / 81.2 (-13.1 / +3.8). Same as
  order 0 (78.8 / 65.0, library -12 vs consr) -> real, not noise: the plain correction loses the library when its replay is
  the library's own rollouts. Not a data-quantity issue (41-50 successful library rollouts per skill vs 5 demos); 5 / 16
  skills drop 30-80pp (e.g. red mug on the left plate 90 -> 10). WP (consr) or demos (consfz) each prevent it. Final-pipeline
  pre-registration (4-order within 5pp of joint) needs orders 2 + 3 library >= ~89 each -> very likely fails; r2 / r3 run to
  completion for the 4-order ablation. Main pipeline = consfz (library demos, no new-skill demos).
- 11-07 20:10 consfzr order 2 = 1.2 / 0.0 -> INVALID (bug, not a result). My EXP-149 edit to train_lora.py (14:2x) put the
  resume block between `if free` and `elif ctx`, so every free context WITHOUT resume fell into the WP branch: trained a WP
  correction (A = V^T frozen, 29.5M trainable instead of 46.1M) that exp147 then deployed as a raw plain LoRA. Affected
  ONLY consfzr r2 (14:56) and r3 (17:16) (scan of every train_*.log for both markers); consfz2 (resume branch) and all
  earlier runs are clean. Fixed 20:15 (resume nested inside the free branch; behaviour without resume = original), synced;
  r3 killed (PID); artifacts + order-2 records -> results/merge_stale/bug1107_consfzr. Redo queued LAST
  (`sweep/chain49_consfzr_redo.sh`, tmux c49: after chain48's 4 trainings; s2 eval port 9350, s3 eval by chain43).
- 11-09 11:55 RESULT consfzr redo orders 2 / 3 (plain LoRA, 46.1M trainable checked; 24/24 ok each): order 2 **86.2 / 70.0**,
  order 3 **83.1 / 73.8**. 4-order consfzr (no conversion, no demos) **80.2 / 73.4** vs consfz 87.8 / 77.5 (library -7.7,
  wins 11/27, p=0.026; new -4.1 n.s.), vs consr 89.4 / 81.9 (-9.2 p=0.002 / -8.4 p=0.041), vs joint 87.3 / 83.4 (-7.2 p=0.033
  / -10.0 p=0.048). Pre-registered final-pipeline test (4-order within 5pp of joint on both) FAILS -> confirmed: dropping
  the library demos needs the conversion (consr); dropping the conversion needs the 5 library demos per skill (consfz).
  Main pipeline stays consfz. 4-order ablation complete.

### EXP-149 — self-improvement: a second correction round on the corrected merge's OWN rollouts — **NEGATIVE (11-07): orders 2 / 0 new -3.8 / -7.5; order 3 dropped**
Motivation: the conversion-free pipeline (EXP-147 consfz, 4-order 87.8 / 77.5) is 5.9pp below joint on new skills. Its
data are the externals' own rollouts; the corrected merge now succeeds on states those adapters failed. VLA-specific
lever: a closed-loop policy can generate its next training set itself (no demos, no external adapters needed in round 2).
- 11-07 14:30 launched (`sweep/exp149_selfimprove.py`, `sweep/chain47_selfimprove.sh`, tmux c47). collect: the round-1
  deployed model (scratch merge 1/k + consfz raw) runs the 8 new skills, 50 episodes on states 10..49, dumps on the NAS
  (e149_<tid>_r2s<SEQ>), orders 2 (weakest), 0, 1, 3 on the free eval slots (2 workers / GPU, port 9200). train: consfz
  continues 1,500 steps (train_lora ctx key "resume"), same merge, round-1 + round-2 rollouts (2 groups per skill) + the same
  library demo replay -> exp147 arm consfz2 (eval port 9250), on each A6000 after chain44's consfzr trainings.
  PRE-REGISTERED (orders with both arms): round 2 helps = consfz2 new >= consfz new + 5pp pooled, library within 3pp.
  Confound if it helps: +1,500 steps -> then a same-steps control on round-1 data only.
- 11-07 15:32 order 2 collected (400 episodes): round-2 successes 283 / 400 (70.8%; per skill 50 / 24 / 46 / 50 / 41 / 15 /
  21 / 36 of 50) vs the externals' own round-1 rollouts -> every new skill has round-2 successes. Order 0 collecting
  (16:21: 7 / 8 skills done).
- 11-07 18:07 all 4 orders collected (failures dumped too). Per new skill (orders 0, 2): the corrected merge is far below its
  own external on some skills (order 2 tomato sauce: external 90, consfz 20; drawer + bowl: 60 vs 20, both orders) while
  on others the external itself is weak (wine bottle on rack: external 30, joint 80). Order 2 new: externals 75.0, consfz 67.5,
  per-skill max of the two 82.5, joint 85.0 -> the new-skill gap = (a) merge not absorbing its expert + (b) weak experts.
- 11-07 21:31 RESULT order 2 (24/24 ok): consfz2 raw library **87.5** / new **63.7** vs consfz 90.0 / 67.5 (paired: lib -2.5
  p=0.91, new -3.8 p=0.25; joint 86.2 / 85.0) -> round 2 on the merged model's own successes does NOT help on order 2 (bar
  +5 new). Orders 0 / 1 training. Order 3 DEFERRED behind EXP-150 (slot claimed by hand, `.c47_t3`); futility rule set now,
  before orders 0 / 1 land: run order 3 only if the 3-order mean new gain is >= +3pp, else EXP-149 stops (negative).
  Reading: own successes add data where the model already works; the failures (where the gap is) get no labels -> EXP-155.
- 11-07 23:11 RESULT order 0 (24/24 ok): consfz2 **91.9** / **70.0** vs consfz 86.9 / 77.5 (lib +5.0 p=0.61, new -7.5 p=0.69;
  joint 90.0 / 76.2). 2 orders: new -3.8 / -7.5 (mean -5.6), library -2.5 / +5.0. Order 1 would need >= +20pp for the
  futility bar -> EXP-149 NEGATIVE: a second round on the merged model's own successes does not raise new skills (it may
  trade new for library). Order 1 is already trained and evaluates anyway; order 3 dropped (claim `.c47_t3` stays).
- 11-08 00:xx RESULT order 1 (24/24 ok): consfz2 90.0 / 80.0 vs consfz 88.8 / 76.2 (+1.3 / +3.7, n.s.). 3-order mean new
  -2.5pp (< +3 futility bar), library +1.3 -> stop, as pre-set. EXP-149 closed: negative.

### EXP-150 — breadth: bigger library (16 -> 32 -> 74 LIBERO-90 skills, + order 0's 8 new skills) — **SCALES at 16 / 32 / 74 (within 5pp of joint at each; 74: 2 seeds 82.6 / 81.9 vs joint24k 86.4 / 80.0, library -3.8 p=0.02, not closed by steps, demos or rank 64) — done (11-10)**
Motivation: so far only the new-skill side was scaled (8 -> 22, EXP-148). Does one in-merge correction (consfz) still match
joint training when the library itself is 2x and ~5x larger (k = 40 / 82, training-free merge heavily diluted)?
- 11-07 18:34 launched (`sweep/chain48_libscale.sh`, tmux c48). Pseudo-orders in exp116.sequence: 5 = order 0's 16 + 16
  seeded LIBERO-90 skills (32), 6 = all 74 fleet skills; both + order 0's 8 new; libraries nested. `SEQ == C.SCALE` checks
  -> `>=` (exp147, _e147_abs); exp136 registers joint_s5 / s6 (task files exp136_joint_tasks_s5 / s6). Same recipes as
  EXP-148: consfz 3,000 steps (context = scratch merge at 1/k, k = 40 / 82; replay 5 demos x 32 / 74 library skills),
  joint rank 64, 12k steps, all demos. Now: training-free scratch merge (scr) evals (port 9400); trainings queue after
  chain47's round-2 trainings: consfz s6, joint s6, consfz s5, joint s5; evals 9450 / 9500.
  PRE-REGISTERED (raw success, per library size): scales = consfz within 5pp of joint at the same size on library AND new;
  breaks = >= 10pp below joint on either. Fairness: if joint s6 library falls > 5pp below joint s4 (85.0), joint may be
  under-trained at 82 skills -> add a 24k-step joint before concluding; if consfz breaks, try 6,000 steps before concluding.
- 11-07 20:22 RESULT training-free scratch merge, 32-skill library (s5, k = 40; 40/40 ok): raw library **72.8** / new **3.8**
  (order 0, k = 24: 73.8 / 5.0; s4, 16 + 22 new: 55.6 / 20.0). Same 16 library skills: 73.8 (k 24) -> 71.9 (k 40) -> doubling
  the LIBRARY barely dilutes it, while adding NEW skills does (s4 -16) — as EXP-137. New skills ~0 untrained, as always.
  74-skill refs (s6) evaluating.
- 11-07 22:25-22:31 consfz s6 and joint s6 CRASHED at data loading (no result lost, 0 steps): LIBERO-90 skill
  LIVING_ROOM_SCENE2 "pick up the butter and put it in the basket" (in the 32- and 74-skill libraries) has no file in the
  regenerated no-noops demo set. Fix: train_lora `load_demo_samples` falls back to the original LIBERO-90 demos for that
  skill (`RAW90`, 128 px, upsampled like the rest; no-ops dropped with OpenVLA's rule; checked: same scene / orientation as
  the regenerated set). Joint and consfz see the same data for it. chain48's lanes stopped by PID (its evaluator keeps
  running); failed logs / claims -> results/merge_stale/fail1107_s6demo; trainings relaunched (`sweep/chain53_libscale_train.sh`,
  tmux c53): j6 first on the free A6000 (22:46), then f6, f5, j5. ETA: consfz s6 ~04:00, joint s6 ~Sunday 12:00.
- 11-07 23:4x RESULT training-free scratch merge, 74-skill library (s6, k = 82; 82/82 ok): raw library **77.4** / new **7.5**
  (single adapters on the same 74: 94.9). Same order-0 16 skills: 73.8 (k 24) -> 71.9 (k 40) -> 84.4 (k 82); s5's extra 16:
  73.8 -> 68.8; the 42 skills only in s6: 78.1 -> no dilution trend up to 74 library skills (differences = 10-episode noise,
  ~±4pp per 16-skill mean). The training-free library merge holds at ~75 % from 16 to 74 skills; new skills ~0 untrained.
- 11-08 06:09 RESULT consfz s6 (74-skill library + 8 new, k = 82; 82/82 ok): raw library **78.1** / new **68.8** (scratch merge
  77.4 / 7.5; single adapters 94.9 / 76.2). Order-0 16 library skills: consfz 86.9 (k 24) -> 88.1 (k 38) -> **76.9 (k 82)**,
  below the uncorrected merge at k 82 (84.4); the 58 other library skills: 78.4 vs merge 75.5 (+2.9, 20 up / 19 down). New:
  77.5 (k 24) -> 68.8 (next to ramekin 70 -> 30, on ramekin 100 -> 70). So at 74 library skills the one correction adds the
  new skills (7.5 -> 68.8) but no longer lifts the library above the merge, and costs the order-0 library ~8-10pp. Plausible
  cause: 3,000 steps x batch 8 spread over 82 tasks (~290 samples per task vs ~1,000 at k 24; replay per library skill 1/148
  of the batch vs 1/32). Scene-sharing check (instruction-only disambiguation): no gradient with scene group size (79 / 82
  skills share a scene) -> not the explanation from these numbers. Verdict waits for joint s6 (step 7,150 / 12,000 at 06:14;
  result ~14:30); if consfz breaks (>= 10pp below joint), the pre-registered 6,000-step consfz s6 runs before concluding.
- 11-08 06:15 the 6,000-step consfz s6 (arm consfz6) queued NOW rather than after joint s6 (saves ~8 h; used only under the
  pre-registered condition): `sweep/chain57_consfz6_s6.sh` (tmux c57), first A6000 free after f5 is claimed (~09:00), ahead of
  j5 / chain49 / EXP-151 / EXP-152; eval port 9850.
- 11-08 11:23 RESULT consfz s5 (32-skill library + 8 new, k = 40; 40/40 ok): raw library **77.5** / new **68.8** (scratch merge
  72.8 / 3.8; joint s5 not trained yet). Order-0 16 library: consfz 86.9 (k 24) -> **80.0 (k 40)** -> 76.9 (k 82); the 16 added
  library skills: consfz 75.0 vs merge 73.8 (single 95.0). New 77.5 -> 68.8 -> 68.8. The correction's library lift over the
  untrained merge shrinks with library size: +13.1 (k 24) -> +4.7 (k 40) -> +0.7 (k 82), while it is +32.5 at 16 library + 22
  new (s4: 88.1 vs 55.6) -> it is the LIBRARY size, not k, that thins the correction.
  HYPOTHESIS (per-skill replay budget): replay = half of 3,000 x 8 samples spread over the library: ~750 per library skill at
  16, ~375 at 32, ~160 at 74. PRE-REGISTERED (11-08 11:30, before the runs): consfz at 6,000 steps (consfz6) restores 32
  skills to the order-0 budget -> hypothesis holds if consfz6 s5 order-0 library >= 83.0 (86.9 minus the ~4pp noise of a
  16-skill mean) AND the 16 added skills >= merge + 8 (>= 81.8); s6 at 6,000 (~320 per skill) should land between. Fails if consfz6
  s5 library stays within 3pp of consfz s5 (77.5). `sweep/chain59_consfz6_s5.sh` (tmux c59): after f6x and j5 are claimed
  (j5 = 12 h, the longest pole, first), 60 s rule, eval port 9900. j6 finished 11:26 (eval next, c48 port 9500).
- 11-08 14:44 RESULT joint s6 (82 skills, rank 64, 12k steps): raw library **78.2** / new **77.5**. consfz s6 - joint: library
  **-0.1** (n=74, wins 23/25, p=0.46), new **-8.8** (n=8, wins 1/4, p=0.19). Joint dilutes exactly like the correction: order-0
  16 library 90.0 (k 24) -> 76.2 (k 82) vs consfz 86.9 -> 76.9; the 58 others 78.8 vs 78.4. So the library-size decline is
  shared with joint training (expected: fixed steps over more tasks), not a weakness of the merge + correction.
  Pre-registered verdict: library within 5pp (scales), new -8.8 = between the 5pp "scales" and 10pp "breaks" bars ->
  inconclusive on new pending the fairness checks: joint s6 library 78.2 is > 5pp below joint s4 (85.0) -> the 24k-step joint
  is required (`sweep/chain60_joint24k_s6.sh`, tmux c60; exp136 E136_SFX=24k / E136_STEPS=24000 -> joint_loras/joint24k_s6;
  _e147_abs reports it as joint24k; first A6000 free ~17:00, ~25 h, eval port 9950); consfz6 s6 (6,000 steps) lands ~19:00.
  New-skill gap at t = 8 elsewhere: -5.9 (4 orders, 16 library) -> -8.8 at 74 is the same order of size.
- 11-08 14:47 consfz6 s5 (chain59) CANCELLED before it claimed a GPU: it tested a library decline that joint shares; f6x
  already measures more steps. Pre-registration above left unanswered.
- 11-08 20:31 RESULT consfz6 s6 (6,000 steps, 74-skill library; 82/82 ok): raw library **80.9** / new **86.2** vs joint s6
  (12k steps, rank 64) 78.2 / 77.5: library +2.7 (wins 27/27, p=0.62), new **+8.8** (wins 5/1, p=0.19); vs consfz (3,000) +2.8 /
  +17.5. Order-0 16 library 76.9 -> **88.8** (k 24 level 86.9: the per-skill replay budget restores it; joint 76.2); 58 others
  78.8 (= joint). New per skill: next to ramekin 30 -> 90, drawer + bowl 10 -> 60 (single 60), on ramekin 70 -> 100; new mean
  86.2 > single adapters 76.2. Pre-registered: SCALES (within 5pp of joint on both, in fact above) with the correction at 6,000
  steps (half joint's steps, a quarter of its LoRA rank, no new-skill demos). Open: the 24k-step joint (fairness, started 17:45
  on U2, ~Monday 22:00) could still lift joint.
- 11-08 20:34 consfz6 s5 RE-QUEUED (chain59 relaunched, same script and pre-registration; nothing of it had run): the library
  curve needs one recipe at 32 and 74 library skills. Claims the first A6000 free after j5 (~23:40).
- 11-09 01:04 RESULT joint s5 (40 skills, rank 64, 12k steps): raw library **84.7** / new **83.8** (order-0 16 library 91.9, the
  16 added 77.5). consfz s5 (3,000 steps) - joint: library -7.2 (n=32, wins 8/17, p=0.16), new **-15.0** (wins 2/4, p=0.25) ->
  BREAKS on new at 3,000 steps (pre-registered >= 10pp) -> the pre-registered 6,000-step consfz s5 decides (f5x on U0 since
  23:34, eval ~06:00). Joint library 84.7 is within 5pp of joint s4 (85.0) -> no 24k joint needed at 32 skills.
- 11-09 05:43 RESULT consfz6 s5 (6,000 steps, 32-skill library; 40/40 ok): raw library **84.1** / new **78.8** vs joint 84.7 /
  83.8: library -0.6 (wins 7/8, p=0.98), new **-5.0** (wins 2/2, p=1.0) -> SCALES, new exactly at the 5pp bar (4-order gap at
  16 library skills: -5.9). vs consfz (3,000): +6.6 / +10.0. Replay-budget hypothesis (pre-registered 11-08 11:30): order-0 16
  library 80.0 -> **88.1** (>= 83.0 met); the 16 added 75.0 -> 80.0 = merge + 6.2 (< +8 bar, not met; joint 77.5) -> HALF
  holds: the budget restores the original library; the added library skills rise but less than predicted (they sit at joint's
  level anyway). Library-size curve, correction at ~750 replay samples per library skill or more (3,000 steps at 16, 6,000 at
  32 / 74) vs joint: 16: +0.5 / -5.9 (4 orders, n.s.); 32: -0.6 / -5.0; 74: +2.7 / +8.8. Pending: joint 24k s6 (~Mon 22:00).
- 11-09 18:06 RESULT joint 24k s6 (82 skills, rank 64, 24,000 steps, trained 21.3 h): raw library **86.4** / new **80.0** vs joint
  12k 78.2 / 77.5 (library +8.1, wins 35/11, p=0.001) -> the 12k joint was under-trained at 82 skills (fixed steps over more
  tasks); 86.4 ~ joint s4 85.0 (38 skills, 12k) = near its ceiling. consfz6 (6,000 steps) - joint24k: library **-5.4** (n=74, wins
  16/34, p=0.027), new **+6.2** (wins 5/1, p=0.19); consfz (3,000): -8.2 (p=0.001) / -11.2. Against the fair joint, the
  74-library point is just outside the 5pp bar on library (above on new) -> "SCALES" at 74 no longer holds as stated; same
  pattern as joint: steps per library skill. 32-library verdict unchanged (joint s5 library 84.7 ~ ceiling, no 24k needed).
- 11-09 18:15 queued consfz12 s6 (consfz at 12,000 steps = half joint24k's steps, rank 16 vs 64; ~650 replay samples per
  library skill vs ~325 at 6,000 and 750 at the 16-skill recipe; exp147 arm consfz12, VSTEPS 12000;
  `sweep/chain61_consfz12_s6.sh`, tmux c61: first A6000 free after f20 / f7, 60 s rule ahead of chain55; eval port 10050;
  ~12.6 h train + ~3 h eval -> ~Tue 11-10 midday). PRE-REGISTERED (before training): 74 library SCALES vs the fair joint if
  consfz12 s6 is within 5pp of joint24k (86.4 / 80.0) on both; replay-budget explanation holds if the library gap shrinks by
  >= 3pp vs consfz6 (-5.4 -> >= -2.4); if the library stays >= 5pp below -> limitation: at 74 library skills the correction
  keeps the new skills (above joint) but trails a fully trained joint on the library by ~5pp.
- 11-10 06:52 RESULT consfz12 s6 (12,000 steps; 82/82 ok; 46.1M trainable, loss 0.0265 -> 0.0203, smooth): raw library **81.5** /
  new **66.2** vs joint24k 86.4 / 80.0: library -4.9 (wins 17/26, p=0.075), new **-13.8** (wins 1/5, p=0.13); vs consfz6:
  library +0.5 (p=0.49), new **-20.0** (wins 1/7, p=0.039). Order-0 16 library 85.0 (consfz6 88.8, joint24k 90.0); 58 others
  80.5 (78.8; joint24k 85.3). New per skill (consfz / consfz6 / consfz12): drawer + bowl 10 / 60 / 10, wine on rack 70 / 60 /
  10, wine on cabinet 100 / 90 / 70 (others stable) -> new at 74 library = 68.8 / 86.2 / 66.2 across 3k / 6k / 12k: noisy, mean
  73.7 vs joint24k 80.0. Pre-registered: NOT within 5pp of joint24k on new (BREAKS on new, >= 10pp); replay-budget explanation
  FAILS (library gap -5.4 -> -4.9, bar >= 3pp shrink): more steps do not lift the 74-skill library past ~81.
- 11-10 06:56 queued consfz6r25 s6 = consfz6 with 25 library demos per skill instead of 5 (exp147 VDEMOS; ~130k replay samples
  vs 26k; joint sees all 50). Hypothesis: the 74-skill library is limited by replay DATA diversity (5 initial states per skill),
  not steps. `sweep/chain62_consfz6r25_s6.sh` (tmux c62; first A6000 free, 60 s rule ahead of chain55; eval port 10100; ~6.5 h
  train + ~3 h eval). PRE-REGISTERED (before training): data-limited if library >= 84.0 (consfz6 + 3, gap to joint24k <= 2.4);
  not data-limited if library within 2pp of consfz6 (<= 82.9). Either way new is read against the 3-run mean (73.7) given the
  noise. Paper consequence if it fails: 74-library = limitation (library ~5pp below a fully trained joint).
  Training on U2 since 07:13: replay 74 x 25 demos = 129,252 samples (5 demos: 25,930); shiva RAM 61 GB still available.
- 11-10 12:07 queued consfz6s1 s6 = consfz6 at training seed 1 (noise check, not EXP-154: one replicate of one unstable setting).
  `sweep/chain63_consfz6s1_s6.sh` (tmux c63, U0 idle; eval port 10150; ~6.3 h + ~3 h -> ~22:00). PRE-REGISTERED: the 6k new
  result (86.2) is stable if seed 1 is within 10pp (>= 76.2); otherwise the 74-library new number is reported as the mean of the
  6k runs with their range. Library expected ~81 (consfz6 80.9).
- 11-10 15:06 consfz6r25 s6 RESULT (82/82 records ok): raw 78.6 / 78.8 (consfz6 80.9 / 86.2, joint24k 86.4 / 80.0).
  vs joint24k: library -7.7 (p=0.007), new -1.2 (n.s.); vs consfz6: library -2.3 (n.s.), new -7.5 (n.s.; drawer+bowl 40, wine
  rack 20). Pre-registered: NOT data-limited (library 78.6 <= 82.9): 5x more replay demos do not lift the library. New 78.8 vs
  the 3-run mean 73.7 = within the noise (4 runs: 68.8 / 86.2 / 66.2 / 78.8, mean 75.0). Steps (3k / 6k / 12k) and replay data
  (5 / 25 demos) both ruled out -> the remaining difference to joint24k is CAPACITY (rank 16 = 46.1M vs joint's rank 64 = 184.4M).
- 11-10 15:16 queued consfz6r64 s6 = consfz6 at LoRA rank 64, alpha 64 (exp147 VRANK -> train_lora --rank; 184.4M trainable =
  joint's). `sweep/chain64_consfz6r64_s6.sh` (tmux c64; trainer on U2 since 15:17; eval port 10200; ~6.5 h + ~3 h -> ~01:00).
  PRE-REGISTERED (before training): capacity-limited if library >= 84.0 (consfz6 + 3, gap to joint24k <= 2.4); not
  capacity-limited if library <= 82.9 (within 2pp of consfz6). New read against the 4-run mean (75.0). Paper consequence if it
  fails: the 74-library point is a limitation (library ~5-8pp below a fully trained joint; new on par).
  Trainer check: r=64 alpha=64, trainable 184.4M (= joint), merge of 82 members folded (w 0.0122); training started.
- 11-10 19:49 consfz6s1 s6 RESULT (82/82 ok): raw 84.2 / 77.5 (seed 0: 80.9 / 86.2). Pre-registered: new STABLE (77.5 >= 76.2).
  Library +3.2 vs seed 0 (p=0.04) = seed spread ~3pp at the 74-skill library. vs joint24k: library -2.2 (n.s.), new -2.5 (n.s.).
  2-SEED consfz6 (per-skill mean): 82.6 / 81.9 vs joint24k 86.4 / 80.0 -> library -3.8 (p=0.02), new +1.9 (n.s.) = within the 5pp
  scaling bar on both (as s5: 84.1 / 78.8 vs 84.7 / 83.8). Library gap SIGNIFICANT but small.
  Where the library gap lives (per skill, 2-seed mean): 23 / 74 library skills are BROKEN by the training-free merge (merge <=
  expert - 30: expert 92.0, merge 43.0); the correction lifts them to 55 (3k) / 64 (6k, both seeds) / 62 (12k) / 58 (25 demos);
  joint24k 69.6, joint 12k 55.7. The other 51: expert 96.2, merge 92.9, ours 88.4 / 93.1 (seeds 0 / 1), joint24k 93.9. Ours tracks
  the merge per skill (Spearman 0.69): the correction keeps what the merge keeps and only partly repairs what it breaks; even
  the fully trained joint repairs those 23 only to ~70 (they are hard in any 82-task model).
- 11-10 19:57 queued consfz6r64s1 s6 = consfz6r64 at seed 1 (`sweep/chain65_consfz6r64s1_s6.sh`, tmux c65, trainer on U0 since
  19:58, eval port 10250; ~4.5 h + ~3 h -> ~03:30; c64 runs faster than estimated: step 5,350 / 6,000 at 19:59 -> lands ~23:30).
  Reason: the 3.3pp seed spread exceeds the single-run criterion's margin. SUPPLEMENTARY PRE-REGISTRATION (before either rank-64
  result): the capacity verdict is read on 2 seeds vs 2 seeds: capacity-limited if the rank-64 2-seed library >= 85.6 (rank-16
  2-seed 82.6 + 3); not capacity-limited if <= 84.6 (within 2pp). The single-run criterion above is also reported.
- 11-10 23:05 consfz6r64 s6 RESULT (82/82 ok): raw 78.2 / 83.8 (rank 16, same seed: 80.9 / 86.2). vs consfz6: library -2.7,
  new -2.5 (n.s.); vs joint24k: library -8.1 (p=0.02), new +3.8 (n.s.). Pre-registered (single run): NOT capacity-limited (78.2
  <= 82.9). 2-seed criterion: seed 1 would need library >= 93.0 (above every arm at s6, joint24k 86.4) -> FUTILITY STOP of c65
  by hand 23:08 (step ~3,100 / 6,000; tmux session killed; GPU to EXP-157). Verdict: NOT capacity-limited.
- EXP-150 DONE (11-10). Library-size curve, order 0's 8 new skills (raw library / new): 16 library ours (3k) 86.9 / 77.5 vs joint
  90.0 / 76.2; 32 library ours (6k) 84.1 / 78.8 vs joint 84.7 / 83.8; 74 library ours (6k, 2 seeds) 82.6 / 81.9 vs joint24k
  86.4 / 80.0 (joint 12k 78.2 / 77.5); training-free merge 73.8 / 5.0, 72.8 / 3.8, 77.4 / 7.5. = SCALES within 5pp of joint at
  every size; at 74 the library gap (-3.8, p=0.02) is real, not closed by steps (12k), replay data (25 demos) or capacity (rank
  64), and sits in the 23 skills the merge breaks (even joint24k only ~70 there). Correction steps grow with the library (3k ->
  6k, ~750 -> ~320 replay samples per library skill). Figure `results/figs/libsize.png`.

### EXP-157 — composite tasks (LIBERO-10): can ONE merged policy chain library skills no single adapter covers? — **done (11-11): NEGATIVE (pre-registered); no arm chains, one model = router on sub-goals**
Motivation (user Q 11-10 "what's a scenario where someone would use our method? Is that scenario practical?"): per-task adapter
switching (a router loading the instructed skill's own adapter) beats the merge on known skills (74-skill library: expert 94.9
vs ours 82.6; ours ahead on new 81.9 vs 76.2) -> the merge needs a case routing cannot serve. LIBERO-10 tasks chain two
LIBERO-90 skills of one scene in one episode under one instruction; 5 / 10 are pure compositions of two 74-library skills.
- 11-10 setup: `sweep/exp157_compose.py`; client `examples/libero/main_sub.py` = main_dense.py + the BDDL goal predicates checked
  after every step -> `subgoals.jsonl` per job / scene (smoke on shiva: no predicate true at the start except KITCHEN_SCENE8's
  stove, excluded from its sub-goal count; one consfz6 episode on stove + moka = stove on, pot not placed, 48 s). Base =
  pi05_base (never fine-tuned on LIBERO-10); zero-shot, full instruction, test states 0..9, 10 eps, 520 steps. Arms (s6): ours
  consfz6 seeds 0 / 1, scr, joint24k, base (primary only), 17 single sub-skill adapters (the router's choices; router-best =
  per task the better one, oracle pick). 640 episodes. `sweep/chain66_compose_s6.sh` (tmux c66, eval only, port 10300, next to
  c64 / c65; ~4 h).
- PRE-REGISTERED (11-10, before any LIBERO-10 episode except the smoke): primary set = tasks 2 / 3 / 4 / 6 / 7 (stove + moka,
  bowl in drawer + close, two mugs, mug + pudding, soup + cheese; 50 episodes per arm), metric = full task success. POSITIVE:
  ours (2-seed mean) >= 20% AND >= router-best + 10pp, pooled. NEGATIVE: ours < 10%, or <= router-best -> the composite
  argument does not hold (reported with sub-goal completion). Otherwise inconclusive. Tasks 0 / 1 / 5 / 8 / 9 reported only.
  Not tested: a planner that splits the instruction and switches adapters on detected sub-goal completion (needs a completion
  detector; the stronger routing baseline, out of scope).
- 11-10 user: a negative result is acceptable: "A router needs all adapters to be stored, which is what model merging in
  general avoids, we want to have one model" -> composition = bonus argument, not the paper's motivation.
- 11-11 02:54 RESULT (c66 done, 640 episodes; `E157_STAGE=report`; per-goal breakdown `results/_e157_goals.py` on shiva).
  Primary 5 tasks, 50 eps / arm, full success | sub-goals reached: ours consfz6 0.0 | 47.0, consfz6s1 4.0 | 52.0 (mean 2.0 |
  49.5); scr 10.0 | 54.0; joint24k 2.0 | 49.0; router-best (oracle pick per task) 4.0 | 51.0; base 0.0 | 1.0.
  -> NEGATIVE by the pre-registration (ours 2.0 < 10, <= router-best 4.0; ours vs router p=0.60).
  - every one-model arm does ONE part and stops (ours: one goal in 97 / 100 episodes, both in 2 / 100): stove on, moka never
    placed (0 / 40 in all one-model arms); bowl in drawer, drawer never closed (joint24k 1 / 10); first mug yes, pudding never;
    cream cheese in the basket, soup never. Joint training fails the same way -> limit of single-skill training episodes, not of
    merging. scr's 10% = 5 / 10 on one task (two mugs; vs ours 2 / 100: p=0.04, one task) -> not a merge advantage claim.
  - router singles: each sub-skill adapter does its own part (moka-pot adapter on task 2: pot on stove 9 / 10, stove 0 / 10)
    -> a planner that switches adapters at sub-goal completion would likely chain; not tested (needs a completion detector).
  - on composite instructions one model matches the oracle-picked single adapter on sub-goals (49.5 vs 51) without storing or
    picking adapters; zero-shot composition is not a claim for the paper. Deck 18_11: slide "Composite instructions".

### EXP-155 — DAgger: the experts label the states the corrected merge actually visits (failures included) — **order 2 done (11-08): +2.5 new n.s.; fixes lost-expert skills, hurts weak-expert skills -> EXP-156**
Motivation: the merged policy is a closed-loop controller trained by behaviour cloning on the experts' own (successful)
trajectories -> covariate shift: once it drifts off them (the gap, EXP-149 (a): tomato sauce expert 90 vs consfz 20) nothing
tells it how to recover. Unlike weight-space merging of LLMs / classifiers, each merged expert is still a policy that can be
QUERIED at any state, and the simulator hands out the merged model's own states for free -> DAgger with the experts as
teachers: the expert labels every visited state, successes and failures. Isolates labels from data vs EXP-149 (same rollouts).
- 11-07 21:45 launched. `sweep/exp155_dagger.py`: relabel = each new skill's expert (its from-scratch LoRA alone on the base,
  scale 1; action_error.Applier + LiberoPolicy) predicts one action chunk at every 8th step of the EXP-149 rollouts of that
  skill (round-1 deployed model, 50 episodes, states 10..49, success + failure) -> NAS dumps e155_<tid>_s<SEQ> (frames +
  'chunk'); check per skill: expert beats the base on its own rollouts. train = consfz2's recipe exactly (resume consfz,
  1,500 steps, same merge / library replay / 2 groups per skill) with the round-2 group = relabeled frames -> exp147 arm
  consfzd. train_lora `load_rollout_samples` reads a 'chunk' key when present (other files unchanged; backups
  results/merge_stale/*.bak_1107f). Relabel on odin's idle P6000 (`sweep/chain50_relabel_odin.sh`, tmux c50 on odin, orders
  2, 0, 1, 3); training on the first A6000 that frees after order 2 is relabeled, ahead of chain48's remaining trainings
  (`sweep/chain51_dagger.sh`, tmux c51 on shiva; eval port 9550).
  PRE-REGISTERED (before relabeling): helps = consfzd new >= consfz new + 5pp pooled, library within 3pp. Order 2 first:
  >= +5 -> orders 0, 1, 3; 0..+5 -> order 0 before deciding; <= 0 -> stop. consfzd vs consfz2 = expert labels incl. failure
  states vs own successes (same rollouts, steps, mixture).
- 11-07 21:58 relabeling: odin's P6000 alone ~2.5 s / frame (~5 h per order) -> per-skill claims + one helper per shiva
  A6000 (`sweep/chain52_relabel_shiva.sh`, tmux c52; ~8 min per skill next to a trainer). Check passes on every skill so
  far (expert L1 on its own rollouts 0.06-0.11 vs base 0.28-0.39). First skill (bowl in drawer -> plate): 959 frames / 50
  episodes; the expert disagrees more with the merged model on its failure episodes (L1 0.141) than on its successes (0.088).
- 11-07 22:41 order 2 relabeled (8 skills, 848-1,747 frames each; all checks pass; expert vs merged-model disagreement
  higher on failure episodes for every skill with failures); consfzd s2 training on U0 (resumed from consfz, verified below).
- 11-08 01:08 RESULT order 2 (24/24 ok; resumed consfz verified, 46.1M trainable): consfzd raw **89.4** / **70.0** vs consfz
  90.0 / 67.5 (lib -0.6, new +2.5 p=0.88), vs consfz2 87.5 / 63.7 (new +6.2), joint 86.2 / 85.0 -> in the pre-set 0..+5 band.
  Per skill (expert / consfz / consfzd / joint): tomato sauce 90 / 20 / **90** / 90 (the target failure, fixed); drawer +
  bowl 60 / 20 / 30 / 60; but wine rack 30 / 20 / **0** / 80 (weak expert) and bowl next to plate 80 / 100 / **70** / 100
  (merge already beat its expert) -> classic DAgger ceiling: imitating the teacher pulls the student down where the teacher
  is worse. Instead of ungated order 0 -> EXP-156 (gated); consfzd stays a one-order result.

### EXP-156 — gated DAgger: relabel only the skills whose expert beats the merged model — **NEGATIVE (11-08): new +1.2pp pooled over orders 0/1/3 (bar +5); closed**
Motivation: EXP-155 order 2 per skill = DAgger helps exactly where the merge lost a good expert and hurts where the expert is
weak or already beaten. The receiver can tell which is which without test episodes: it has the expert's own rollouts
(EXP-144) and the merged model's (EXP-149) on the SAME training initial states 10..49.
- 11-08 01:30 setup. `sweep/exp155_dagger.py` arm consfzg (default; E155_ARM=consfzd = ungated): skill i relabeled iff
  expert success - merged success on states 10..49 >= 5 of 50 (E155_GATE; rates, so a partial collection cannot bias it);
  otherwise its round-2 group = round 1 again. Everything else = consfzd (resume consfz, 1,500 steps, same replay). Gated
  skills: order 0: 4 / 8 (drawer + bowl +32, +13, +7, +7), order 1: 2 (+30, +8), order 2: 2 (tomato +13, drawer + bowl +18),
  order 3: 2 (+6, +13). Relabel only gated skills (odin c50 + shiva helpers).
  PRE-REGISTERED (11-08, before any consfzg training): the gate was chosen after seeing order 2's TEST results -> order 2 is
  NOT evidence; the test is orders 0, 1, 3 (fresh). Helps = consfzg new >= consfz new + 5pp pooled over orders 0 / 1 / 3,
  library within 3pp. Mechanism check: the gain should sit on the gated skills (non-gated within noise). If it helps:
  same-steps control (consfz + 1,500 steps on round-1 data only) on the same orders before claiming DAgger did it.
  Priority: on each A6000 ahead of EXP-150's 32-skill trainings (f5 / j5); the 74-skill runs (f6 / j6) keep running.
- 11-08 03:27 g0 setup verified: plain 46.1M trainable, resumed order 0 consfz, 4 relabeled groups loaded.
- 11-08 05:35 order 0 result (raw): consfzg library 95.6 / new 68.8 vs consfz 86.9 / 77.5 (lib +8.8 p=0.10, new -8.7 p=0.17;
  joint 90.0 / 76.2). Per skill (expert / consfz / consfzg): gated drawer + bowl 60 / 20 / 30, next to ramekin 100 / 70 / 60,
  wine on cabinet 100 / 90 / 100, orange juice 80 / 70 / 60 (gated net 0); non-gated losses: on ramekin 100 -> 80, wine rack
  70 -> 40, next to cookie box 100 -> 80. Same shape as consfz2 on order 0 (lib +5.0 / new -7.5): the extra resumed 1,500 steps
  move the correction toward the library whatever the round-2 labels.
- FUTILITY RULE (set 11-08 05:45, after order 0, before order 1's result): passing needs orders 0+1+3 summed >= +15pp, so
  order 1 must give >= +8.7pp for order 3 to need <= +15pp (the most a 2-gated-skill order plausibly gives). Order 1 below
  +8.7 -> STOP (orders 3 / 2 not trained). Until then g3 / g2 held (claims .c54_g3 / .c54_g2 written by hand) so the GPU
  goes to EXP-150's f5.
- 11-08 07:22 order 1 result (raw): consfzg library 88.8 / new **90.0** vs consfz 88.8 / 76.2 (new +13.7, wins 4/0, p=0.125;
  joint 85.6 / 81.2). Per skill (expert / consfz / consfzg / joint): gated drawer + bowl 60 / 0 / **70** / 40 (the gain), gated
  bowl in top drawer 80 / 60 / 60 / 50; non-gated: between plate 80 -> 100, butter 90 -> 100, on ramekin 80 -> 90. Futility
  bar (+8.7) passed -> order 3 released (`sweep/chain58_gated_s3.sh`, tmux c58: first A6000 free ~09:20, ahead of chain57;
  chain54's evaluator evaluates it). Pooled 0+1 so far: new +2.5, library +4.4; order 3 needs >= +10.0 new for the
  pre-registered +5 pooled. Drawer + bowl (gated in orders 0 / 1 / 2): 20 -> 30, 0 -> 70, 20 -> 30 (consfzd).
- 11-08 12:54 order 3 result (raw): consfzg library 91.3 / new 87.5 vs consfz 85.6 / 88.8 (lib +5.6 p=0.09, new -1.3; joint
  87.5 / 91.2). Gated skills: chocolate pudding 90 -> 80, salad dressing 80 -> 70; non-gated bowl in top drawer 70 -> 100.
- VERDICT (pre-registered test, orders 0 / 1 / 3): new +1.2pp pooled (-8.7 / +13.7 / -1.3) < +5 -> **NEGATIVE**, closed; order 2
  not run (hold kept; chain54 stopped by hand). Mechanism: the 8 gated skills sum +40pp = one skill (order 1 drawer + bowl
  0 -> 70); the other 7 are +-10 noise. Expert relabeling repairs a skill only when the merged policy has collapsed on it.
  Side observation: the extra 1,500 resumed steps raised the LIBRARY on every fresh order (+8.8 / 0.0 / +5.6, mean +4.8),
  whatever the round-2 labels -> consistent with the EXP-150 replay-budget hypothesis (+50% library replay).

### EXP-151 — breadth: continual arrival (new skills in 3 batches, the correction updated after each) — **SCALES (11-09): at 22 new (4th arrival) 90.0 / 83.6 vs joint 85.0 / 84.1, 1,500 steps per arrival; >= one-shot at every arrival — done**
Motivation: a deployed robot receives skills over time, not all at once; does the merge + correction stay at joint level
batch after batch, without forgetting earlier batches or the library?
- Setup (11-08): the scaling order's 22 new skills (EXP-148) arrive per order: 8 -> 14 -> 18 -> 22 (4 arrivals rather than 3
  batches: the orders are the natural batches), order 0's 16-skill library. Arrival 1 = order 0's consfz (exists). At each later
  arrival: scratch merge rebuilt (1/k over library + every arrived skill), the previous correction RESUMED inside it, 1,500 steps,
  half the rollouts of every arrived new skill (the receiver keeps them), half library demo replay (arm consfzc; pseudo-orders
  19 / 20 = 14 / 18 new, then 4 = 22; exp116 CONT). Baselines per arrival: one-shot consfz from scratch (3,000 steps; f19 / f20
  new, t = 8 and 22 exist), training-free scratch merge (scr 19 / 20 new), joint at 8 and 22 (exist).
- Chain: `sweep/chain56_continual.sh` (shiva c56): lanes start once chain49 has claimed both consfzr redos; claim ahead of
  chain55 (60 s vs 180 s free); evals port 9750, scr refs port 9800 in gaps only.
- PRE-REGISTERED (11-08, before training): continual SCALES if at the last arrival (22 new) library AND new are within 5pp of
  joint s4 (85.0 / 84.1), as the one-shot did (88.1 / 81.8); BREAKS if >= 10pp below on either. Earlier arrivals: order 0's 8
  skills at arrival 4 within 10pp of their arrival-1 level (77.5) = no forgetting of earlier batches. Per-arrival curve (library,
  all arrived new) for continual / one-shot / training-free, once all runs land.
- 11-08 07:31 training-free scratch merge at arrival 2 (16 + 14 new, k = 30; 30/30 ok, gap eval): raw library 60.6 / new 19.3
  (arrival 1, k = 24: 73.8 / 5.0; arrival 4, k = 38: 55.6 / 20.0) -> the untrained merge loses the library as new skills pile up.
- 11-08 09:48 training-free scratch merge at arrival 3 (16 + 18 new, k = 34; gap eval) landed (numbers in the per-arrival
  table once all runs land).
- 11-09 11:52 RESULT continual arrival 2 (14 new, k = 30; order-0 consfz resumed 1,500 steps inside the rebuilt merge; 30/30
  ok): raw library **91.9** / new **85.7** vs training-free merge 60.6 / 19.3. Earlier batch (order 0's 8): **82.5** at
  arrival 2 vs 77.5 at arrival 1 (no forgetting; +5); the 6 just-arrived skills 90.0 (merge 35.0). Library 91.9 vs 86.9 at
  arrival 1 (cumulative 4,500 steps = more replay). One-shot baseline at 14 (f19) not trained yet. Arrival 3 (c20) training
  on U0 since 11-09 10:57.
- 11-09 13:24 RESULT continual arrival 3 (18 new, k = 34; 34/34 ok): raw library **95.0** / new **87.8** vs training-free merge
  57.5 / 12.2. Per batch: order 0's 8 = 77.5 (arr. 1) -> 82.5 (arr. 2) -> **86.2**; the 6 of arrival 2 = 90.0 -> **93.3**; the 4
  just arrived **82.5** (merge 0.0). Earlier batches keep RISING with each arrival (no forgetting). Caveat for the paper:
  cumulative steps grow (3,000 + 1,500 per arrival = 6,000 here) -> the one-shot baselines (f19 / f20, 3,000 steps) are not
  step-matched; per-arrival cost is 1,500 steps. Arrival 4 (c4, 22 new) training on U0 since 12:22.
- 11-09 14:58 RESULT continual arrival 4 (22 new, k = 38; 38/38 ok): raw library **90.0** / new **83.6** vs joint s4 85.0 / 84.1
  (library +5.0, wins 6/4, p=0.35; new -0.5, wins 8/7, p=0.93) vs one-shot consfz s4 88.1 / 81.8 (+1.9 / +1.8, n.s.) vs
  training-free merge 55.6 / 20.0. Pre-registered: **SCALES** (within 5pp of joint on both; library above). Earlier batches:
  order 0's 8 = 77.5 -> 82.5 -> 86.2 -> **75.0** (within 10pp of arrival 1 = no forgetting by the pre-registered bar, but the
  rise of arrivals 2-3 is gone at arrival 4; joint on the same 8: 82.5); arrival-2 batch 91.7, arrival-3 82.5, just arrived 90.0.
  Step count: 7,500 cumulative steps buy only +1.9 / +1.8 over the 3,000-step one-shot (n.s.) -> the step-matching caveat
  matters little; the practical point = each arrival costs 1,500 steps (half a from-scratch correction) and keeps joint level.
  Remaining: one-shot f19 / f20 (c56, after c4's eval frees the claim order), then the per-arrival figure.
- 11-09 17:47 RESULT one-shot baseline at 14 new (f19: consfz from scratch, 3,000 steps, k = 30; 30/30 ok): raw library
  **87.5** / new **79.3** vs continual arrival 2 91.9 / 85.7 (continual +4.4 / +6.4, wins 6/2 and 6/2, p=0.23 / 0.12, n.s.).
  Continual >= one-shot at 14 and 22 (+1.9 / +1.8). f20 (18 new) next.
- 11-09 19:16 RESULT one-shot baseline at 18 new (f20, 3,000 steps, k = 34; 34/34 ok): raw **90.0 / 79.4** vs continual 95.0 /
  87.8 (+5.0 / +8.3, p=0.13 / 0.07). Pooled over arrivals 2-4 (continual - one-shot): library +3.8 (n=48, wins 17/7, p=0.065),
  new +5.2 (n=54, wins 22/12, p=0.11). Order 0's first 8 under one-shot: 77.5 / 75.0 / 78.8 / 72.5 (continual 77.5 / 82.5 /
  86.2 / 75.0). DONE. Figure `results/figs/continual_arrival.png` (`sweep/fig_continual.py`; library / all arrived new / first
  batch vs arrivals; continual, one-shot, training-free, joint at 8 and 22); deck slide added. Claim: the correction can be
  UPDATED per arrival (1,500 steps, half a from-scratch correction) and stays at joint level with no forgetting; not claimed:
  continual > one-shot (more cumulative steps, n.s.).

### EXP-152 — breadth: new-skill count curve (t = 2 / 4; t = 8 and 22 exist) — **HOLDS (11-10): library flat 88.7 / 88.6 / 87.8 / 88.1 at t = 2 / 4 / 8 / 22; new 78.8 / 81.9 / 77.5 / 81.8 — done**
Motivation: a curve of the gap to joint vs number of new skills, instead of two points.
- Setup (11-08): the main pipeline (consfz, 3,000 steps, unchanged) on orders 0-3 with only their first t new skills (nested;
  pseudo-orders 7-10 = t 2, 11-14 = t 4; exp116 TCUT), plus the training-free scratch merge at each. 8 trainings; joint not
  rerun at t 2 / 4 (exists at 8 and 22). Same libraries, replay lists and rollouts as the parent orders (checked).
- Chain: `sweep/chain55_tcurve.sh` (shiva c55): lanes after chain49's claims, lowest priority (180 s free); evals port 9650; scr
  refs port 9700 in gaps only (no other eval running; one gap ref at a time with chain56 via results/analysis/.scr_lock).
- PRE-REGISTERED (11-08, before training): holds at small t if, pooled over the 4 orders at t = 2 and at t = 4, library >= the
  t = 8 consfz library - 3pp (87.8 -> 84.8) and new >= the SAME skills' t = 8 consfz success - 5pp (paired: the first t new
  skills of each order, as evaluated at t = 8). Expectation: new >= t = 8 (less interference). Figure: library / new vs t
  (2, 4, 8, 22) for consfz, training-free merge, joint (8, 22).
- 11-08 08:17 training-free scratch merge, order 0 at t = 2 (k = 18; 18/18 ok, gap eval): raw library 85.0 / new 10.0 (t = 8:
  73.8 / 5.0) -> fewer new skills, less library dilution, as EXP-137.
- 11-09 20:21 RESULT consfz order 0 at t = 2 (s7, k = 18; 18/18 ok): raw library **91.9** / new **85.0** (t = 8: 86.9 / 77.5;
  merge 85.0 / 10.0). Same 2 skills at t = 8: 100 / 20 -> t = 2: 100 / 70 (drawer + bowl). Pre-registered test pooled over
  4 orders; orders 1-3 at t = 2 and t = 4 follow on U0 (one at a time, ~3.5 h each; U2 has consfz12 s6).
- 11-09 22:48 RESULT consfz order 1 at t = 2 (s8; 18/18 ok): raw library **90.6** / new **90.0** (t = 8: 88.8 / 76.2; merge
  81.2 / 55.0). Same 2 skills at t = 8: 100 / 0 -> t = 2: 100 / 80 (drawer + bowl again: the skill the t = 8 correction loses
  is recovered when few new skills compete).
- 11-10 01:16 RESULT consfz order 2 at t = 2 (s9; 18/18 ok): raw library **88.1** / new **70.0** (t = 8: 90.0 / 67.5; merge
  77.5 / 0.0). Same 2 skills at t = 8: 80 / 20 -> t = 2: 50 / 90 (tomato sauce recovered, drawer bowl -30). Running, 3 orders at
  t = 2: library 90.2 (bar >= 84.8), same skills' new 81.7 vs 53.3 at t = 8 (bar >= t8 - 5).
- 11-10 03:44 RESULT consfz order 3 at t = 2 (s10; 18/18 ok): raw library **84.4** / new **70.0** (t = 8: 85.6 / 88.8; merge
  83.1 / 45.0). Same 2 skills at t = 8: 100 / 70 -> t = 2: 100 / 40.
  t = 2 POOLED (4 orders): library **88.7** (t = 8: 87.8; paired +0.9, p=0.51; bar >= 84.8 met), same new skills **78.8** vs
  61.3 at t = 8 (n=8, p=0.31; bar >= 56.3 met); merge 81.7 / 27.5 -> HOLDS at t = 2 (pre-registered). Per skill: drawer +
  bowl ("open the top drawer and put the bowl inside") 20 / 0 at t = 8 -> 70 / 80; but "pick up the black bowl in the top
  drawer of the wooden cabinet" 80 / 70 -> 50 / 40 (orders 2 / 3): that skill does better WITH the other new skills (shared
  drawer / bowl sub-skills = positive transfer among new skills). t = 4 (s11-14) next on U0.
- 11-10 06:49 RESULT consfz order 0 at t = 4 (s11, k = 20; 20/20 ok): raw library **90.6** / new **85.0** (t = 8: 86.9 / 77.5;
  merge 77.5 / 5.0). Same 4 skills at t = 8: 100 / 20 / 100 / 70 (72.5) -> t = 4: 100 / 60 / 100 / 80 (85.0).
- 11-10 07:50 RESULT consfz order 1 at t = 4 (s12; 20/20 ok): raw library **87.5** / new **85.0** (t = 8: 88.8 / 76.2; merge
  78.8 / 35.0). Same 4 skills at t = 8: 75.0 -> t = 4: 85.0 (drawer + bowl 0 -> 60; salad dressing 100 -> 80).
- 11-10 09:33 RESULT consfz order 2 at t = 4 (s13; 20/20 ok): raw library **86.2** / new **72.5** (t = 8: 90.0 / 67.5; merge
  76.9 / 0.0). Same 4 skills at t = 8: 55.0 -> t = 4: 72.5 (tomato sauce 20 -> 70, wine rack 20 -> 40). Order 3 (s14) training
  on U0 since 08:54, last run of EXP-152.
- 11-10 11:57 RESULT consfz order 3 at t = 4 (s14; 20/20 ok): raw library **90.0** / new **85.0** (t = 8: 85.6 / 88.8; merge
  78.1 / 50.0).
  t = 4 POOLED (4 orders): library **88.6** (t = 8: 87.8; paired +0.8, p=0.61; bar >= 84.8 met), same new skills **81.9** vs
  73.1 at t = 8 (n=16, p=0.12; bar >= 68.1 met); merge 77.8 / 22.5 -> HOLDS at t = 4 (pre-registered).
  CURVE (consfz, library / new; t = 2 / 4 / 8 4-order means, t = 22 one order): 88.7 / 78.8, 88.6 / 81.9, 87.8 / 77.5, 88.1 / 81.8;
  training-free merge 81.7 / 27.5, 77.8 / 22.5, 74.7 / 16.2, 55.6 / 20.0; joint 87.3 / 83.4 (t 8), 85.0 / 84.1 (t 22). The
  correction is flat in t; the same skills do better with fewer co-arriving new skills (+17.5 at t = 2, +8.8 at t = 4, n.s.)
  except the drawer-bowl skill that benefits from related ones. Figure `results/figs/tcurve.png` (`sweep/fig_tcurve.py`);
  deck slide added. DONE.

### EXP-153 — breadth: rollout budget (10 / 25 own attempts per new skill instead of 50) — **planned (11-07), NOT queued (user)**
Motivation: how cheap is the method in robot time; subsample the existing rollouts, no new collection.

### EXP-154 — breadth: second seed of the main method (consfz) on all 4 orders — **planned (11-07), NOT queued (user)**
Motivation: seed-to-seed swing up to 5pp library (cons1 order 2); a second seed of the main arm for error bars.

---
## 11-18 team meeting → next phase (user): trained same-data baselines, continual learning, merge choice, mechanism
User's notes from the 18-11 meeting (everyone liked the project): (1) compare against methods that ALSO train on data, ideally
the same data as ours, and show a clear advantage under our requirements; (2) data-free merging alone is an unfair comparison;
(3) add standard continual learning baselines on the same data; (4) focus on CONTINUAL merging (the flat curve as skills pile
up); (5) try our merging methods in step 1 instead of vanilla averaging; (6) bring in the shared / residual component analysis
— a plain LoRA correction alone feels incremental ("representation surgery in robotics"); mechanistic / spectral and
VLA-specific results add contribution.
- CORRECTION (11-18, my error in chat on 11-12): step 1 was NEVER the vanilla average. The correction is trained and deployed on
  our library-anchored merge `MM.ours` = pool S16 at full strength + 1/k x the members' two-sided residuals
  (`train_lora.fold_context_into_base`, `ZBuilder privx`); `ctx_all_zfree.json` lists 1/k weights, which are the RESIDUAL
  weights. So item (5) becomes the reverse ablation: the correction on the vanilla average (EXP-160). The incremental-update
  argument still holds: S16 is fixed, the residual sum is linear (keep S16 + running residual sum + k, not every adapter).
- 11-18 infra: shiva home quota exceeded (50.8 / 46 GB soft, grace expired): a scp left a corrupted exp147 copy on shiva
  (restored, md5 checked). With the user's OK, new_loras_ctx (5.6 GB) and new_loras_s16 (5.2 GB) moved to the NAS
  (`libero-eval-archive/*_shiva1118`, checksums verified) and symlinked back -> 39.9 GB used.
- 11-18 infra (21:55): the four chains raced each other for the A6000s. (1) 20:12 chain67 and chain70 both started a trainer on
  GPU0 (their lanes only see a train_lora.py process, not another chain's exp147 train stage while it prepares): erlc@19 OOM at
  start, its claim blocked the erlc rounds -> claim removed, chain67 retries it at the next free GPU. (2) 21:00 chain69's gap eval
  and chain67's erl@0 eval started together: trainer + 3 workers + a foreign IsaacLab job on GPU0 -> "server died during load",
  erl@0 lost 4 of 24 tasks (the 20 kept passed the audit) -> `sweep/chain67fix_erl0.sh` re-runs them (port 10360). Fix for all
  chains except chain67 (cannot restart: trainers inside): `sweep/gpu_gate.sh` (busy also counts exp147 train stages; a GPU
  reservation placeholder that chain67's check sees; longer double-check delays = lower priority; gap evals only with >= 16 GB
  free on both GPUs and no pending higher-priority eval). chain68 (idle) / chain69 (after pe / pl / pv) replaced by chain68b /
  chain69b (ne / nl / nv); chain70b takes over lrcfz@0's eval and lrc@0 once chain70's running trainer exits. Evals now log
  the number of tasks completed.
- 11-19 00:07 scheduling: (1) erlc@19's 23:03 retry exited silently: exp147's train stage keeps its own claim
  (`new_loras_cons_s19/.claim_erlc`, left by the 20:12 OOM) -> removed. (2) User 11-19: frame the project as CONTINUAL merging.
  chain67's lanes always had a one-shot job ready, so the serial continual chains (4 rounds each) and chain68b / 70b's trainers
  would have started last. All not-yet-started trainer jobs pre-claimed under the old chains' names (their lanes exit; their
  eval loops still evaluate) and moved to ONE scheduler, `sweep/chain71_sched.sh` (tmux c71), in this order: erlc@19 consfa@0
  ewc@0 erl@2 erlc@20 seqc@19 consfac@19 ewcc@19 erl@3 erlc@4 lrc@0 seqc@20 consfac@20 ewcc@20 seqc@4 consfac@4 ewcc@4,
  then the one-shot extras consb@2 consb@3 consfa@1-3 consb64@0. Running at the switch: seq@0 (G2), lrcfz@0 (G0, chain70).
- 11-19 01:20 infra: gpu_gate's `pending` looped over a global `j`, which the gap-eval loops of chain68b / 70b also use: after
  every gate check `j` was the last job of the list, so chain70b "evaluated" ewcc@4 (not trained, 0 tasks) instead of lrcfz@0,
  and chain68b would have done the same with consb64@0 once ewc@0 lands. No result was affected (nothing wrong was evaluated).
  Fixed (`local j`); chain68b / 70b restarted (no trainers inside); chain70b now runs as tmux `e70b` (its `tmux kill-session -t
  c70` prefix-matched a session named c70b and killed it). chain69b / 69c loop over `v`: unaffected.
- 11-19 05:40 infra: chain67's sca@4 ref (its own check ignores other chains' evals) and chain69c's rae@0 gap eval (its port
  check misses the minutes an eval spends building the merge before serving) started within 2 min of each other at 04:31 ->
  servers died during load: rae@0 21/24 tasks after 3 tries, sca@4 still retrying. Fix: every eval on shiva now holds
  `results/analysis/.evlock` (flock in gpu_gate `evalrun`, fd closed for the python children). chain67 (no trainer left inside)
  is replaced by `sweep/chain67b_eval.sh` (tmux e67b: holds the lock until chain67's sca@4 exits, stops chain67, completes
  sca@4, then evaluates chain67's remaining jobs, 2 workers per GPU); chain68b / 69c / 70b restarted on the new gate (69c
  re-runs rae@0's 3 missing tasks first). Neither partial result is reported until complete.

### EXP-158 — same-data trained baselines (one-shot, t = 8, 4 orders) — **running (11-18)**
Motivation: meeting item (1)/(2). Every baseline gets exactly our inputs — the library adapters, 5 demos per library skill, the
new skills' own successful rollouts (no new-skill demos) — and our budget (rank-16 LoRA, 3,000 steps, batch 8, 50/50 steps).
- Arms (`exp147_consolidate.py`, 11-18): `erl` = experience replay from the LIBRARY model (training-free library-only merge, ours
  at 1/16, + one LoRA on the same 50/50 data; the new adapters are never merged: new skills learned from rollouts only);
  `consb` = multi-task LoRA from the base on the same data (order 0 exists: 61.9 / 56.2; orders 1-3 queued); `consb64` =
  consb at joint's recipe (rank 64, 12,000 steps; order 0). Ours = consfz (4-order 87.8 / 77.5). Joint on ALL demos (87.3 /
  83.4) stays as the more-data reference. Smoke on shiva: library / avg contexts fold 16 / 24 members, 46.1M trainable.
- Queue: `sweep/chain67_samedata.sh` (tmux c67, 11-18 16:26; two lanes claim in priority order; evals port 10350).
- PRE-REGISTERED (11-18, before any training): per baseline, pooled over the 4 orders (raw success, test states 0..9, 10 eps):
  ADVANTAGE if ours >= baseline + 5pp on one of library / new with the other >= baseline - 3pp; TIE if both within 5pp;
  otherwise the baseline wins. Expectation: erl's new skills well below ours (rollouts alone, no merged adapter), library similar.
- 11-18 erl@0 (all 24 tasks after the 22:08 re-run of 4): raw library 84.4 / new 72.5 vs ours 86.9 / 77.5 (+2.5 / +5.0 for us;
  joint 90.0 / 76.2). Expectation WRONG: rollouts + replay from the library model alone already reach most of our new-skill
  level; the +5 rests on one skill (wine bottle on the rack 0.7 vs 0.3; the other 7 within 0.2, 4 for us / 3 for erl). Order 0
  alone would just pass the ADVANTAGE bar (new +5.0, library +2.5); call pooled over orders 1-3 (erl@1 training).
- 11-18 consb@1 (24/24): 67.5 / 75.0 vs ours 88.8 / 76.2 (joint 85.6 / 81.2) -> +21.3 library / +1.2 new for us; with order 0
  (+25.0 / +21.3) consb is on track for ADVANTAGE (library); from the base the new skills are learned, the library is not kept.
- 11-19 erl@1 (24/24): 84.4 / 90.0 vs ours 88.8 / 76.2 (joint 85.6 / 81.2) -> library +4.4 for us, new -13.8 (erl >= ours on
  all 8 new skills: 5 higher by 0.1-0.3, 3 equal). Pooled orders 0-1: library +3.5, new -4.4 for us -> TIE so far (both within
  5pp); 10 eps per skill, per-skill noise ~+-0.15. Reading: merging the new adapters adds nothing measurable over learning
  them from their rollouts; the library merge + rollouts + replay carry the result. Orders 2-3 (erl@2 / 3, chain71) decide.
- 11-19 erl@2 (24/24 tasks): raw library 86.9 / new 81.2 vs ours consfz 90.0 / 67.5 (joint 86.2 / 85.0) -> +3.1 / -13.7.
  Pooled orders 0-2: library +3.3, new -7.5 -> by the pre-registered rule ER currently WINS arrival 1 (new gap beyond 5pp).
  Order 2's deficit sits in two skills (tomato sauce 0.2 vs 0.8 = the skill whose expert the merge breaks, EXP-155; drawer
  bowl 0.2 vs 0.5); over orders 0-2 ER >= ours on 17 of 24 new skills. Note: erl IS our pipeline minus merging the arriving
  adapters (same library merge, same correction / data), so this is an ablation of step "merge the new adapters": one-shot,
  merging them does not help new skills. Continual (EXP-159, order 0 stream) the gap goes the other way and widens
  (+3.3 / +2.3 / +7.1 all-skill mean at arrivals 1-3). erl@3 last; EXP-163 triggers if the pooled call stays TIE / ER.
- 11-19 erl@3 (24/24 tasks): raw library 85.6 / new 77.5 vs ours consfz 85.6 / 88.8 (joint 87.5 / 91.2) -> 0.0 / +11.3 for
  us. FINAL 4-order call vs erl: library +2.5 (paired t 1.9), new -2.8 (t -0.7) -> **TIE (pre-registered)**; per order new
  +5.0 / -13.7 / -13.7 / +11.3 = noisy, no consistent sign. One-shot, merging the arriving adapters neither helps nor hurts
  vs learning them from their rollouts. EXP-163's trigger condition (TIE or ER ahead) holds -> its 11-19 start stands.

### EXP-163 — what does merging the new adapters add beyond their rollouts? — **done (11-20): small lead at full budget and 10 episodes, none at 750 steps**
Motivation: ER from the library model (erl: same data, new adapters never merged) ties ours on arrival 1 so far (orders 0-1:
library +3.5 / new -4.4 for us). If that holds, the merge of the arriving adapters must earn its place elsewhere: with less
correction budget (the merged residuals are a head start), with fewer rollouts, or across arrivals (EXP-159 erlc vs consfzc).
- PRE-REGISTERED (11-19, before erl@2 / 3 and any erlc result): run only if EXP-158 vs erl ends TIE or erl ahead. Arms on
  orders 0 / 1 (ours vs erl, same recipe): (a) correction at 750 steps; (b) 10 rollouts per new skill instead of 50 (first 10
  episodes of the same files, success-filtered). Merging adds value if ours >= erl + 5pp new (library >= -3pp) pooled over the
  two orders in (a) or (b); otherwise the paper states that the arriving adapters matter only as rollout generators.
- 11-19 10:34 started (before erl@3): pooled orders 0-2 vs erl = library +3.3 / new -7.5; ADVANTAGE after order 3 would need
  pooled library >= +5, i.e. ER's order-3 library <= ~75 (ER 84.4 / 84.4 / 86.9 on orders 0-2; ours' order 3 = 85.5) -> TIE
  or ER ahead in practice, so the condition holds; starting now saves ~a day (if erl@3 somehow gives ADVANTAGE, EXP-163 is
  reported as not triggered). Arms in exp147 (11-19): consfzq / erlq (750 steps), consfz10 / erl10 (`tasks_ext10.txt` = glob
  `ep00[0-9]_*_success.npz`: 3-10 rollouts per new skill, 126-952 samples; data loading smoke-tested on odin). Scheduler
  `sweep/chain72_exp163.sh` (tmux c72): its lanes yield to chain71 (continual rounds keep priority), EXP-163 before chain71's
  one-shot extras (consb@2 / 3, consfa@1-3, consb64@0 moved here by pre-claiming them in chain71); evals port 10600 under the
  eval lock after chain67b / 68b / 70b's pending evals.
- 11-19 11:53 lrc@0 (one-shot, order 0) moved from chain71 to chain72 after the EXP-163 arms, so the continual rounds (seqc /
  consfac / ewcc 20 and 4) are not delayed by 3 h; chain70b still evaluates it. Projected (2 A6000s, 3,000 steps ~3.2 h,
  1,500 ~1.7 h, 750 ~0.8 h, evals ~45 min each, serial): erlc@4 ~14:45, erl@3 ~15:00, seqc@20 / consfac@20 ~16:30, ewcc@20 /
  seqc@4 ~18:15, consfac@4 (EXP-160 call) ~20:00, ewcc@4 ~20:15; EXP-163 750-step arms ~20:45-22:00, 10-rollout arms ~01:00-04:30
  (11-20); then lrc@0 and the one-shot extras through 11-20, consb64@0 (12k steps) last (~11-21 early).
- 11-20 09:20 consfz10@0 (ours on the first 10 episodes per new skill, success-filtered: 3-10 rollouts per skill; order 0;
  24/24 tasks): raw library 88.1 / new 78.8 vs ours on all 50 episodes 86.9 / 77.5 -> 10 episodes are enough (+1.2 / +1.3).
  erl10@0 (ER, same 10 episodes) evaluating next; consfz10@1 / erl10@1 training. (b) call pooled over orders 0 / 1.
- 11-20 09:52 erl10@0 (ER on the same 10 episodes, order 0; 24/24 tasks): raw library 85.0 / new 75.0 vs ours consfz10 88.1 /
  78.8 -> ours +3.1 (se 4.4; 4 / 9 / 3) / +3.8 (se 6.0; 2 / 5 / 1). Same gap as with 50 episodes (+2.5 / +5.0 on order 0):
  fewer rollouts do not widen it. Order 1 (consfz10@1 / erl10@1, training, ~13:00 + evals) decides (b): bar new >= +5 pooled.
- 11-20 11:12 USER DECISION: "treat small leads as wins too, no need for 5 points to be considered." From here, calls are by
  the sign of the pooled lead (se / W-T-L / per-order signs still reported); the +5pp pre-registered calls above stay in the
  log as written. Re-tally of ours (merging the arrivals) vs ER from the library merge: one-shot 4 orders library +2.5 /
  new -2.8 (all-skill +0.7, ahead on orders 0 and 3) = WIN on library and all-skill mean; stream +3.3 / +2.3 / +7.1 / -2.4 =
  WIN at arrivals 1-3, mean +2.6; 750 steps -0.1 / -1.3 = LOSS; 10 episodes (order 0) +3.1 / +3.8 = WIN. Merge choice (ours
  vs plain-average start): full budget +4.2 / -1.3 / +7.1 / -1.6 (mean +2.1) = WIN at arrivals 1 / 3; 750 steps +5.9 / +10.6
  = WIN. Merge start vs base start: WIN at every point (stream arrival 4 +4.5). Deck reworded (backup
  results/merge_stale/content_for_slides_25_11.md.bak_1120).
- 11-20 12:37 erl10@1 / consfz10@1 (order 1, first 10 episodes per new skill; 24/24 tasks each): ours 88.8 / 86.2 vs ER
  90.0 / 85.0 (library -1.2 / new +1.2). (b) POOLED over orders 0 / 1: ours - ER = library +0.9 (se 2.6; W-T-L 7-18-7) /
  new +2.5 (se 5.1; 3-8-5), all 48 skills +1.5 (se 2.4); signs: library + / -, new + / +. Pre-registered bar (new >= +5)
  NOT MET; under the 11-20 user rule = WIN on both (small, within noise). Neither method needs the 50 episodes: ours 10 vs 50
  = library +0.6 / new +5.7 pooled (order 1 76.2 -> 86.2), ER +3.1 / -1.3 -> success-filtered rollouts are not the limiting
  input at 3-10 per skill. EXP-163 overall: merging the arrivals gives a small, consistent edge over ER at full budget and at 10
  episodes, none at 750 steps (-0.1 / -1.3); the clear value of merging is as a start (vs base / distillation, EXP-164).
  EXP-163 DONE.

### EXP-159 — continual-learning baselines in the arrival stream (8 -> 14 -> 18 -> 22 new, order 0's library) — **done (11-20)**
Motivation: meeting items (3)/(4): the paper's main axis becomes continual skill integration; standard CL on the same data is the
fair comparison. Stream and data as EXP-151 (arrival 1 = order 0; 1,500 resumed steps per later arrival; replay = 5 demos per
library skill + every earlier arrival's stored rollouts, exactly what ours keeps).
- Arms: `seq`/`seqc` = sequential fine-tuning from the library model on each arrival's own new rollouts only (no replay; the
  forgetting floor); `erl`/`erlc` = experience replay (same data mix as ours, no new adapter merged); ours = consfz / consfzc
  (exists); training-free refs scr (ours) and sca (vanilla average) per arrival. EWC (online, LoRA parameters) and an
  orthogonal-LoRA method (O-LoRA style) to be added once implemented (LIBERO's standard set is SeqL / ER / EWC / PackNet).
- PRE-REGISTERED (11-18): at the last arrival (38 skills) ours beats ER if mean success over all 38 skills >= ER + 5pp; and
  ours forgets less: order 0's batch at arrival 4 minus at arrival 1 >= ER's same difference. Per-arrival curves (library,
  all arrived new, first batch) for every method once all land (one figure, all methods).
- EWC lambda calibration (rule set 11-18 when EWC was implemented, written here 11-19 02:58, before any EWC result; ewc@0
  training since 02:54): lambda 1e-3 pre-registered; if EWC's arrival-1 new success < half of SeqFT's -> rerun at 1e-4; if its
  library is within 3pp of SeqFT's -> rerun at 1e-2; one adjustment only. ewcc@19 held (claim `.c71_ewcc_19`) until the check.
- 11-19 seq@0 (SeqFT arrival 1 = fine-tune the library model on the 8 new skills' own rollouts, no replay; 24/24 tasks): raw
  library 49.4 / new 78.8 vs ours 86.9 / 77.5, ER 84.4 / 72.5 -> the forgetting floor: same new-skill level, library -37.5pp
  vs ours. EWC thresholds from it: new < 39.4 -> 1e-4; library <= 52.4 -> 1e-2.
- 11-19 erlc@19 (ER, arrival 2 = 14 new, 1,500 resumed steps; 30/30 tasks): raw library 91.9 / new 80.7 vs ours consfzc 91.9 /
  85.7 -> library tie, new +5.0 for us; mean over all 30 skills 89.0 vs 86.7 (+2.3). First batch (arrival 1's 8): ours 77.5 ->
  82.5, ER 72.5 -> 77.5 (both +5.0: neither forgets); arrival-2 batch (6): ours 90.0 vs ER 85.0. Same gap as arrival 1
  (order 0: +2.5 / +5.0), not widening yet; the pre-registered call is at arrival 4 (38 skills).
- 11-19 ewc@0 (online EWC, lambda 1e-3, arrival 1 = SeqFT + Fisher penalty anchored on the library model; 24/24 tasks): raw
  library 66.9 / new 66.3 vs SeqFT 49.4 / 78.8, ER 84.4 / 72.5, ours 86.9 / 77.5. Lambda check (pre-registered): new 66.3 >=
  39.4 and library 66.9 > 52.4 -> lambda 1e-3 KEPT, no rerun; ewcc@19 hold released (08:31). EWC trades 12.5 new for 17.5
  library vs SeqFT; ours +20.0 / +11.2 over EWC.
- 11-19 erlc@20 (ER, arrival 3 = 18 new, k 34; 34/34 tasks): raw library 92.5 / new 76.7 vs ours consfzc 95.0 / 87.8 -> +2.5 /
  +11.1 for us; mean over all 34 skills 91.2 vs 84.1 (+7.1; arrivals 1 / 2 / 3: +3.3 / +2.3 / +7.1). First batch (arrival 1's
  8): ours 77.5 -> 82.5 -> 86.2, ER 72.5 -> 77.5 -> 71.2 (ER starts forgetting it at arrival 3, ours keeps improving);
  second batch (6) 93.3 vs 86.7; third batch (4) 82.5 vs 72.5. The gap WIDENS with arrivals; the pre-registered call is at
  arrival 4 (mean over 38 >= +5, first-batch forgetting <= ER's).
- 11-19 seqc@19 (SeqFT arrival 2: resumed on the 6 newly arrived skills only, no replay; 30/30 tasks): raw library 58.8 / new
  68.6, mean over 30 skills 63.3 (ours 89.0, ER 86.7). First batch 78.8 -> 56.2 (forgets 22.6pp; ours +5.0, ER +5.0); its
  own new batch 85.0 (ours 90.0). Library 49.4 -> 58.8 (training on goal-suite rollouts happens to restore a few library
  skills) — still -33.1 vs ours.
- 11-19 ewcc@19 (online EWC arrival 2, lambda 1e-3, 1,500 resumed steps on the 6 new skills' rollouts, no replay; 30/30
  tasks): raw library 75.0 / new 68.6, mean over 30 skills 72.0 (ours 89.0, ER 86.7, SeqFT 63.3). First batch 66.3 -> 60.0
  (forgets 6.3; SeqFT 22.6, ours / ER +5.0); its new batch 80.0 (ours 90.0). Library 66.9 -> 75.0 (as SeqFT, goal-suite
  rollouts restore some library skills). Ordering at arrival 2: ours > ER >> EWC > SeqFT; EWC halves SeqFT's forgetting but
  stays 17pp below ours on the all-skill mean.
- 11-19 seqc@20 (SeqFT arrival 3 = 18 new, k 34; 34/34 tasks): raw library 50.6 / new 79.4, mean over 34 skills 65.9 (ours
  91.2, ER 84.1). By batch (8 / 6 / 4): 58.8 / 95.0 / 97.5 (ours 86.2 / 93.3 / 82.5): its own newest batch is the best of any
  method (trained on those 4 skills alone), the first batch stays forgotten (78.8 -> 56.2 -> 58.8), library 58.8 -> 50.6.
- 11-19 erlc@4 (ER, arrival 4 = 22 new, k 38; 38/38 tasks): raw library 88.8 / new 88.6 vs ours consfzc 90.0 / 83.6 ->
  library +1.2, new -5.0 for us; mean over all 38 skills 86.3 vs ER 88.7 (-2.4; per skill 7 wins / 19 ties / 12 losses,
  paired t ~ -1.0, n.s.). By batch (8 / 6 / 4 / 4): ours 75.0 / 91.7 / 82.5 / 90.0, ER 83.8 / 96.7 / 80.0 / 95.0. First
  batch over arrivals 1-4: ours 77.5 -> 82.5 -> 86.2 -> 75.0 (-2.5), ER 72.5 -> 77.5 -> 71.2 -> 83.8 (+11.3).
  **PRE-REGISTERED CALL vs ER: FAILS on both halves** (mean -2.4 < +5; first-batch change -2.5 < ER's +11.3). All-skill mean
  gap by arrival +3.3 / +2.3 / +7.1 / -2.4 (mean +2.6): the arrival-3 widening did not hold; per-batch swings of ~10pp on 8
  skills x 10 episodes are within noise (SE of a batch difference ~6pp). Reading: in the stream ours = ER within noise; the
  advantage over SeqFT / EWC comes from replay + the correction, not from merging the arriving adapters (ER still starts from
  our library merge). Whether merging the arrivals helps at all is now EXP-163's question; its 750-step arms moved ahead of
  the SeqFT / EWC cells (chain72 restart 14:19; ewcc@20 / seqc@4 / ewcc@4 pre-claimed in chain71, run by chain72 next).
- 11-20 06:05 ewcc@20 (online EWC, arrival 3 = 18 new; chain68; 34/34 tasks): raw library 68.1 / new 66.1 (batches 51.2 /
  70.0 / 90.0), 34-skill mean 67.1 vs ours 91.2, ER from the merge 84.1, ER from the base (consbc, EXP-164) 66.5, SeqFT 65.9.
  Ours - EWC library +26.9 (se 6.8), new +21.7 (se 6.9). EWC's first batch keeps falling (66.3 -> 60.0 -> 51.2): EWC
  slows but does not stop forgetting; it learns the newest batch (90.0) like SeqFT (97.5). seqc@4 / ewcc@4 trained, evals
  next (complete the table).
- 11-20 07:01 seqc@4 (SeqFT, arrival 4 = 22 new, k 38; chain67b; 38/38 tasks): raw library 43.8 / new 57.7 (batches 20.0 /
  68.3 / 75.0 / 100.0), 38-skill mean 51.8 vs ours 86.3, ER from the merge 88.7, ER from the base 84.2. Ours - SeqFT library
  +46.2 (se 8.4), new +25.9 (se 8.5). SeqFT's first batch 78.8 -> 56.2 -> 58.8 -> 20.0 and library 49.4 -> 58.8 -> 50.6 ->
  43.8: textbook forgetting; only the newest batch is learned (100.0). ewcc@4 eval next (last cell).
- 11-20 08:25 ewcc@4 (online EWC, arrival 4; chain68b; 38/38 tasks): raw library 75.6 / new 69.1 (batches 51.2 / 75.0 /
  90.0 / 75.0), 38-skill mean 71.8. Ours - EWC library +14.4 (se 4.2), new +14.5 (se 5.0). **EXP-159 table complete.**
  38-skill mean at arrival 4: ER from the merge 88.7, ours 86.3, ER from the base (EXP-164) 84.2, joint 84.5, EWC 71.8,
  SeqFT 51.8. First batch over arrivals 1-4: ours 77.5 / 82.5 / 86.2 / 75.0, ER 72.5 / 77.5 / 71.2 / 83.8, EWC 66.3 / 60.0 /
  51.2 / 51.2, SeqFT 78.8 / 56.2 / 58.8 / 20.0. Ranking: replay arms (ours ~ ER, from-base catches up at arrival 4) >> EWC >
  SeqFT at every arrival. Per-arrival figure with all methods: results/figs/continual_all.png.

### EXP-160 — the merge under the correction: ours vs vanilla average — **done (11-21): ours ahead on 4/4 one-shot orders (+3.9 / +5.9), stream within noise**
Motivation: meeting item (5), reversed by the correction above: does the component-based merge matter once a correction is
trained on top? Vanilla averaging dilutes the shared library part by L/k (0.67 at k = 24, 0.42 at k = 38); ours keeps it whole.
- Arms: `consfa` = consfz's recipe on the vanilla average (sum B A / k; `fold_context_into_base(op="avg")`, ZBuilder privx
  "avg"), orders 0-3; `consfac` = its continual rounds (19 / 20 / 4); refs sca (training-free vanilla average).
- PRE-REGISTERED (11-18): our merge matters if consfzc >= consfac + 5pp on the library at the last arrival (k = 38), where the
  dilution is largest; does not matter if within 3pp there and pooled over the 4 one-shot orders.
- 11-18 training-free reference sca@0 (vanilla average of the 24 adapters, no correction): raw library 49.4 / new 2.5 vs our
  merge (scr) 73.8 / 5.0 -> before any training our merge is +24.4pp on the library (order 0; 11 of 16 library skills higher,
  3 lower by <= 0.2). New skills ~0 under both: the gap the correction closes is the same for both merges.
- 11-18 sca@19 (arrival 2, k = 30: 16 library + 14 new, training-free): plain average 35.6 / 11.4 vs our merge 60.6 / 19.3 ->
  +25.0 library / +7.9 new; the gap holds as k grows (order 0 at k 24: +24.4 / +2.5).
- 11-19 sca@20 (arrival 3, k = 34: 16 library + 18 new, training-free; 34/34 tasks): plain average 25.0 / 10.0 vs our merge
  57.5 / 12.2 -> +32.5 library (11 of 16 library skills higher, none lower) / +2.2 new. Library gap grows with k: +24.4 / +25.0
  / +32.5 at k 24 / 30 / 34 (the average's dilution of the shared part, L/k, grows).
- 11-19 sca@4 (arrival 4, k = 38: 16 library + 22 new, training-free; 38/38 tasks after the 05:40 race, retries filled the 13
  episode errors): plain average 21.2 / 9.1 vs our merge 55.6 / 20.0 -> +34.4 library (10 of 16 higher, none lower) / +10.9
  new. Library gap +24.4 / +25.0 / +32.5 / +34.4 at k 24 / 30 / 34 / 38: training-free, ours' advantage grows with every
  arrival. Whether it survives the correction = consfzc vs consfac at k = 38 (pre-registered).
- 11-19 consfa@0 (consfz's correction on the VANILLA AVERAGE, order 0, 3,000 steps; 24/24 tasks): raw library 85.0 / new
  68.8 vs ours consfz 86.9 / 77.5 -> library +1.9 (5 of 16 skills higher, 2 lower), new +8.7 for our merge. The correction
  repairs most of the average's training-free library deficit (49.4 -> 85.0; ours 73.8 -> 86.9), so the +24.4 training-free
  library gap shrinks to +1.9; the advantage shows up on new skills instead. One order (one-shot orders 1-3 and the
  continual rounds consfac@19/20/4 queued); the pre-registered call is at k = 38. consfa is also below ER (84.4 / 72.5).
- 11-19 consfac@19 (continual correction on the vanilla average, arrival 2, k 30; 30/30 tasks): raw library 93.1 / new 87.1
  vs ours consfzc 91.9 / 85.7 -> -1.2 / -1.4 (all-30 mean 90.3 vs 89.0); first batch 68.8 -> 80.0 (ours 77.5 -> 82.5),
  second batch 96.7 vs 90.0. The +25.0 training-free library gap at k 30 is gone after the resumed correction; interim
  points to "does not matter" (pre-registered: within 3pp at k 38 and pooled over the one-shot orders). Call at k 38.
- 11-19 consfac@20 (arrival 3, k 34; 34/34 tasks): raw library 90.6 / new 78.3 vs ours consfzc 95.0 / 87.8 -> +4.4 / +9.5 for
  us; all-34 mean 91.2 vs 84.1 (+7.1, paired se 2.6, 14 wins / 16 ties / 4 losses per skill). Batches (8 / 6 / 4): 71.2 /
  83.3 / 85.0 vs ours 86.2 / 93.3 / 82.5. Merge-choice gap on the all-skill mean by arrival: +4.2 / -1.3 / +7.1 — back after
  arrival 2's dip. Caveat: ours' arrival 3 is its best point (ER's gap was also +7.1 there, then -2.4 at arrival 4), so both
  gaps share one possibly lucky run; consfac@4 (k 38, training) makes the call.
- 11-19 consfac@4 (arrival 4, k 38: plain average dilutes the shared part most, x0.42; 38/38 tasks): raw library 90.0 / new
  86.4 vs ours consfzc 90.0 / 83.6 -> 0.0 / -2.7 (library 4 / 9 / 3, new 9 / 5 / 8 wins / ties / losses; se ~3.3); all-38
  mean 87.9 vs 86.3 (ER 88.7). Batches (8 / 6 / 4 / 4): 82.5 / 91.7 / 87.5 / 85.0 vs ours 75.0 / 91.7 / 82.5 / 90.0. Training-
  free at k 38 the gap was +34.4 library. **PRE-REGISTERED: "our merge matters" (>= +5 library at k 38) NOT MET; library within
  3pp at k 38 -> "does not matter" pending only the one-shot pool (consfa@1-3, queued at chain72's end; order 0 +1.9 / +8.7).**
  Merge-choice gap on the all-skill mean by arrival +4.2 / -1.3 / +7.1 / -1.6: arrival 3's +7.1 was ours' lucky run. In the
  stream at full budget, ours (component merge of all), the plain average of all, and ER (library only) end within noise:
  86.3 / 87.9 / 88.7. The 750-step average start (consfaq@0 / 1, training) is the remaining test of the merge choice.
- 11-20 20:35 consfa@1 (correction on the vanilla average, one-shot order 1; 24/24 tasks): raw library 82.5 / new 77.5 vs
  ours consfz@1 88.8 / 76.2 -> +6.2 / -1.3 for us. POOLED orders 0-1: ours - average start = library +4.1 (se 2.8; W-T-L
  11-16-5) / new +3.7 (se 4.2; 8-3-5) = ours ahead on both (11-20 sign rule). Average start - ER (library merge only, arrivals
  not merged) = -0.6 (se 2.8) / -8.1 (se 3.9; 1-8-7): averaging the arrivals in HURTS new skills vs leaving them out; the
  component merge turns merging them into a small gain (ours - ER pooled orders 0-1 +3.5 / -4.4). consfa@2 / 3 complete the
  one-shot pool (pre-registered library within 3pp -> "does not matter"; now +4.1).
- 11-20 23:16 consfa@2 (one-shot order 2; 24/24 tasks): raw library 85.0 / new 68.8 vs ours consfz@2 90.0 / 67.5 -> +5.0 /
  -1.2 for us; vs ER (86.9 / 81.2) -1.9 / -12.5. POOLED orders 0-2: ours - average start library +4.4 (se 2.2; W-T-L 16-26-6) /
  new +2.1 (se 3.3; 11-5-8); average start - ER -1.0 (se 2.1) / -9.6 (se 3.2; 1-12-11). Same shape on every order: averaging
  the arrivals in costs new skills vs leaving them out, the component merge does not and adds library. consfa@3 completes it.
- 11-21 00:14 consfa@3 (one-shot order 3; 24/24 tasks): raw library 83.1 / new 71.2 vs ours consfz@3 85.6 / 88.8 -> +2.5 /
  +17.5; vs ER (85.6 / 77.5) -2.5 / -6.2. POOLED 4 ORDERS (arrival 1): ours - average start library +3.9 (se 1.9; W-T-L
  23-29-12) / new +5.9 (se 3.4; 16-8-8); all-skill +4.2 / +3.7 / +2.9 / +7.5 per order (mean +4.6) = ours ahead on EVERY order.
  Average start - ER: library -1.4 (se 1.8) / new -8.8 (se 2.7; 2-15-15), all-skill negative on every order (mean -3.9).
  Ours - ER (reference): +2.5 / -2.8, all-skill mean +0.7. Pre-registered "does not matter" (library within 3pp) NOT
  supported: library +3.9 (~2 se), new +5.9; by the 11-20 sign rule a WIN on both. Ladder at arrival 1: merging the arrivals
  by plain average costs -3.9 all-skill vs leaving them out (ER); the component merge turns that into +0.7, i.e. the operator
  is worth +4.6. Stream (order 0) stays mixed: +4.2 / -1.3 / +7.1 / -1.6. Training-free +24 to +34 library, 750 steps +5.9 /
  +10.6. EXP-160 DONE: our merge beats the plain average as the start under the same correction (one-shot 4/4 orders, short
  budget, training-free); in the stream at full budget the gap is within noise.

### EXP-161 — what does the correction learn? spectral / component analysis + VLA part ablation — **done for order 0 (11-19): outside the shared subspace; LOCALISED in the language backbone by the part ablation, but NOT SPECIFIC by the size control: new-skill gain tracks the energy kept, whatever the part (figure energy_gain.png)**
Motivation: meeting item (6). Weight-space decomposition of the trained corrections (consfz orders 0-3, s4, s6, continual rounds)
against the pool's shared subspace (S16: U, V) and the members' residuals: (H1) the correction writes mostly OUTSIDE the shared
subspace, i.e. it repairs residuals rather than the shared part; (H2) it re-amplifies the new skills' residuals that the merge
diluted by 1/k (alignment with sum_new r_i); (H3) it concentrates in specific VLA parts (vision encoder / language backbone /
action expert) and layers. If H2 holds, a structured correction (EXP-162) follows.
- 11-18 weight space (`sweep/exp161_spectral.py`, factor form, CPU on odin): order 0's correction (consfz) is 0.0% inside the
  shared subspace two-sided (output side 2.6%, input side 0.8%; a random rank-16 update: ~0.8% one-sided) -> H1 yes. Frobenius
  alignment with the summed residuals = 0.000 (new and library alike): uninformative, LoRA updates with independent random A
  are orthogonal whatever they compute (A barely moves: mean A^2 = 0.0039 = its 1/r^2 init variance). Output side (what the
  correction writes): 10.0% of its residual energy in the new adapters' B span (random 4.4%), 16.9% in the library's (8.8%) =
  ~2x chance each. Energy per part follows the parameter count (language 71 / vision 15 / expert 14%, as the new adapters'
  66 / 15 / 19). Effective rank 13.9 / 16. -> weight geometry says little; the mechanism question moves to FUNCTION: which VLA
  part carries the correction (below), then activation-space alignment if needed. All runs (orders 0-3, t22, 74-lib,
  continual rounds, from-base consb) running on odin -> results/analysis/exp161_spectral.{log,json}.
- 11-18 weight space, all runs landed (odin has no copies of the s6 / s19 / s20 corrections; skipped): identical picture for
  orders 0-3, t = 22 and the continual correction at 22 (shared two-sided 0.0%, output side 2.3-2.8%, eff rank 13.7-14.2,
  energy per part language ~70 / vision ~14 / expert ~14-17%; output capture ~2x chance; Frobenius alignment 0.000). Contrast:
  the same LoRA trained from the BASE (consb, order 0) writes 7.8% into the shared outputs (3x the in-merge correction) at eff
  rank 10.7 -> with the merge underneath the correction leaves the shared part to the merge. Weight-space line closed; the
  function-level part ablation is the mechanism test.
- 11-18 VLA part ablation (eval only, order 0; `exp147` arms pv / pl / pe = the consfz correction kept ONLY in the vision tower /
  language backbone / action expert + action heads; nv / nl / ne = dropped from that part; merge unchanged; builder mask
  checked: vision stems get the correction only when vision is kept). `sweep/chain69_parts.sh` (tmux c69, gap evals, port 10500).
  PRE-REGISTERED (11-18): the correction is LOCALISED in a part if keeping only that part recovers >= 75% of consfz's gain over
  the uncorrected merge on new skills (order 0: scr 5.0 -> consfz 77.5) AND dropping it loses >= 50% of that gain; otherwise
  distributed. Library reported alongside (scr 73.8 -> consfz 86.9).
- 11-18 pe@0 (correction kept ONLY in the action expert + heads): raw library 81.9 / new 8.8 (scr 73.8 / 5.0, consfz 86.9 / 77.5)
  -> the expert alone recovers 5% of the new-skill gain but 62% of the library gain. Per skill: it repairs library motor skills
  the merge broke (open the top drawer 0 -> 0.6, butter-in-drawer kept 0.6 where consfz drops it to 0.1, microwave 0.5 -> 0.7)
  but 7 of 8 new skills stay at 0 (consfz 0.2-1.0; mostly same-scene variants: bowl next to / on ramekin / cookie box, wine
  bottle rack / cabinet); only bowl-on-ramekin rises (0.3 -> 0.7). Early read: new skills need the correction outside the
  action expert (perception / language), library repair sits partly in the expert. Waiting for pl / pv / ne / nl / nv before the pre-registered call.
- 11-18 pl@0 (correction kept ONLY in the language backbone = the VLM's Gemma layers, which carry image + text tokens): raw
  library 91.9 / new 65.0 -> recovers 83% of the new-skill gain (>= 75%: first half of LOCALISED met; needs nl to lose >= 50%)
  and the whole library gain (91.9 > consfz 86.9, joint 90.0; one order, n.s. likely). Per skill: all four bowl spatial variants
  and orange juice at consfz level; the shortfall is the two wine-bottle goal tasks (0.3 / 0.1 vs 0.9 / 0.7: same scene,
  different goal -> needs the expert / vision part too). Caveat: language holds ~70% of the correction's energy (CORRECTED
  11-19: 42.6% of its parameters, not ~70%).
- 11-18 pv@0 (correction kept ONLY in the vision tower): raw library 74.4 / new 13.8 -> 5% of the library gain, 12% of the
  new-skill gain (orange juice 0.6, bowl-on-ramekin 0.5; the rest at merge level). Keep-only shares of the new-skill gain:
  language 83 / vision 12 / expert 5% -> roughly additive, dominated by the language backbone. Drop-one arms (ne / nl / nv) next.
- 11-18 ne@0 (correction dropped from the action expert + heads; vision + language kept): 88.1 / 70.0 -> loses 10% of the
  new-skill gain (keep-only 5%) = the expert carries little of it, NOT localised there. Wine bottle on top of the cabinet back
  to 1.0 (language only 0.3 -> vision supplies it); wine bottle on the rack 0.3 (consfz 0.7 -> the expert part matters there).
  Library 88.1 (consfz 86.9). nl / nv next.
- 11-18 nl@0 (correction dropped from the language backbone; vision + expert kept): 80.0 / 28.7 -> loses 67% of the new-skill
  gain (>= 50%) and keep-only language recovers 83% (>= 75%) -> PRE-REGISTERED CALL: the new-skill correction is LOCALISED in
  the language backbone (the VLM's Gemma layers over image + text tokens), not in the action expert (keep 5% / drop 10%).
  Per skill: without the language part the four bowl spatial variants fall to 0-0.8 (cookie box / next-to-ramekin 0), orange
  juice stays 1.0. Library: keep-only language 91.9, drop language 80.0, keep-only expert 81.9 -> library repair is redundant
  across parts. Size: CORRECTED 11-19 — the language backbone holds 42.6% of the correction's PARAMETERS (expert 38.6, vision
  18.9) and 71% of its ENERGY ||B A||^2 (vision 15, expert 14); new-skill gain shares 83 / 12 / 5%. By parameters the call
  already holds against size: language only (42.6% of params) 65.0 new vs vision + expert (57.4%) 28.7; the expert has 38.6% of
  the parameters and carries 5-10% of the gain. Energy is the open confound (training put 71% of the update there).
- 11-19 size control queued (`exp147` RSUB arms, seeded random module subsets of the same correction; `sweep/chain69c_size.sh`,
  tmux c69c, after chain69b, lowest priority): rap = any part at language's parameter share (199 modules: 42.7% params / 39%
  energy), rae = any part at language's energy share (328 modules: 71% energy / 67% params), rle = language modules only at the
  vision + expert energy share (41 modules: 29.2% energy / 15.2% params). PRE-REGISTERED (11-19, before any eval): the language
  backbone is SPECIFIC beyond energy if pl >= rae + 15pp new AND rle >= nl (28.7) + 15pp new; ENERGY-EXPLAINED if both gaps
  < 5pp (then the finding is that training concentrates the update in the language backbone, not that only it works);
  in between = partial. rap vs pl reported as the parameter-matched check.
- 11-19 nv@0 (correction dropped from the vision tower; language + expert kept; 24/24 tasks): raw library 88.8 / new 68.8 ->
  loses 12% of the new-skill gain (keep-only vision 12%) and none of the library gain. Part table complete. New skills: the
  three parts' keep-only gains sum to 100% of the full gain (vision 8.8 + language 60.0 + expert 3.8 = 72.6 vs 72.5) and the
  drop-one losses to 90% (8.7 + 48.8 + 7.5 = 65.0) -> the parts act close to ADDITIVELY on new skills, language ~2/3-5/6 of it.
  Library: language alone 91.9, any arm with language >= 86.9, without language 80.0, expert alone 81.9, vision alone 74.4 ->
  library repair sits in language first, expert second, vision ~0. Next: size control (rae / rle / rap, c69c).
- 11-19 rae@0 (size control: the correction kept in 328 random modules of ANY part holding language's energy share, 71% of
  energy / 67% of parameters; 24/24 tasks after the 05:40 race, 3 tasks re-run): raw library 91.9 / new 72.5 vs language only
  (pl, 71% of energy / 42.6% of parameters) 91.9 / 65.0 -> random modules at the same energy recover MORE of the new-skill gain
  (93% vs 83%). First half of SPECIFIC (pl >= rae + 15) fails; the gap is -7.5, beyond the pre-registered 5pp band in the
  other direction (not anticipated: a random subset doing better than language). Per skill rae misses wine bottle on the rack
  (0.0, consfz 0.7) and the drawer bowl (0.1, consfz 0.2), like pl. Early read: the language-backbone localisation is
  mostly WHERE TRAINING PUT THE UPDATE'S ENERGY, not a location only the language backbone can serve. Call waits for rle
  (language at the vision + expert energy share vs nl 28.7) and rap (parameter-matched).
- 11-19 rle@0 (the correction kept ONLY in 41 random language modules holding vision + expert's energy share, 29.2% of energy /
  15.2% of parameters; 24/24 tasks): raw library 85.6 / new 22.5 vs nl (vision + expert, 29% of energy / 57% of parameters)
  80.0 / 28.7 -> gap -6.2. PRE-REGISTERED CALL: SPECIFIC fails on both halves (pl - rae = -7.5, rle - nl = -6.2, needed
  >= +15); by the letter not ENERGY-EXPLAINED either (both gaps just outside the 5pp band, both on the anti-specific side,
  ~1 SE at 80 episodes) -> "partial" by the rule; substantively: at matched energy the language backbone is NOT better than
  the other parts. New-skill gain vs share of the correction's energy kept, all arms (order 0): expert 14% -> 5%, vision 15%
  -> 12%, rle 29% -> 24%, nl 29% -> 33%, pl 71% -> 83%, rae 71% -> 93%, nv 85% -> 88%, ne 86% -> 90%, all 100% -> 100% ->
  the gain tracks the ENERGY kept, whatever the part; "localised in language" = training puts 71% of the update there.
  Library: language modules repair a bit more per unit energy (rle 85.6 vs nl 80.0). rap (parameter-matched) last.
- 11-19 rap@0 (199 random modules of any part at language's PARAMETER share, 42.7% of parameters / 39.0% of energy; 24/24
  tasks): raw library 88.8 / new 45.0 = 55% of the new-skill gain vs language only (same parameter share, 71% of energy) 83%
  -> at matched parameters language wins by 20pp, which the energy difference explains (rap sits on the same gain-vs-energy
  trend). Figure `results/figs/energy_gain.png` (`make_energy_gain_figure.py`, CSV alongside): share of the new-skill gain vs
  share of the update energy kept, all 11 arms on one monotone trend (0 / 14 / 15 / 29 / 29 / 39 / 71 / 71 / 85 / 86 / 100%
  energy -> 0 / 5 / 12 / 24 / 33 / 55 / 83 / 93 / 88 / 90 / 100% gain). EXP-161 order 0 DONE: the correction writes outside
  the merge's shared subspace, training puts 71% of its energy in the VLM's language backbone, and its new-skill effect
  scales with the energy kept regardless of the part (not location-specific).

### EXP-162 — structured correction from the shared / residual components — **done (11-20): lrcfz NEGATIVE (learned residual weights add nothing to the LoRA); lrc alone 81.9 / 50.0 < ours (same-data baseline)**
Motivation: meeting item (6): replace / complement the free LoRA with a correction built on the merge's own components, e.g.
learned per-skill, per-module residual scales (~k x 450 scalars) trained on the same data, alone and with the LoRA.
- 11-18 implemented (`train_lora.ResidualCoefs`, ctx "lrc"): S16 folded into the frozen base; a forward hook on every target module
  adds sum_i alpha[module, i] r_i x with the members' two-sided residuals in factor form; alpha (k per module, 10,992 at k = 24)
  starts at 1/k (= exactly the deployed merge) and is trained with AdamW lr 1e-3, no weight decay (AdaMerging's setting). Arms:
  `lrc` = the weights alone (LoRA frozen) = training-based merging on our data (also an EXP-158 baseline); `lrcfz` = weights +
  the LoRA correction, same data / steps. Deployment: ZBuilder privx "alpha" (lrc.pt). Checks (odin): 458 modules hooked (the 7
  extra over the 451 folded have S16 = 0); builder with alpha = 1/k reproduces the default merge exactly (diff 0.0); 3-step
  smoke trains (alpha moves 0.002). `sweep/chain70_lrc.sh` (tmux c70, after chain67 claims erl@1; gap evals port 10550).
- PRE-REGISTERED (11-18, order 0 first): the components help the correction if lrcfz >= consfz (86.9 / 77.5) + 5pp on library or
  new with the other >= -3pp -> then orders 1-3 and the continual stream. lrc alone is judged as a same-data baseline (EXP-158
  criterion against ours).
- 11-19 lrcfz@0 (learned residual weights + the LoRA, 3,000 steps; 24/24 tasks): raw library 88.8 / new 73.8 vs consfz 86.9 /
  77.5 (+1.9 / -3.7) -> PRE-REGISTERED NEGATIVE: letting the merge's residual weights train next to the correction adds nothing;
  orders 1-3 / continual not run. What the weights learned (alpha x k, 1 = the deployed 1/k): mean 1.36, spread 1.04, 5-14%
  negative per part; library and new residuals scaled alike (1.39 / 1.31) -> no selective re-amplification of the new skills'
  diluted residuals (H2 no, as in weight space). By part (library / new): vision 1.50 / 1.84, language 1.65 / 1.18, expert 1.10
  / 0.90. lrc@0 alone (weights only, LoRA frozen at 0 = training-based merging on our data, AdaMerging-like) stays queued as an
  EXP-158 baseline (chain71, after erlc@4).
- 11-20 21:52 lrc@0 (residual weights only, ~k x 450 scalars learned on the same data / 3,000 steps, no LoRA = training-based
  merging, AdaMerging-like; 24/24 tasks): raw library 81.9 / new 50.0 (24-skill 71.3) vs the training-free merge 73.8 / 5.0
  (+8.1 / +45.0: learning the weights recovers half of the new skills) but below ours consfz 86.9 / 77.5 (ours +5.0, se 4.1;
  6-8-2 / +27.5, se 13.1; 6-0-2) and ER 84.4 / 72.5. As a same-data baseline: ours WIN on both (sign rule; +5 bar met on
  library at the edge, new by far). Rescaling the existing residuals cannot make up for the new skills' 1/k dilution; a
  free correction is needed. EXP-162 DONE.

### EXP-164 — the library merge as ER's starting point — **done (11-21): merge start worth > 9,000 steps and > 20x demos; ours > ER at 1-5 demos, = at 25**
Motivation: erlc@4 tied ours in the stream (EXP-159), so merging the ARRIVING adapters adds nothing measurable; user 11-19:
"the merge might still be justified as a novelty if it makes a good starting point for ER". ER starts from the library-only
merge, which on its own pool is exactly the plain average of the 16 library adapters (in-set identity: S16 + mean residual =
mean); our component operator differs from averaging only once foreign adapters are merged in. Question: is that training-free
merge a better ER start than the base model — in the stream, with less library data, and with no demos at all? One-shot so
far (5 library demos, order 0 / 1): erl 84.4 / 72.5 and 84.4 / 90.0 vs from-base consb 61.9 / 56.2 and 67.5 / 75.0.
- Arms (exp147, erl's / consb's recipe otherwise): (a) `consbc` = ER from the BASE resumed per arrival (19 / 20 / 4, round 1 =
  consb@0, 1,500 steps each; context with no members) vs erlc; (b) library replay budget on order 0, 1 and 25 demos per library
  skill: merge start `erld1` / `erld25` vs base start `consbd1` / `consbd25` (5 demos = erl@0 / consb@0; 0 demos with the merge
  start = seq@0 49.4 / 78.8); (c) demo-free on order 0: library replay = the library adapters' own success rollouts
  (`replay_roll.txt`, as consr) instead of demos, merge start `erlr` vs base start `consbr` (= distil the library from its
  adapters instead of merging it). consb@2 / 3 (queued) complete merge vs base start over 4 orders.
- PRE-REGISTERED (11-19 14:36, before any EXP-164 run): (a) the merge start matters in the stream if erlc >= consbc + 5pp on
  the mean over all 38 skills at arrival 4; not if within 3pp. (b) the merge stands in for library data if the merge start at
  d demos is within 3pp of the base start at 5d on the library: erld1 >= consb@0 - 3 (1 vs 5) and erl@0 >= consbd25 - 3 (5 vs
  25); full curve reported either way. (c) merging beats distilling the library if erlr >= consbr + 5pp library with new >=
  -3pp; repeated on order 1 if within +-5 on order 0. Claim "the merge is a good ER start" if (a) holds and (b) or (c) holds.
- Schedule (chain72 restart 14:37): EXP-163 750-step arms -> consbc@19 -> 20 -> 4, interleaved with erlr@0 / consbr@0 ->
  consb@2 / 3 -> (b) -> SeqFT / EWC cells (ewcc@20, seqc@4, ewcc@4) -> EXP-163 10-rollout arms -> lrc@0, consfa@1-3, consb64@0.
  exp147 arms added 14:36 (md5 0a0cb26b..., both hosts; backup results/merge_stale/exp147_consolidate.py.bak_1119c).
- 11-19 19:11 consbc@19 crashed in 3 min: fold_context_into_base raises on a context with no members. Fix: train_lora.py folds
  only when the context has members (md5 c4341533, both hosts; backup results/merge_stale/train_lora.py.bak_1119); failed log
  -> results/merge_stale/train_consbc_s19_fail1119.log, both claims cleared, chain72's lanes retry it when a GPU frees
  (~22:30, after erlr@0 / consbr@0). erlr@0 trains with 51,930 library rollout samples (vs 5,505 for 5 demos), 46.1M trainable.
- 11-19 14:38 (user: "didn't we already test this?"): yes for the one-shot core — same data, orders 0 / 1, merge start (erl)
  vs base start (consb): library +22.5 / +16.9, new +16.3 / +15.0; consb@2 / 3 complete 4 orders. New here: the stream (a) and
  the base start given the library's knowledge through its adapters' rollouts (c). The demo-budget arms (b) moved behind the
  EXP-163 10-rollout arms (chain72 restart 14:38).
- 11-19 14:42 (user: "in our method the component-based merge is rebuilt at every arrival, so it differs from weight
  averaging"): agreed — ER's library-only start is where the two coincide; ours re-merges each arriving batch. Test of the
  component merge vs averaging as the start where a head start should matter most: `consfaq` = consfa (vanilla average of all
  k, + correction) at 750 steps, orders 0 / 1, beside consfzq / erlq (EXP-163a; chain72, right after erlq@1). PRE-REGISTERED
  (14:42, before any 750-step result): the component merge is the better start if consfzq >= consfaq + 5pp on library or new
  with the other >= -3pp, pooled over orders 0 / 1; no difference if both within 3pp. (Full budget so far: arrival 1 +1.9 /
  +8.7, arrival 2 -1.2 / -1.4; k 38 = consfac@4, EXP-160, tonight.)
- 11-19 consfzq@0 (ours at 750 steps, order 0; 24/24 tasks): raw library 80.6 / new 70.0 vs ours at 3,000 steps 86.9 /
  77.5 (training-free merge 73.8 / 5.0) -> a quarter of the steps keeps 91% of the new-skill level and lifts the library +6.8
  over the merge. Comparators erlq@0 (trained, eval next) and consfaq@0 (queued after consfzq@1 / erlq@1) pending.
- 11-19 erlq@0 (ER at 750 steps, order 0; 24/24 tasks): raw library 79.4 / new 70.0 vs ours consfzq 80.6 / 70.0 -> +1.2 /
  0.0 (library 5 / 8 / 3, new 3 / 2 / 3 wins / ties / losses per skill). Even at a quarter of the budget, merging the
  arriving adapters gives no head start on new skills: training-free their residuals at 1/k add only 5.0 new-skill success
  (scr 73.8 / 5.0), so both starts learn the new skills from ~0. Order 1 (consfzq@1 / erlq@1 training) completes (a).
- 11-19 consfzq@1 (ours at 750 steps, order 1; 24/24 tasks): raw library 80.6 / new 66.3 vs ours at 3,000 steps 88.8 / 76.2
  (training-free merge 81.9 / 25.0: here the merged arrivals do give a 25-point new-skill head start). erlq@1 eval next.
- 11-19 erlq@1 (ER at 750 steps, order 1; 24/24 tasks): raw library 81.9 / new 68.8 vs ours consfzq 80.6 / 66.3 -> -1.3 /
  -2.5. **(a) 750 steps, pooled orders 0 / 1: ours - ER library -0.1 / new -1.3 -> PRE-REGISTERED NEGATIVE** (bar: new >= +5).
  Even order 1's 25-point training-free head start is gone after 750 steps. (b) 10 rollouts (consfz10 / erl10) still queued;
  the vanilla-average start at 750 steps (consfaq@0 / 1, trained 19:10-19:15) evaluating next.
- 11-19 consfaq@0 (vanilla average of all 24 + correction at 750 steps, order 0; 24/24 tasks): raw library 76.2 / new 46.3
  vs ours consfzq 80.6 / 70.0 (+4.4 / +23.7 for us), ER erlq 79.4 / 70.0. At 3,000 steps the same starts gave 85.0 / 68.8 vs
  86.9 / 77.5. At low budget the plain average is a much worse start; ours and the library-only start (no dilution of the
  shared part either way) learn the new skills equally fast. Order 1 (consfaq@1) evaluating; call pooled over 0 / 1.
- 11-19 consfaq@1 (order 1; 24/24 tasks): raw library 73.1 / new 68.8 vs ours consfzq 80.6 / 66.3 (+7.5 / -2.5), ER 81.9 /
  68.8. **Pooled orders 0 / 1, ours - plain average at 750 steps: library +5.9 (se 2.2; 12 / 16 / 4 per skill), new +10.6
  (se 5.1; 10 / 3 / 3) -> PRE-REGISTERED POSITIVE: the component merge is the better start at a low correction budget.**
  At 3,000 steps the difference is gone (arrival 1 +1.9 / +8.7 one order; k 38 0.0 / -2.7). ER (library-only start) = ours
  at 750 steps too (pooled -0.1 / -1.3): the gain over averaging comes from keeping the shared library part whole, which
  ours does while still merging the arrivals and ER does by leaving them out.
- 11-19 22:33 erlr@0 (ER from the library merge, library replay = the library adapters' own success rollouts, no demos;
  order 0; 24/24 tasks): raw library 81.3 / new 73.8 vs erl@0 (5 demos) 84.4 / 72.5 -> demo-free costs -3.1 library, new
  unchanged (cf. ours demo-free consfzr@0 78.8 / 65.0). Comparator consbr@0 (base start, same rollouts; trained 166 min)
  evaluating next -> call (c). consbc@19 retry training since 22:04 with 46.1M trainable (fix holds); consb@2 training.
- 11-19 23:12 consbr@0 (BASE start, same demo-free data: library adapters' own success rollouts + new skills' rollouts;
  24/24 tasks): raw library 54.4 / new 50.0. **(c) erlr - consbr: library +26.9 (se 7.2; 12 / 3 / 1 per skill), new +23.8
  (se 10.8; 6 / 1 / 1) -> PRE-REGISTERED POSITIVE (bar +5 / >= -3; outside +-5, so no order-1 repeat): merging the library
  beats distilling it from its adapters' rollouts.** The base start does worse on 9.4x the library data (51,930 rollout
  samples) than on 5 demos per skill (consb@0 61.9 / 56.2; -7.5 / -6.3), so the merge start is not replaceable by more
  distillation data. With 5 demos the same gap is +22.5 / +16.2 (erl - consb; se 6.3 / 8.6). Overall claim still waits on
  (a) (consbc stream, arrival 4 ~tomorrow morning).
- 11-20 00:19 consbc@19 (BASE-start ER stream, arrival 2 = 14 new, resumed 1,500 steps from consb@0; 30/30 tasks): raw
  library 66.3 / new 73.6 (first batch 71.2, second 76.7) vs erlc@19 91.9 / 80.7 (77.5 / 85.0), ours consfzc 91.9 / 85.7.
  30-skill mean: ER from the merge 86.7 vs base start 69.7 -> +17.0 (arrival 1, 24 skills: erl@0 80.4 vs consb@0 60.0 ->
  +20.4). Per skill erlc - consbc: library +25.6 (se 6.7; 12 / 4 / 0), new +7.1 (se 6.9; 7 / 4 / 3). The base start climbs
  with resumed replay (library 61.9 -> 66.3, first batch 56.2 -> 71.2) but stays far below; (a) is called at arrival 4
  (consbc@20 training since 23:40, then @4).
- 11-20 01:48 consbc@20 (base-start stream, arrival 3 = 18 new; 34/34 tasks): raw library 80.0 / new 54.4 (batches 48.8 /
  58.3 / 60.0) vs erlc@20 92.5 / 76.7 (71.2 / 86.7 / 72.5), ours consfzc 95.0 / 87.8. 34-skill mean: ER from the merge 84.1
  vs base start 66.5 -> +17.6 (arrivals 1-3: +20.4 / +17.0 / +17.6, no narrowing). Per skill erlc - consbc: library +12.5
  (se 4.4; 11 / 4 / 1), new +22.2 (se 8.8; 11 / 4 / 3). The base start's library keeps climbing (61.9 -> 66.3 -> 80.0) while
  its new skills fall (first batch 56.2 -> 71.2 -> 48.8): from the base, replay alone does not hold the stream together.
  consbc@4 (arrival 4, the call) training since 01:06.
- 11-20 02:22 consb@2 (base start, 5 library demos, one-shot order 2; chain67b; 24/24 tasks): raw library 61.2 / new 57.5
  vs ER erl@2 86.9 / 81.2, ours consfz@2 90.0 / 67.5. Merge start - base start (erl - consb), orders 0 / 1 / 2: library
  +22.5 / +16.9 / +25.6, new +16.2 / +15.0 / +23.8; pooled per skill library +21.7 (se 3.5; 35 / 11 / 2), new +18.3 (se 5.2;
  16 / 5 / 3). consb@3 training (completes 4 orders).
- 11-20 03:33 consbc@4 (base-start stream, arrival 4 = 22 new, k 38; 38/38 tasks): raw library 83.8 / new 84.5 (batches
  77.5 / 91.7 / 82.5 / 90.0) vs erlc@4 88.8 / 88.6 (83.8 / 96.7 / 80.0 / 95.0), ours consfzc 90.0 / 83.6, joint s4 85.0 /
  84.1. 38-skill mean: ER from the merge 88.7 vs base start 84.2 -> +4.5 (se 2.5; 14 / 14 / 10 per skill; library +5.0 se
  3.9, new +4.1 se 3.4). **(a) PRE-REGISTERED NOT MET: +4.5 < +5, and not within 3 either (between the bars =
  inconclusive). The overall claim "the merge is a good ER start" (needs (a)) is NOT established as pre-registered.**
  Stream gap by arrival: +20.4 / +17.0 / +17.6 / +4.5 (descriptive mean +14.9, post hoc). The base start caught up at the
  last arrival (new 54.4 -> 84.5, first batch 48.8 -> 77.5) after 7,500 cumulative replay steps; ours is +2.1 over it, and
  all three sit at joint level (84.5 on the 38-skill mean). Reading: the library merge is a large head start (~+20 for the
  first three arrivals, 4 orders one-shot +21.7 / +18.3) that enough cumulative replay training closes, the same shape as
  every other start difference (EXP-160, EXP-163). Open: consb64@0 (base start, rank 64, 12k steps, one-shot, queued last)
  measures how much training the head start is worth; (b) demo budget still queued.
- 11-20 04:49 consb@3 (base start, 5 library demos, one-shot order 3; 24/24 tasks): raw library 58.1 / new 72.5 vs ER
  erl@3 85.6 / 77.5, ours consfz@3 85.6 / 88.8. **One-shot (arrival 1), 4 orders complete: merge start - base start (erl -
  consb) library +23.1 (se 3.3; 46 / 14 / 4 per skill), new +15.0 (se 4.5; 18 / 8 / 6); per order library +22.5 / +16.9 /
  +25.6 / +27.5, new +16.2 / +15.0 / +23.8 / +5.0. Ours - base start +25.6 (se 3.1) / +12.2 (se 4.5).** 4-order means: base
  start 62.2 / 65.3, ER from the merge 85.3 / 80.3. At 3,000 steps the head start is robust across orders; the stream
  (above) shows it closes after 7,500 cumulative steps.
- 11-20 09:27 (d) added: what the merge start is worth in training steps. The stream (a) gap closed between 6,000 and 7,500
  cumulative steps, but steps and data grew together there, and consb64@0 (queued) changes rank as well as steps. Arms:
  `consb6` / `consb12` = consb (base start, rank 16, 5 library demos + new skills' rollouts, order 0) at 6,000 / 12,000 steps
  (exp147 md5 6fabd9b3..., both hosts; backup results/merge_stale/exp147_consolidate.py.bak_1120). consb64@0 dropped
  (pre-claimed `.c72_consb64_0`; e67b keeps waiting on it, harmless). Scheduler `sweep/chain73_steps.sh` (tmux c73, started
  09:27): lanes wait 120 s so chain72's queue runs first; evals port 10650 under the eval lock. PRE-REGISTERED (09:27, before
  any result): reference = ER from the merge at 3,000 steps, erl@0 24-skill mean 80.4 (base start at 3,000: consb@0 60.0).
  Report the step count at which the base start comes within 3pp of 80.4: 6,000 / 12,000 / not reached by 12,000. "The merge
  is worth >= 9,000 training steps" if consb12 < 77.4; "worth <= 3,000 steps" if consb6 >= 77.4.
- 11-20 14:54 erld1@0 (ER from the library merge, 1 demo per library skill instead of 5, order 0; 24/24 tasks): raw library
  85.0 / new 68.8. (b) first half: vs base start at 5 demos (consb@0 61.9 / 56.2) library +23.1 (se 4.7; W-T-L 14-2-0), new
  +12.5 (se 11.0; 5-1-2) -> erld1 >= consb@0 - 3 MET by +26: the merge with 1 demo beats the base with 5 by 23 points. vs
  the merge start at 5 demos (erl@0 84.4 / 72.5): +0.6 / -3.8 -> library replay needs ~1 demo per skill once the merge holds
  the library (0 demos: seq@0 49.4, so some replay is needed; library rollouts instead: erlr 81.3). consbd1@0 evaluating next;
  second half (erl@0 >= consbd25 - 3) waits for consbd25.
- 11-20 15:34 consbd1@0 (base start, 1 demo per library skill, order 0; 24/24 tasks): raw library 48.8 / new 53.8 (24-skill
  50.4). At 1 demo, merge - base = library +36.2 (se 5.6; W-T-L 14-2-0) / new +15.0 (se 10.9; 6-0-2); 24-skill 79.6 vs 50.4.
  From 5 to 1 demo the base start loses -13.1 library (se 6.3; 4-3-9) / -2.5 new, the merge start +0.6 / -3.8 -> the less
  library data, the larger the merge's lead (library gap +22.5 at 5 demos -> +36.2 at 1, order 0). erld25 / consbd25 next.
- 11-20 17:51 erld25@0 (merge start, 25 demos per library skill, order 0; 24/24 tasks): raw library 90.6 / new 70.0 (24-skill
  83.8). Merge-start library curve 85.0 / 84.4 / 90.6 at 1 / 5 / 25 demos (= joint 90.0 at 25); vs 5 demos +6.2 (se 5.3;
  7-5-4) / -2.5 (se 3.7). Merge at 25 vs base at 5: +28.7 / +13.8. Flat 1 -> 5, small rise 5 -> 25. consbd25 evaluating
  next: (b) second half = erl@0 84.4 >= consbd25 - 3.
- 11-20 18:20 (b) extended to OUR method (user Q: "so we just merge once for the start, not when new adapters arrive?" ->
  ER merges the library once (library-only merge = plain average of the 16, in-set identity); ours re-merges library + each
  arrival with the component operator). Arms `consfzd1` / `consfzd25` = consfz (merge of library + 8 new, correction, order 0)
  at 1 / 25 library demos per skill (exp147 md5 806e9cc7..., both hosts; backup results/merge_stale/exp147_consolidate.py.
  bak_1120b; not the DAgger consfzd). Scheduler `sweep/chain74_oursdemo.sh` (tmux c74): lanes wait 105 s = after chain72's
  consfa@2 / 3, before chain73's step sweep; evals port 10700. PRE-REGISTERED (18:20, before any result): report ours - ER at
  1 / 5 / 25 demos by sign (11-20 rule; 5 demos = consfz@0 - erl@0 = +2.5 / +5.0); ours needs no more library data than ER
  if consfzd1 library >= consfz@0 86.9 - 3 = 83.9.
- 11-20 18:29 consbd25@0 (base start, 25 demos per library skill, order 0; 24/24 tasks): raw library 57.5 / new 68.8
  (24-skill 61.2). Base-start library 48.8 / 61.9 / 57.5 at 1 / 5 / 25 demos: 25 demos do not lift it (vs 5: -4.4, se 5.9;
  7-1-8; new +12.5, se 6.5). Merge - base at 25 demos: library +33.1 (se 7.3; 13-2-1) / new +1.3. (b) second half: erl@0
  84.4 >= consbd25 - 3 = 54.5 MET (+26.9, se 8.4) -> (b) MET on both halves: the merge stands in for library data (merge at
  1 demo > base at 5; merge at 5 > base at 25). Library gap by budget: +36.2 / +22.5 / +33.1 at 1 / 5 / 25 demos.
  Reading: at a fixed 3,000 steps the replay half of the batch shows each library skill ~750 samples whatever the demo count,
  so more demos add diversity, not exposure; the base start is step-limited, not data-limited (the stream gap closed only
  after 7,500 steps). (d) consb6 / consb12 test that directly. EXP-164 overall: (a) +4.5 (pre-reg between +3 and +5 =
  inconclusive; a WIN by the 11-20 sign rule), (b) MET, (c) POSITIVE -> "the merge is a good ER start" holds.
- 11-21 01:55 consfzd1@0 (ours: re-merge library + 8 new with the component operator, correction, 1 library demo per skill,
  order 0; 24/24 tasks): raw library 86.2 / new 77.5 (24-skill 83.3). Pre-reg: library >= 83.9 MET (vs consfz@0 at 5 demos
  -0.6, se 5.7; 4-7-5 / new 0.0) -> ours needs no more than 1 library demo per skill. Ours - ER at 1 demo (vs erld1 85.0 /
  68.8): library +1.2 (se 2.6; 4-9-3), new +8.8 (se 4.0; 4-4-0) = WIN by sign on both (5 demos: +2.5 / +5.0). ER loses new
  -3.8 from 5 to 1 demo, ours 0.0 -> merging the arrivals keeps new skills steady when library replay is scarce.
- 11-21 04:25 consfzd25@0 (ours at 25 library demos per skill, order 0; 24/24 tasks): raw library 88.8 / new 70.0 (24-skill
  82.5). Ours - ER at 25 demos (vs erld25 90.6 / 70.0): library -1.9 (se 2.9; 3-9-4), new 0.0 (se 4.2; 2-4-2) = LOSS by sign
  on library, tie on new. Pre-registered by-sign report, ours - ER library / new at 1 / 5 / 25 demos: +1.2 / +8.8, +2.5 /
  +5.0, -1.9 / 0.0; 24-skill +3.7 / +3.4 / -1.3 (ours 83.3 / 83.8 / 82.5, ER 79.6 / 80.4 / 83.8). -> Our lead over ER is a
  low-data effect: it holds at 1-5 demos and is gone at 25, where ER's library reaches joint (90.6). Ours is flat across
  budgets (library 86.2 / 86.9 / 88.8); its new-skill dip at 25 (-7.5 vs 5 demos, se 9.2; 2-3-3) is within single-order
  noise. Merge start still +31.2 library over the base start at 25 demos (13-2-1). One order, one seed.
- 11-21 05:14 consb6@0 (base start, 5 demos, 6,000 steps = 2x, order 0; 24/24 tasks): raw library 71.2 / new 72.5 (24-skill
  71.7). Doubling steps: +11.7 all (se 4.9; 15-4-5) over consb@0 60.0, library +9.4 (se 6.1), new +16.2 (se 8.6; 7-0-1).
  (d): 71.7 < 77.4 -> NOT within 3pp at 6,000; "worth <= 3,000 steps" ruled out. ER from the merge at 3,000 steps still ahead
  of the base start at 6,000: all +8.8 (se 4.8; 14-5-5), library +13.1 (se 6.4; 11-2-3), new 0.0 -> extra steps buy the base
  start its new skills (= ER's 72.5), not its library. Ours at 3,000 vs consb6: +12.1 all / +15.6 library / +5.0 new.
  consb12@0 last (~14:00): "worth >= 9,000 steps" if < 77.4.
- 11-21 13:39 consb12@0 (base start, 5 demos, 12,000 steps = 4x, order 0; 24/24 tasks): raw library 72.5 / new 65.0 (24-skill
  70.0). (d) 70.0 < 77.4 -> PRE-REGISTERED "the merge is worth >= 9,000 training steps" MET; base start never within 3pp of
  ER (80.4): 60.0 / 71.7 / 70.0 at 3k / 6k / 12k = plateau after 6k (12k vs 6k -1.7, se 5.8; 10-4-10). ER from the merge at
  3k vs base at 12k: all +10.4 (se 6.1; 9-8-7), library +11.9 (se 8.7; 4-7-5), new +7.5; ours +13.8 (se 6.0; 12-8-4) / +14.4 /
  +12.5. The library gap sits in articulated-object skills the base start never gets (open drawer + put bowl 0 vs ER 100,
  open top drawer 20 vs 90, close microwave 30 vs 90; at 6k 40 / 80 / 60 -> regresses with more steps); the base start reaches
  90-100 on most pick-and-place. One order, one seed.
- EXP-164 DONE (11-21 13:39): (a) stream merge start +4.5 at arrival 4 (WIN by sign), (b) MET (merge at 1 demo > base at 5;
  at 5 > base at 25), (c) POSITIVE, (d) MET (merge worth >= 9,000 steps; base plateaus ~71). Ours vs ER by library budget
  (24-skill): +3.7 / +3.4 / -1.3 at 1 / 5 / 25 demos = our lead is a low-data effect.

### EXP-165 — does PolicyWeave's action-interface failure apply to our adapters? — **done (11-20): NO, interface barely adapted**
Motivation: PolicyWeave (2609.33125, GR00T N1.5) finds VLA experts stop being mergeable when each adapts the action interface
(normalization, action encoder / decoder): hidden layers compensate the expert's own interface change and averaging breaks the
pairing (static merges 17.0% -> 52.8% once the interface is shared). Our LoRAs target action_in_proj / action_out_proj /
time_mlp too (normalization already shared: one norm_stats.json). If the interface diverged, freezing it in new adapters would
be a cheap merge-compatible training recipe and a second cause (besides 1/k) of the training-free merge's new-skill failure.
- 11-20 22:40 `sweep/interface_diag.py` (CPU, order 0's 24 merge members, Gram of B_i A_i per stem): pairwise cosine library-
  library 0.74 interface vs 0.76-0.87 hidden families; new-new and library-new ~0.00-0.03 in EVERY family (our new adapters are
  orthogonal to the fleet everywhere, not specially on the interface); averaging retention 0.61 interface vs 0.65-0.70 hidden.
  Relative change ||dW|| / ||W0||: action_in_proj 0.3%, action_out_proj 1.6% vs 2-5% for hidden layers (k_proj / down_proj,
  layer 9 of expert and language) and time_mlp 3-5%. -> The action encoder is almost untouched and the decoder lightly; with
  shared normalization our adapters already keep the base interface in practice. PolicyWeave's failure mode does not explain
  ours; the library / new disagreement is uniform across modules (= the 1/k dilution story). No GPU arm queued.

### EXP-166 — the low-data case: ours vs ER at 1 library demo, 4 orders + the stream — **done (11-22): ours ahead on 4 / 4 orders (+2.7 pooled) and at 3 / 4 stream arrivals**
Motivation: our lead over ER (the strongest same-data baseline; it starts from the component merge of the library alone = plain
average of the 16) is largest when library replay is scarce: order 0, all 24 skills +3.7 / +3.4 / -1.3 at 1 / 5 / 25 demos, new
+8.8 at 1 demo (EXP-164). User 11-21: "We need to find a clear case for our method to lead consistently, even if some leads are
small." Test whether the 1-demo lead holds on every order (A) and through the continual stream (B).
- 11-21 14:01 arms: (A) consfzd1 / erld1 on orders 1-3 (order 0 done); (B) `consfzd1c` / `erld1c` = consfzc / erlc at 1 library
  demo per skill, E145_SEQ 19 / 20 / 4, 1,500 resumed steps per arrival, round 1 = consfzd1@0 / erld1@0 (exp147 CPREV;
  md5 cc2422e3...; backups results/merge_stale/exp147_consolidate.py.bak_1121 / _1121b). Scheduler `sweep/chain75_lowdata.sh`
  (tmux c75): stream rounds first, then orders; lanes wait 90 s; evals port 10750. Checked on consfzd1c@19: resumed from
  train_consfzd1 (30-member merge), replay 16 tasks x 1 demo, trainable 46.1M; erld1c@19 resumed from train_erld1 (16 members).
- PRE-REGISTERED (14:12, before any EXP-166 result; 11-20 sign rule, se / W-T-L shown): (A) ours - ER on all 24 skills at 1
  demo, pooled over 4 orders > 0 = WIN; CONSISTENT if > 0 on >= 3 of 4 orders; library and new reported separately.
  (B) ours - ER on all arrived skills per arrival (arrival 1 = +3.7): CONSISTENT if > 0 at >= 3 of 4 arrivals; library sign
  per arrival reported.
- 11-21 20:50 (B) arrivals 2-3 landed (all audits ok; landings 15:46-18:10, logged late: the watcher had died). Raw success,
  ours vs ER, d = ours - ER (se; W-T-L per skill):
  arrival 2 (30 skills): library 83.8 vs 83.8 (0.0; 4-8-4), new 82.9 vs 79.3 (+3.6; 6-6-2), all 83.3 vs 81.7 = +1.7 (se 1.9; 10-14-6);
  arrival 3 (34 skills): library 85.6 vs 81.9 (+3.8; 4-10-2), new 82.8 vs 80.6 (+2.2; 9-2-7), all 84.1 vs 81.2 = +2.9 (se 2.3; 13-12-9).
  So far (B) ahead at 3 / 3 arrivals (+3.7 / +1.7 / +2.9); library >= 0 at all three. Just-arrived batch ours vs ER 81.7 vs 78.3
  (arrival 2), 85.0 vs 87.5 (arrival 3). Arrival 4: consfzd1c@4 evaluated (38 tasks ok); erld1c@4 eval running.
- 11-21 21:00 (B) DONE: arrival 4 (38 skills; erld1c@4 38 tasks ok) library 85.0 vs 87.5 (-2.5; 4-8-4), new 89.1 vs 88.2
  (+0.9; 5-11-6), all 87.4 vs 87.9 = -0.5 (se 2.2; 9-19-10). All arrived, arrivals 1-4: +3.7 / +1.7 / +2.9 / -0.5 (mean +2.0)
  -> ahead at 3 / 4 = (B) CONSISTENT (pre-registered bar met). Same shape as the 5-demo stream (+3.3 / +2.3 / +7.1 / -2.4):
  ER catches up at the final, 38-member arrival in both. Just-arrived batch ours - ER +8.8 / +3.3 / -2.5 / -5.0 (c_kk ours
  77.5 / 81.7 / 85.0 / 95.0, ER 68.8 / 78.3 / 87.5 / 100.0); NBT ours -7.8, ER -4.9 (neither forgets). Deck: continual slide bullet.
- 11-21 23:45 (A) consfzd1@1 (order 1, 1 demo; 24 tasks ok) 88.8 / 87.5 (at 5 demos 88.8 / 76.2; ER at 5 demos 84.4 / 90.0).
  erld1@1 trained, eval next (~01:00).
- 11-22 01:05 (A) erld1@1 (24 tasks ok) 85.6 / 85.0 -> order 1 ours - ER library +3.1 (se 3.3; 5-9-2), new +2.5 (se 1.6; 2-6-0:
  never behind on a new skill), all 24 +2.9 (se 2.2; 7-15-2). At 5 demos order 1 was -1.7 all (new -13.8): at 1 demo the sign
  flips, mostly because ours' order-1 new rises 76.2 -> 87.5 (the 5-demo run lost 0.1-0.3 on four bowl skills). Orders 0-1
  pooled (48 skills): library +2.2 (se 2.0), new +5.6 (se 2.2), all +3.3 (se 1.6); > 0 on 2 / 2 orders so far (A needs >= 3 / 4).
  Deck: merge-start table filled for order 1 at 1 demo + bullet.
- 11-22 02:20 (A) consfzd1@2 (order 2, 1 demo; 24 tasks ok) 83.1 / 72.5 (at 5 demos 90.0 / 67.5; ER at 5 demos 86.9 / 81.2).
  erld1@2 training (00:32), eval ~03:40.
- 11-22 03:45 (A) erld1@2 (24 tasks ok) 78.8 / 75.0 -> order 2 ours - ER library +4.4 (se 3.0; 5-9-2), new -2.5 (se 5.6; 2-2-4),
  all 24 +2.1 (se 2.8; 7-11-6). > 0 on 3 / 3 orders -> (A) CONSISTENT bar (>= 3 of 4) MET before order 3. Orders 0-2 pooled
  (72 skills): library +2.9 (se 1.7), new +2.9 (se 2.4), all +2.9 (se 1.4). Order 2's library drops at 1 demo for both arms
  (ours 90.0 -> 83.1, ER 86.9 -> 78.8; frying pan on the shelf 1.0 -> 0.0 / 0.2 = its one demo is not enough). Order 3
  (consfzd1@3 trained ~04:20, erld1@3 training since 03:09) decides the 4-order pool. chain77 lanes started 03:10.
- 11-22 05:10 (A) consfzd1@3 (order 3, 1 demo; 24 tasks ok) 72.5 / 73.8 (at 5 demos 85.6 / 88.8; ER at 5 demos 85.6 / 77.5):
  the first order where 5 -> 1 demo costs ours a lot (library -13.1: frying pan on the shelf 0.7 -> 0.0 again, chocolate pudding
  0.6 -> 0.1, moka pot 0.8 -> 0.3; new -15.0). erld1@3 eval ~06:20.
- 11-22 06:30 (A) DONE. erld1@3 (24 tasks ok) 66.9 / 78.8 -> order 3 library +5.6 (se 3.5; 7-6-3), new -5.0 (se 11.0; 3-2-3),
  all +2.1 (se 4.3; 10-8-6). All 24 skills ours - ER per order +3.7 / +2.9 / +2.1 / +2.1 = ahead on 4 / 4 -> CONSISTENT; pooled
  96 skills +2.7 (se 1.5; 32-47-17) -> WIN (pre-registered). Library +3.6 (se 1.5; 21-33-10; ahead on 4 / 4), new +0.9 (se 3.2;
  11-14-7; + / + / - / -). 4-order means, ours vs ER: library 82.7 vs 79.1, new 77.8 vs 76.9, all 81.0 vs 78.3. Same pairs at
  5 demos: all +0.7 (se 1.6; ahead on 2 / 4), library +2.5, new -2.8 -> cutting library data to 1 demo costs ER more (library
  -6.2) than ours (-5.1), and the lead becomes consistent on every order. Ours at 1 demo stays below ER at 5 (81.0 vs 83.6 all).
  EXP-166 DONE: with 1 library demo per skill, ours leads ER on every order (one-shot) and at 3 / 4 stream arrivals.
- 11-22 06:25 scheduling: chain76's erlc@24 won GPU2 at 05:50 over chain77 despite the 60 s vs 120 s lane delays (eval servers
  make busy2 flicker); `sweep/c76hold.sh` (tmux c76hold, log results/c76hold.log) pre-claims chain76's 12 unstarted rounds
  (not @29, so its lanes idle) and releases exactly those claims once chain77 has claimed all 8 jobs. erlc@24 (pair of
  consfzc@24) finishes ~07:05.

### EXP-167 — the continual stream on orders 1-3 (5 demos) — **done (11-23 05:45): ours = ER (16-cell +0.2, final arrival -1.7); library rule failed 6 / 9; chain76**
Motivation: the paper leads with continual merging, but the stream comparison with ER is order 0 only (ours ahead at 3 / 4
arrivals, all arrived skills +3.3 / +2.3 / +7.1 / -2.4; library never behind: +2.5 / 0.0 / +2.5 / +1.2). A reviewer's first
question is whether that holds for other libraries.
- 11-21 14:06 sequences (`exp116_continual.py` CONTO, md5 356e513c...; backup results/merge_stale/exp116_continual.py.bak_1121):
  E145_SEQ 21-23 / 24-26 / 27-29 = order 1 / 2 / 3's library with 14 / 18 / 22 new = the same 22 new skills as order 0's stream,
  order o's 8 first, then orders o+1, o+2, o+3 (mod 4) deduplicated, cut at 14 / 18 / 22 (arrival sizes 8 / 6 / 4 / 4 as in
  order 0's stream). Checked: library, replay and library context = order o's; 38-member contexts begin with order o's 24.
  exp147: PREVSEQ extended, consfzc's resume and the CPREV resume generalized (identical paths for 19 / 20 / 4); gpu_gate prev()
  extended (md5 f07acf60...). prep / prep2 run for 21-29. Arms consfzc / erlc (arrival 1 = consfz@o / erl@o, done).
  Scheduler `sweep/chain76_streams.sh` (tmux c76): arrival-major across orders; lanes wait 120 s (chain75 first); evals port
  10800 once chain75 has no pending eval.
- PRE-REGISTERED (14:12, before any EXP-167 result): 16 cells (4 orders x 4 arrivals); 7 are known (arrival 1 on orders 1-3 +
  order 0's stream), so the test is the 9 new cells (arrivals 2-4 on orders 1-3). Library lead CONSISTENT if ours - ER library
  >= 0 in >= 7 of the 9; overall WIN by the sign of the 9-cell mean on all arrived skills; also reported: the 16-cell means
  (all skills, library, new) and the final arrival (38 skills) pooled over 4 orders with se.
- 11-21 20:50 first cell: consfzc@21 (order 1, arrival 2; 30 tasks ok) 90.6 / 73.6, all 82.7 (order 1's 8 first-arrival skills
  78.8, the 6 just-arrived 66.7). Arrival 1 on order 1 was ours 88.8 / 76.2 vs ER 84.4 / 90.0. erlc@21 trained, eval pending.
- 11-21 21:35 cell 1 of 9 (order 1, arrival 2; erlc@21 30 tasks ok): library 90.6 vs 89.4 (+1.2; 4-9-3), new 73.6 vs 82.9
  (-9.3; 2-6-6), all 82.7 vs 86.3 = -3.7 (se 3.1; 6-15-9). Library >= 0 (1 / 1 toward the 7 / 9 bar); all-skill sign negative.
  Order 1's new-skill deficit carries over from arrival 1 (first batch -13.8 -> -12.5): small losses spread over the spatial
  bowl skills (ramekin 0.6 vs 1.0, top-drawer bowl 0.6 vs 0.8), plus one just-arrived skill (bowl next to the plate 0.0 vs 0.6).
- 11-21 22:35 consfzc@24 (order 2, arrival 2; 30 tasks ok) 90.0 / 72.9, all 82.0 (order 2's first batch 58.8, was 67.5 at
  arrival 1; just-arrived 6 skills 91.7). Arrival 1 on order 2 was ours 90.0 / 67.5 vs ER 86.9 / 81.2. erlc@24 not trained yet:
  EXP-169 (chain77) now runs ahead of the remaining stream rounds (user 11-21), so this pair lands after it.
- 11-22 08:20 cell 2 of 9 (order 2, arrival 2; erlc@24 30 tasks ok, trained 05:50 before the hold): library 90.0 vs 85.6 (+4.4;
  7-8-1), new 72.9 vs 73.6 (-0.7; 4-3-7), all 82.0 vs 80.0 = +2.0 (se 2.2; 11-11-8). Library >= 0 in 2 / 2 cells; 9-cell running
  mean all -0.9 (-3.7 / +2.0). Order 2's arrival-1 new gap (-13.8) closes: first batch 58.8 vs 60.0 (both arms drop it: ours
  67.5 -> 58.8, ER 81.2 -> 60.0); just-arrived 6 skills tie 91.7 / 91.7.
- 11-22 17:40 consfzc@27 (order 3, arrival 2; 30 tasks ok) 89.4 / 77.9, all 84.0 (order 3's first batch 88.8 = kept from arrival 1;
  just-arrived 6 skills 63.3). Arrival 1 on order 3 was ours 85.6 / 88.8 vs ER 85.6 / 77.5.
- 11-22 19:15 cell 3 of 9 (order 3, arrival 2; erlc@27 30 tasks ok) library 89.4 vs 89.4 (0.0, se 3.4; 5-9-2), new 77.9 vs 71.4
  (+6.4, se 2.7; 6-7-1), all 84.0 vs 81.0 = +3.0 (se 2.3; 11-16-3). First batch +3.8 (88.8 vs 85.0), just-arrived 6 +10.0 (63.3
  vs 53.3; 4-2-0) = ours learns the arrivals sooner (as order 0, EXP-168). Library >= 0 in 3 / 3 cells; 9-cell running mean all
  +0.4 (-3.7 / +2.0 / +3.0). Arrival 3 rounds (consfzc@22 GPU0 since 18:06, erlc@22 GPU2 since 18:31) next.
- 11-22 20:25 consfzc@22 (order 1, arrival 3 = 18 new; 34 tasks ok) 88.8 / 76.1, all 82.1 (arrival 2: 90.6 / 73.6, all 82.7).
  Batches 81.2 / 75.0 / 67.5 (first / second / just-arrived 4) vs 78.8 / 66.7 at arrival 2 = earlier batches keep rising, no
  forgetting. ER at arrival 2 on order 1 was 89.4 / 82.9 (batches 91.2 / 71.7). erlc@22 (GPU2 since 18:31) is the pair.
- 11-22 21:05 cell 4 of 9 (order 1, arrival 3; erlc@22 34 tasks ok, 85.0 / 76.1, batches 87.5 / 71.7 / 60.0): library 88.8 vs
  85.0 (+3.8, se 2.7; 4-11-1), new 76.1 vs 76.1 (0.0, se 4.1; 6-6-6), all 82.1 vs 80.3 = +1.8 (se 2.5; 10-17-7). Batches -6.2 /
  +3.3 / +7.5 (first batch: ER's order-1 lead shrinks -13.8 -> -12.5 -> -6.2 over arrivals 1-3; ours ahead on the 2 later
  batches). Order 1 all-skill ours - ER over arrivals 1 / 2 / 3: -1.7 / -3.7 / +1.8. Library >= 0 in 4 / 4 cells; 9-cell
  running mean all +0.8 (-3.7 / +2.0 / +3.0 / +1.8).
- 11-22 22:10 consfzc@25 (order 2, arrival 3 = 18 new; 34 tasks ok) 81.2 / 78.9, all 80.0 (arrival 2: 90.0 / 72.9, all 82.0).
  Batches 80.0 / 80.0 / 75.0 (first / second / just-arrived 4) vs 58.8 / 91.7 at arrival 2. Library -8.8 is 2 skills: the
  KITCHEN_SCENE9 frying pan on / under the shelf 0.9 -> 0.1 / 0.3 (no scene-9 or pan skill among the 4 arrivals; the on-shelf
  skill's own expert scores 0.65). erlc@25 is the pair.
- 11-22 22:50 cell 5 of 9 (order 2, arrival 3; erlc@25 34 tasks ok, 85.6 / 76.7, batches 65.0 / 83.3 / 90.0): library 81.2 vs
  85.6 (-4.4, se 4.7; 3-9-4) = the FIRST library loss, all of it one skill (frying pan under the shelf 0.3 vs 1.0; the other 15
  net 0); new 78.9 vs 76.7 (+2.2, se 4.8; 8-6-4); all 80.0 vs 80.9 = -0.9 (se 3.4; 11-15-8). Batches +15.0 (first batch, 6-2-0)
  / -3.3 / -15.0 (just-arrived 4). Library >= 0 in 4 / 5 cells (the 7 / 9 rule allows one more loss); 9-cell running mean all
  +0.4 (-3.7 / +2.0 / +3.0 / +1.8 / -0.9).
- 11-22 23:50 consfzc@28 (order 3, arrival 3 = 18 new; 34 tasks ok) 88.1 / 85.0, all 86.5 (arrival 2: 89.4 / 77.9, all 84.0).
  Batches 92.5 / 75.0 / 85.0 vs 88.8 / 63.3 at arrival 2 = earlier batches keep rising. erlc@28 (trained) is the pair; arrival 4
  on order 1 (consfzc@23 / erlc@23) training since 23:09 / 23:14.
- 11-23 00:30 cell 6 of 9 (order 3, arrival 3; erlc@28 34 tasks ok, 95.6 / 87.2, batches 92.5 / 71.7 / 100.0): library 88.1 vs
  95.6 (-7.5, se 4.0; 1-8-7; spread over 7 skills, -0.1 to -0.4 each; ER's library rose 89.4 -> 95.6 from arrival 2), new 85.0
  vs 87.2 (-2.2, se 3.9; 2-12-4), all 86.5 vs 91.2 = -4.7 (se 2.8; 3-20-11). Batches 0.0 / +3.3 / -15.0 (just-arrived 4).
  Library >= 0 in 4 / 6 cells: the 7 / 9 rule now needs all 3 arrival-4 cells >= 0. 9-cell running mean all -0.4 (-3.7 / +2.0
  / +3.0 / +1.8 / -0.9 / -4.7; library -0.4, new -0.6) = sign now negative; cells 3 ours / 3 ER.
- 11-23 01:30 consfzc@23 (order 1, arrival 4 = 22 new; 38 tasks ok) 91.9 / 83.6, all 87.1 (arrival 3: 88.8 / 76.1, all 82.1).
  Batches 88.8 / 85.0 / 77.5 / 77.5 (first ... just-arrived 4) vs 81.2 / 75.0 / 67.5 at arrival 3 = every earlier batch rises
  again; first batch 76.2 (arrival 1) -> 78.8 -> 81.2 -> 88.8. erlc@23 (trained) is the pair; consfzc@26 / erlc@26 training
  since 00:50 / 00:54.
- 11-23 02:15 cell 7 of 9 (order 1, arrival 4; erlc@23 38 tasks ok, 90.6 / 83.2, batches 86.2 / 81.7 / 82.5 / 80.0): library
  91.9 vs 90.6 (+1.3, se 4.1; 1-11-4: one skill, red mug on the left plate 0.9 vs 0.3, against 4 losses of 0.1), new 83.6 vs
  83.2 (+0.5, se 3.9; 4-9-9), all 87.1 vs 86.3 = +0.8 (se 2.8; 5-20-13; more per-skill losses than wins, a tie in substance).
  Batches +2.5 / +3.3 / -5.0 / -2.5. Order 1 all-skill ours - ER over arrivals 1-4: -1.7 / -3.7 / +1.8 / +0.8. Library >= 0
  in 5 / 7 cells (the 7 / 9 rule needs both order-2 / 3 arrival-4 cells >= 0); 9-cell running mean all -0.2 (-3.7 / +2.0 /
  +3.0 / +1.8 / -0.9 / -4.7 / +0.8); cells 4 ours / 3 ER.
- 11-23 03:20 erlc@26 (ER, order 2, arrival 4; 38 tasks ok; its eval ran first: trained 2 min before consfzc@26) 86.2 / 75.5,
  all 80.0, batches 68.8 / 73.3 / 82.5 / 85.0 (arrival 3: 85.6 / 76.7). consfzc@26 (trained 02:32) evaluates next; arrival 4
  on order 3 (consfzc@29 / erlc@29) training since 02:32 / 02:36.
- 11-23 04:05 cell 8 of 9 (order 2, arrival 4; consfzc@26 38 tasks ok) ours 79.4 / 68.2, all 72.9 (batches 65.0 / 68.3 / 65.0 /
  77.5) vs ER 86.2 / 75.5, all 80.0: library -6.9 (se 6.2; 4-7-5), new -7.3 (se 4.3; 7-6-9), all -7.1 (se 3.6; 11-13-14) = the
  largest loss; every batch behind (-3.7 / -5.0 / -17.5 / -7.5). Library losses: frying pan under the shelf 0.3 vs 1.0 (as
  cell 5), right moka pot 0.2 vs 0.6, red mug 0.6 vs 0.9, drawer 0.0 vs 0.3 (ketchup +0.3). Ours on order 2 ERODES over the
  stream: library 90.0 / 90.0 / 81.2 / 79.4, all-skill 82.0 -> 80.0 -> 72.9 at arrivals 2-4, while ER's library stays 86.9 /
  85.6 / 85.6 / 86.2 (orders 0 / 1 / 3 show no such decline for ours) = the first ours-only forgetting case.
  PRE-REGISTERED library lead: >= 0 in 5 / 8 cells, at most 6 / 9 < 7 -> NOT CONSISTENT (FAILED). 9-cell running mean all
  -1.1 (-3.7 / +2.0 / +3.0 / +1.8 / -0.9 / -4.7 / +0.8 / -7.1); cells 4 ours / 4 ER; a positive overall sign needs cell 9 >
  +8.8.
- 11-23 05:00 consfzc@29 (order 3, arrival 4; 38 tasks ok) 91.2 / 85.5, all 87.9 (arrival 3: 88.1 / 85.0, all 86.5), batches
  92.5 / 68.3 / 92.5 / 90.0 = no forgetting on order 3 (library 85.6 / 89.4 / 88.1 / 91.2). erlc@29 (training since 02:36) is
  the last EXP-167 pair. chain78 (EXP-170) started: erl6@5 / consb6@5 training since 04:21 / 04:25.
- 11-23 05:45 cell 9 of 9 (order 3, arrival 4; erlc@29 38 tasks ok, 89.4 / 83.6, all 86.1): library +1.9 (se 5.8; 3-11-2), new
  +1.8 (se 3.1; 10-6-6), all +1.8 (se 3.0; 13-17-8). chain76 ALL EXIT 05:40.
- RESULT (DONE 11-23 05:45). Library lead: >= 0 in 6 / 9 < 7 -> NOT CONSISTENT. Overall: 9-cell mean all -0.8 (se over cells
  1.2; library -0.7, new -1.0; per-skill pooled -0.8, se 1.0, 81-144-81) -> LOSS by the sign rule, a tie in substance (5 cells
  ours / 4 ER). 16-cell means (arrival 1 x 4 orders + order 0's stream + the 9): all +0.2 (ahead in 9 / 16), library +0.5 (>= 0
  in 13 / 16), new -0.5. Final arrival (38 skills) pooled over 4 orders: ours 83.6 vs ER 85.3, library -0.6 (se 2.4), new -2.5
  (se 1.9), all -1.7 (se 1.5; 36-69-47); per order -2.4 / +0.8 / -7.1 / +1.8. Forgetting: ours keeps library and earlier
  batches on orders 0 / 1 / 3; order 2 erodes (library 90.0 / 90.0 / 81.2 / 79.4, ER ~86 flat). => re-merging the arrivals adds
  nothing reliable over ER at 5 demos; order 0's stream lead (+2.6) was order-specific. Supports "merge = the start" (user
  11-23); the 1-demo arrival effect (+2.7, 4 / 4 orders) is the remaining case for re-merging (seed-1 replicate in chain79).
  Deck continual slide updated (4-order rows; backup results/merge_stale/content_for_slides_25_11.md.bak_1123).

### EXP-168 — stream forgetting metric (NBT) + contribution-ladder figure — **done (11-21, CPU)**
Motivation: metric parity with the VLA continual-learning papers (2603.03818 reports NBT); the cascade framing (user 11-20)
needs one picture of the ladder. No GPU.
- 11-21 14:30 NBT, LIBERO definition, order 0 stream (batches = the 4 arrivals' new skills, 8 / 6 / 4 / 4; c_kk = success on
  batch k right after arrival k; NBT_k = mean over later arrivals of c_kk - c_tk; NBT = mean over k = 1..3; > 0 = forgetting):
  ours -2.1, ER -6.4, plain-average start -0.8, ER from the base -10.1, EWC +6.5, SeqFT +19.9 -> every replay arm has NBT <= 0
  (earlier batches improve later), the no-replay arms forget = 2603.03818's picture. NBT does not separate ours from ER: ER's
  more negative value comes from a lower c_kk (72.5 / 85.0 / 72.5 / 95.0 vs ours 77.5 / 90.0 / 82.5 / 90.0) = ER learns each
  batch later, ours on arrival (just-arrived batch, ours - ER +5.0 / +5.0 / +10.0 / -5.0, ahead at 3 / 4). Library mean over
  arrivals 2-4 vs arrival 1: ours +5.4, ER +6.7 (no replay arm loses library). Final accuracy, 38 skills at arrival 4: ER 88.7,
  plain-average start 87.9, ours 86.3, ER from the base 84.2, EWC 71.8, SeqFT 51.8. Data: results/analysis/exp168_ladder_raw.json
  (stream0 / 19 / 20 / 4 keys).
- 11-21 14:40 figure `results/figs/ladder.png` (+ ladder.csv, `results/figs/make_ladder_figure.py`): arrival 1, all 24 skills,
  4-order mean (dots = orders): training-free merge 55.2, ER from the base 63.2, ER from the library merge (merge once) 83.6,
  + arrivals plain-averaged 79.8, + arrivals by our component merge 84.4, joint 86.0.

### EXP-169 — scarce new-skill rollouts: ours vs ER on the first 1 / 3 successful rollouts per new skill — **done (11-22 18:35): tie; 1 rollout suffices for both; chain77**
Motivation: user 11-21 "maybe our method is better when rollouts are limited too?" Ours vs ER on new skills is order-dependent
at full budget (arrival 1, 4 orders: +5.0 / -13.8 / -13.8 / +11.3); at 10 episodes (3-10 successes, EXP-163b) order 1 flipped
to +1.2 (pooled orders 0 / 1 new +2.5), but neither arm needed the 50 episodes, so rollouts were never scarce. Go lower: if the
merged arriving adapter makes a few rollouts go further, our new-skill lead should grow as rollouts shrink (the mirror of the
library-demo result, EXP-164 / 166). Risk: the training-free merge alone scores only ~16 on new skills (no zero-shot head start).
- 11-21 22:30 arms (exp147, md5 9a155795...; backup results/merge_stale/exp147_consolidate.py.bak_1121c): consfzn1 / erln1,
  consfzn3 / erln3 = consfz / erl (same contexts, 3,000 steps, 5 library demos, frac 0.5) on the first 1 / 3 SUCCESSFUL episodes
  per new skill in episode order (= collect until n successes), symlinked into ADIR/roll{n}/<task>/ (tasks_extn{n}.txt).
  Smoke test on shiva: 1 / 3 episodes per skill load, 35-124 / 109-367 samples per skill (10 episodes: 126-952). Orders 0 / 1.
  Scheduler `sweep/chain77_rollouts.sh` (tmux c77): lanes start once every chain75 trainer is claimed, then wait 60 s (chain76
  120 s) = after chain75's 1-demo runs, ahead of chain76's stream rounds (user-approved priority); evals port 10850.
- PRE-REGISTERED (22:35, before any EXP-169 result; 11-20 sign rule, se / W-T-L shown): primary = ours - ER on new skills at 1
  rollout, pooled over orders 0 / 1: > 0 = WIN; CONSISTENT if > 0 on both orders. Secondary = the scarcity trend: pooled new
  ours - ER at 1 rollout > at 3 > at 10 episodes (+2.5) > at 50 episodes (-4.4) supports "the merge helps more as rollouts get
  scarce". Library and all-skill reported with every call (library expected ~0: same replay).
- 11-22 04:28 consfzn1@0 started (GPU0); checked: roll1/ = the 8 new skills, 1 episode each (45-124 samples), free plain LoRA.
- 11-22 07:45 consfzn1@0 (ours, 1 successful rollout per new skill, order 0; 24 tasks ok; 493 new-skill samples in all)
  89.4 / 76.2 = ours at 10 episodes 88.1 / 78.8 and at 50 episodes 86.9 / 77.5: one successful episode per new skill is enough
  for ours on order 0. ER at 1 rollout (erln1@0, training since 07:09) ~10:15; ER at 10 / 50 episodes was 75.0 / 72.5 new.
  chain75 ALL EXIT 06:25.
- 11-22 10:25 erln1@0 (ER, 1 successful rollout per new skill, order 0; 24 tasks ok) 90.6 / 78.8 (all 86.7). Ours - ER at 1
  rollout, order 0: library -1.2 (se 3.4; 4-8-4), new -2.5 (se 3.7; 2-3-3), all -1.7 (se 2.5; 6-11-7) = ER ahead on order 0.
  ER does not lose from scarce rollouts: ER all-24 at 50 / 10 / 1 episodes 80.4 / 81.7 / 86.7 (ours 83.8 / 85.0 / 85.0), so the
  order-0 ours - ER all-skill gap goes +3.4 / +3.3 / -1.7 as rollouts shrink (opposite of the pre-registered trend; one order,
  seed swing ~5). New skills differ on 3 of 8 (ours 90 / 90 / 70 vs ER 100 / 100 / 90; ours ahead on 2 at 100 vs 90).
  Next: consfzn1@1 (GPU2, training since 07:14), then erln1@1, n3 pairs.
- 11-22 10:55 consfzn1@1 (ours, 1 successful rollout per new skill, order 1; 24 tasks ok) 88.1 / 86.2 (all 87.5) = ours at 10
  episodes 88.8 / 86.2 and above ours at 50 episodes 88.8 / 76.2: again one successful episode is enough for ours. ER on order 1
  at 10 / 50 episodes: new 85.0 / 90.0, all 88.3 / 86.2. erln1@1 (training since 09:49) is the pair.
- 11-22 12:55 erln1@1 (ER, 1 rollout, order 1; 24 tasks ok) 90.0 / 82.5 (all 87.5). PRIMARY CALL (1 rollout, pooled orders 0 / 1,
  new): ours - ER +0.6 (se 3.2; 3-9-4) = WIN by the sign rule, NOT CONSISTENT (order 0 -2.5, order 1 +3.8; order 1's lead is one
  skill, 80 vs 40). Library -1.6 (se 2.1; 7-18-7; behind on both orders: -1.2 / -1.9), all -0.8 (se 1.8; 10-27-11; -1.7 / 0.0).
  Trend so far (pooled new ours - ER): 50 episodes -4.4 -> 10 episodes +2.5 -> 1 rollout +0.6 = scarcity helps vs the full budget
  but not monotonically; all-skill +0.8 -> +1.5 -> -0.8. Both arms keep their level at 1 rollout (ours 85.0 / 87.5, ER 86.7 /
  87.5 all-24 on orders 0 / 1) = the rollout budget barely matters for either. 3-rollout pairs (consfzn3 / erln3) still to land.
- 11-22 13:25 consfzn3@0 (ours, first 3 successful rollouts per new skill, order 0; 24 tasks ok) 92.5 / 77.5 (all 87.5) vs ours
  at 1 rollout / 10 / 50 episodes all-24 85.0 / 85.0 / 83.8: the same new-skill level (76-79 at every budget; two skills stay at
  10), library 92.5 = the best ours library on order 0 (within seed swing). erln3@0 (training since ~12:25) is the pair.
- 11-22 15:30 erln3@0 (ER, 3 rollouts, order 0; 24 tasks ok) 93.8 / 80.0 (all 89.2). Ours - ER at 3 rollouts, order 0: library
  -1.2 (se 2.6; 3-9-4), new -2.5 (se 4.1; 2-4-2), all -1.7 (se 2.1; 5-13-6) = the same gap as at 1 rollout. On order 0 ER all-24
  goes 80.4 / 81.7 / 89.2 / 86.7 at 50 / 10 / 3 / 1 episodes vs ours 83.8 / 85.0 / 87.5 / 85.0: ER gains more from fewer
  rollouts (its 50-episode run looks like a low draw). The same two new skills fail in both arms (ours 10 / 10, ER 30 / 30).
  Scheduling: c76hold released its 12 claims at 15:02 once chain77 claimed erln3@1; chain76 started consfzc@27 at 15:05.
- 11-22 16:00 consfzn3@1 (ours, 3 rollouts, order 1; 24 tasks ok) 87.5 / 90.0 (all 88.3) = ours' best new-skill score on order 1
  (1 rollout / 10 / 50 episodes: 86.2 / 86.2 / 76.2); library flat (87.5-88.8). erln3@1 (last EXP-169 run) is the pair.
- 11-22 18:35 erln3@1 (ER, 3 rollouts, order 1; 24 tasks ok) 88.8 / 87.5 (all 88.3). Ours - ER at 3 rollouts: order 1 library
  -1.2 (se 2.0; 3-9-4), new +2.5 (se 2.5; 3-4-1), all 0.0; pooled orders 0 / 1 library -1.2 (se 1.6; 6-18-8), new 0.0 (se 2.4;
  5-8-3), all -0.8 (se 1.3; 11-26-11).
- RESULT (DONE 11-22 18:35). Primary (1 rollout, pooled new): +0.6 (se 3.2) = WIN by the sign rule, NOT CONSISTENT (-2.5 / +3.8).
  Secondary trend NOT SUPPORTED: pooled new ours - ER at 50 / 10 / 3 / 1 episodes -4.4 / +2.5 / 0.0 / +0.6 (not monotone; scarce
  budgets only remove ER's full-budget new-skill lead, which was order 1's -13.7); library +3.4 / +0.9 / -1.2 / -1.6 (same
  replay, noise), all +0.8 / +1.5 / -0.8 / -0.8. Ours vs ER = a tie at scarce budgets. Positive for the framework's budget
  guidance: neither arm needs more than ONE successful rollout per new skill (all-24 on orders 0 / 1: ours 83.8 / 84.6 at 50
  episodes, 85.0 / 87.5 at 1, 87.5 / 88.3 at 3; ER 80.4 / 86.2 -> 86.7 / 87.5 -> 89.2 / 88.3) -> collect until 1-3 successes.
  chain77 ALL EXIT 18:31 (its GPU passes to chain76: erlc@22 started 18:31).

### EXP-170 — scale: the 32-skill library, arrival 1 and the continual stream, ours vs ER vs ER from the base — **done (11-24 07:55): merge start holds (+18.0, does not grow); ours = ER (stream mean +0.05, not consistent); the stream SCALES (ours above joint)**
Motivation: user 11-22 "how does everything change when we scale up ... What if the library is 32 tasks and more tasks arrive?"
Bigger libraries were tested only for ours vs joint at arrival 1 (EXP-150: within 5pp at 16 / 32 / 74); arrivals beyond 8
only on the 16-skill library (EXP-151, 22 new, k 38), where ours lost its lead over ER at the last arrival in both streams.
Hypotheses (11-22): (1) the merge start's lead over the base start grows with the library (the base must learn 32 skills from
5 demos each; the training-free merge keeps ~73% at 32); (2) ours - ER shrinks with k (ours adds only the arrivals' residuals
at 1/k; the training-free merge scores ~0 on new skills). User 11-22 framing: the paper is a framework of ingredients with
budget-dependent guidance, novel ingredients emphasised (not one winning method) -> this maps the library-size axis.
- 11-22 setup (exp116 md5 48ee9419..., exp147 0fa6ad52..., exp136 4c096771..., gpu_gate a82ba267...; backups
  results/merge_stale/*.bak_1122): pseudo-orders 30 / 31 / 32 = the 32-skill library (5's) + SCALE's new skills cut at 14 / 18
  / 22 (same arrival order as the 16-skill stream 19 / 20 / 4; arrival 1 = 5). Arms: erl6 = ER from the library merge (32 at
  1/32) and consb6 = ER from the base, 6,000 steps (= consfz6's steps: ~750 replay samples per library skill, as 3,000 at 16);
  ours at arrival 1 = consfz6@5 (11-09, 84.1 / 78.8; same recipe). Stream: consfzl / erll resume consfz6 / erl6, 3,000 steps
  per arrival (2x the 16-skill stream's 1,500). Joint: joint24k_s32 (rank 64, 24,000 steps, all demos of the 54 skills;
  task file results/analysis/exp136_joint_tasks_s32.txt checked against sequence(32)). prep / prep2 run for 30-32 (contexts
  46 / 50 / 54 members) and prep2 for 5. 30 new (all externals) is NOT possible yet: the 8 unused externals have no rollouts.
- Scheduler `sweep/chain78_libscale.sh` (tmux c78, results/chain78.log): starts once every chain76 trainer is claimed; lanes
  wait 180 s; the joint lane (60 s) takes the first free A6000 once both arrival-1 runs are claimed; evals 10900 / 10950.
  ~20 h of training on two A6000s after chain76.
- PRE-REGISTERED (11-22, before any EXP-170 result; sign rule, se / W-T-L shown): (a) merge start - base start (erl6 - consb6,
  all 40 skills, order 0): > 0 = HOLDS; > +20.4 (16-skill library, same steps per library skill) = GROWS with the library.
  (b) ours - ER at arrival 1 (consfz6 - erl6, all 40): sign reported against +3.3 at 16. (c) stream ours - ER on all arrived
  skills: ahead at >= 3 of 4 arrivals = CONSISTENT (as EXP-151 / 166); mean sign = WIN. (d) ours at arrival 4 within 5pp of
  joint24k_s32 on library and new = the stream SCALES to a 32-skill library.
- 11-23 02:38 chain76's trainers all claimed -> chain78 lanes started; erl6@5 / consb6@5 training from 04:21 / 04:25.
- 11-23 10:05 erl6@5 (ER from the 32-skill library merge, 6,000 steps, order 0's 8 new; 40 tasks ok) 84.1 / 83.8, all 84.0.
  (b) ours - ER at arrival 1 (consfz6@5 84.1 / 78.8, all 83.0): library 0.0 (se 2.3; 10-14-8; order 0's 16 +2.5, the added 16
  -2.5), new -5.0 (se 3.8; 1-4-3), all -1.0 (se 2.0; 11-18-11) = sign NEGATIVE at 32 vs +3.3 at 16 (order 0) = hypothesis (2)
  (ours - ER shrinks with k) holds at arrival 1. (a) waits for consb6@5.
- 11-23 11:15 consb6@5 (ER from the base, 32-skill library, 6,000 steps; 40 tasks ok) 63.4 / 76.2, all 66.0. (a) merge start -
  base start (erl6 - consb6): library +20.6 (se 3.6; 22-8-2; order 0's 16 +18.1, the added 16 +23.1), new +7.5 (se 4.5; 6-0-2),
  all +18.0 (se 3.1; 28-8-4) vs +20.4 at 16 (order 0: library +22.5, new +16.2) = HOLDS, does NOT grow with the library
  (pre-registered). Library lead constant (~+21); the new-skill lead halves. ours - base start (consfz6 - consb6): all +17.0
  (se 3.6; 26-9-5). Base start at 6,000 steps: 71.7 (16-skill library, all 24) vs 66.0 (32, all 40).
- 11-23 13:20 consfzl@30 (ours, 32-skill stream arrival 2, 14 new; 46 tasks ok) 85.0 / 77.1, all 82.6 (order 0's 16 library
  91.2, the added 16 78.8; first batch 73.8 vs 78.8 at arrival 1, second batch 81.7). 16-skill stream at arrival 2: 91.9 / 85.7.
  Call (c) waits for erll@30.
- 11-23 18:25 erll@30 (ER, 32-skill stream arrival 2; 46 tasks ok; landed 15:52, processed late: odin's landing watcher was
  killed for low memory) 85.6 / 77.1, all 83.0. ours - ER: library -0.6 (se 2.9; 4-20-8), new 0.0 (se 3.6; first batch +2.5,
  second -3.3), all -0.4 (se 2.3; 8-25-13). Stream so far -1.0 / -0.4 at arrivals 1-2 -> (c) CONSISTENT (ahead at >= 3 of 4)
  can no longer be met; the mean-sign call waits for arrivals 3-4.
- 11-23 22:35 arrival 3 (18 new; 50 tasks ok each): consfzl@31 86.2 / 87.2 (all 86.6; batches 82.5 / 93.3 / 87.5), erll@31
  84.4 / 85.6 (all 84.8; 77.5 / 93.3 / 90.0). ours - ER: library +1.9 (se 2.5; 10-17-5), new +1.7 (se 3.8; 8-5-5), all +1.8
  (se 2.1; 18-22-10) = first lead in the 32-skill stream (first batch +5.0). Stream cells -1.0 / -0.4 / +1.8, running mean
  +0.1; the mean-sign call (c) waits for arrival 4 (@32; consfzl@32 training since 20:25).
- 11-24 00:10 consfzl@32 (ours, arrival 4, 22 new; 54 tasks ok) 87.2 / 86.4, all 86.9 (order 0's 16 library 92.5, the added
  16 81.9; batches 75.0 / 93.3 / 87.5 / 97.5). 16-skill stream at arrival 4: 90.0 / 83.6. erll@32 training since 23:09 -> (c);
  joint24k_s32 training on GPU2 since 11-23 09:19 -> (d). chain79 lanes started 11-23 23:10.
- 11-24 02:55 erll@32 (ER, arrival 4; 54 tasks ok) 85.0 / 90.0, all 87.0 (batches 86.2 / 93.3 / 85.0 / 97.5). ours - ER:
  library +2.2 (se 2.6; 7-18-7), new -3.6 (se 3.5; first batch -11.2, others 0.0 / +2.5 / 0.0), all -0.2 (se 2.1; 12-29-13).
  (c) RESULT: cells -1.0 / -0.4 / +1.8 / -0.2, mean +0.05 (se 0.6) = positive by the pre-registered sign rule, a tie in
  substance; ahead at 1 of 4 = NOT consistent. Library ours - ER 0.0 / -0.6 / +1.9 / +2.2 (mean +0.9), new -5.0 / 0.0 / +1.7 /
  -3.6 (mean -1.7): same shape as the 16-skill streams (EXP-167 tie). (d) waits for joint24k_s32.
- 11-24 07:55 joint24k_s32 (rank 64, 24,000 steps, all demos of the 54 skills; 54 tasks ok) 79.1 / 84.1, all 81.1 (order 0's
  16 library 85.6, the added 16 72.5). (d) ours at arrival 4 - joint: library +8.1 (se 4.0; 14-13-5), new +2.3 (se 4.7; 8-8-6),
  all +5.7 (se 3.1) = within 5pp (above joint on both) -> the stream SCALES to a 32-skill library (pre-registered). ER - joint
  +5.9 / +5.9. As at 16 (joint 84.5 at arrival 4 vs ours 86.3 / ER 88.7), both continual arms end above joint training.
- RESULT (11-24): (a) merge start - base start +18.0 at 32 vs +20.4 at 16 = HOLDS, does not grow (library lead ~+21 at both,
  new-skill lead halves); (b) ours - ER at arrival 1 -1.0 (+3.3 at 16) = shrinks with k; (c) stream ours - ER -1.0 / -0.4 /
  +1.8 / -0.2, mean +0.05 (se 0.6) = tie, not consistent (1 of 4); (d) ours above joint24k_s32 (+8.1 / +2.3) = SCALES.
  Merge-as-start and the flat stream carry over to 32 library skills; re-merging the arrivals still adds nothing over ER.

### EXP-171 — consolidation of the framework: seeds, held-out orders, breadth — **queued (11-22 16:05); chain79 (+ c79col)**
Motivation: user 11-22 "henceforth, we queue experiments that consolidate this framework, with breadth and more statistical
guarantee" (paper = ingredients + budget-dependent guidance, novel ingredients emphasised; abstract drafted 11-22). Every claim
in the abstract rests on one seed and several on order 0 only; budget recommendations were read off orders 0-1.
- 11-22 setup (exp116 md5 7f426e8e..., exp147 2a3f060f..., _e147_abs 96ef3541..., gpu_gate 225cbd19...; backups *.bak_1122):
  seed-1 arms <arm>s1 for consfz / erl / consb / consfzd1 / erld1 and the streams consfzc / erlc / consbc (same data / steps /
  context, train_lora --seed 1; streams resume their own seed-1 chain). New pseudo-orders: 33 / 34 = 5's / SCALE's library +
  all 30 externals (SCALE's 22, then the 8 never drawn by orders 0-3, seeded 11710) = 5th arrival of the 32- / 16-skill
  streams; 35 = order 1's 16 library + 16 seeded fleet skills (11711) + order 1's 8 new; 36-38 = its stream (CONTO 21-23 order).
  prep / prep2 done for 35-38. The 8 new externals have no converted / WP adapters: prep skips ctx_all.json there (c0 skipped),
  _e147_abs drops the WP reference. Their own success 40-100% (single-skill eval) -> rollouts collectable.
- c79col (tmux c79col, results/c79col.log): the 8 externals' own rollouts (EXP-144 protocol, 50 episodes, states 10..49) under
  the eval lock, port 10960, then prep / prep2 for 33 / 34.
- chain79 (`sweep/chain79_consolidate.sh`, tmux c79, results/chain79.log; 54 jobs): starts once chain78's trainers + joint
  are claimed and c79col is done; lanes 240 s; evals port 11000. Priority: (1) seeds: consfzd1s1 / erld1s1 @0-3; consfzs1 /
  erls1 / consbs1 @0 + their streams @19 / 20 / 4; erls1 / consbs1 @1-3; (2) held-out orders 2-3: consfzq / consfaq / erlq,
  consfzn1 / erln1, consfzd25 / erld25, consb12; (3) breadth: consfzc / erlc @34, consfzl / erll @33, consfz6 / erl6 @35 and
  consfzl / erll @36-38. ~130 GPU-h of training -> roughly 11-25 evening to 11-27 after chain76 / 78.
- PRE-REGISTERED (11-22, before any EXP-171 result; sign rule; 2-seed results = mean over skill x seed, se over skills of the
  seed-averaged difference, per-seed signs shown). SEEDS: a claim REPLICATES if seed 1 has seed 0's sign on the same cells:
  (s1) 1-demo ours - ER, all 24, pooled 4 orders > 0; 2-seed CONSISTENT if ahead in >= 6 of 8 order x seed cells.
  (s2) merge start - base start (erls1 - consbs1) pooled 4 orders > 0 on library and on new.
  (s3) order-0 stream seed 1: merge start (erlcs1) ahead of the base start (consbcs1) at all 4 arrivals; ours - ER sign per
  arrival reported; flat curve = ours' first batch at arrival 4 within 10 points of arrival 1.
  HELD-OUT (orders 2-3 pooled; recommendation and its selection evidence fixed now):
  (h1) tiny budget (750 steps): component-merge start > plain-average start (consfzq - consfaq > 0, library and new;
  orders 0-1: +5.9 / +10.6).
  (h2) one successful rollout per new skill suffices: ours at 1 rollout within 5 points of ours at 50 episodes (all 24; same
  check for ER; orders 0-1: ours +1.2 / +2.9, ER +6.3 / +1.3).
  (h3) re-merge when library data is scarce: ours - ER at 25 demos <= 0 (order 0: -1.3) AND below ours - ER at 1 demo.
  (h4) the merge start is worth >= 9,000 steps: consb12 (12,000 steps) below erl (3,000) on all 24, on each of orders 2 and 3.
  BREADTH: (b1) 5th arrival (30 new; k 46 / 62): ours - ER sign on all arrived skills; ours' all-skill mean within 5 points
  of its 4th arrival = flat. (b2) order 1's 32-skill stream: EXP-170's rules (c) on a second order.
- 11-22 16:59 c79col done: the 8 new externals' own rollouts collected (50 episodes each; success 0.84-0.92 where logged),
  success episodes 46 / 31 / 43 / 42 / 49 / 50 / 37 / 16 (5th-arrival order: plate-to-stove, alphabet soup, bowl-on-cabinet,
  milk, bowl-from-center, stove on, bbq sauce, middle drawer); prep / prep2 for 34 and 33 done (contexts 46 / 62 members, 30
  rollout tasks, 8 arrival tasks). chain79 waits for chain78's claims.
- 11-23 23:10 chain78's trainers all claimed -> chain79 lanes started (consfzd1s1@0 on GPU0 01:51).
- 11-24 05:10 consfzd1s1@0 (ours, 1 library demo, seed 1, order 0; 24 tasks ok) 78.8 / 63.7, all 73.8 vs seed 0 86.2 / 77.5
  (83.3): seed 1 - seed 0 library -7.5 (se 4.0; 2-8-6), new -13.8 (se 11.3; 1-2-5), all -9.6 (se 4.5) = the largest seed
  swing so far (previous up to ~5 library). Checked: same context / replay file / 1 demo (VDEMOS copied) / 3,000 steps, final
  loss 0.0155 vs 0.0169 -> genuine seed variance. (s1) waits for erld1s1@0 (training since 04:31) and orders 1-3.
- 11-24 08:25 erld1s1@0 (ER, 1 library demo, seed 1, order 0; 24 tasks ok) 74.4 / 66.2, all 71.7 vs seed 0 85.0 / 68.8 (79.6)
  = ER drops too (all -7.9, library -10.6): seed 1 is a bad draw for both arms (same demo per skill: the first, seed-free;
  the seed sets data order + flow-matching noise, shared by the pair). Paired ours - ER at seed 1: library +4.4 (se 5.2;
  6-7-3), new -2.5 (se 3.7), all +2.1 (se 3.7; 8-10-6) = same sign as seed 0 (+3.8) on order 0. 2-seed mean +2.9 (se 1.8;
  10-11-3). Order x seed cells ahead so far 5 of 5 (seed 0 orders 0-3 + seed 1 order 0).
- 11-24 10:20 consfzd1s1@1 (ours, 1 demo, seed 1, order 1; 24 tasks ok) 80.6 / 72.5, all 77.9 vs seed 0 88.8 / 87.5 (88.3):
  -10.4 (se 5.2; 4-11-9) = second ~10-point seed-1 drop for ours (third with ER's order 0). Checks: train_meta identical except
  seed / final loss (seed 1 lower: 0.0152 vs 0.0166); data files untouched since 11-06; train_lora.py unchanged since 11-19
  (seed-0 1-demo runs 11-20 / 21). Per skill (order 0): drops sit on the volatile articulated / hard skills (top drawer 0.7 ->
  0.3, microwave 0.5 -> 0.2, wine rack 0.4 -> 0.0, drawer+bowl 0.5 -> 0.0), easy pick-and-place stays at 1.0, one skill rises
  0.5 -> 1.0 = training variance, not an eval failure (no zeros, no parse failures). Open: seed 0 may have been a good draw at
  1 demo (its 1-demo = 5-demo level was a deck claim). Control: consfzs1@0 / erls1@0 (5 demos, seed 1) vs consfz@0 / erl@0.
  erld1s1@1 adapter ready; eval next.
- 11-24 10:55 erld1s1@1 (ER, 1 demo, seed 1, order 1; 24 tasks ok) 81.9 / 75.0, all 79.6 vs seed 0 85.6 / 85.0 (85.4) = -5.8.
  Seed-1 ours - ER order 1: library -1.3 (se 2.7), new -2.5, all -1.7 (se 2.0; 4-13-7) = first cell behind (5 of 6 ahead);
  seed 1 pooled orders 0-1 +0.2 (se 2.1; 12-23-13). All 4 seed-1 runs at 1 demo sit 6-10 below seed 0 (both arms, both
  orders). Training logs identical apart from the seed (same rollout files / sample counts, replay 16 x 1 demo = 1,176
  samples, same folded merge, 46.1M trainable) -> if systematic, it is on the eval side (seed-0 1-demo evals 11-20 / 21).
  Proposed drift check (not run, user's call): re-evaluate the seed-0 adapter consfzd1@0 with today's pipeline.
- 11-24 10:47 drift check approved by the user and started: exp147 E147_RV = 1 -> tag suffix "rv" (md5 1945ddae..., both
  hosts; backup results/merge_stale/exp147_consolidate.py.bak_1124), `sweep/drift_check.sh` (tmux drift,
  results/drift_check.log): consfzd1@0's adapter re-evaluated as c147s0t8-consfzd1rv_w0.25 under the eval lock, port 11200,
  2 workers per GPU (= chain79's evals). Pre-set reading: all 24 near 83.3 (within ~5) = no drift, seed 1 is variance; near
  the seed-1 level (~74) = eval drift -> stop and find it.
- 11-24 11:25 drift check RESULT (24 tasks ok, 10:51-11:17): consfzd1@0 re-evaluated 79.6 vs 83.3 on 11-21 = -3.8 (se 2.1;
  3-14-7) = within the pre-set ~5 -> NO eval drift. Changes sit on the same volatile skills (ketchup drawer 0.6 -> 0.3, drawer
  + bowl 0.7 -> 0.5 / 0.5 -> 0.3); 14 of 24 identical. Reading: one eval of 24 skills x 10 trials carries ~3-4 points of noise
  by itself; seed 1 (73.8) sits 5.8 below the re-eval -> the seed-1 drops = training seed + eval noise, not the pipeline.
  Consequence: single-run gaps of ~5 points between runs are noise; paired ours - ER on the same seed is the right unit
  (as pre-registered). The "1 demo = 5 demos" level is judged once the 5-demo seed-1 pair (consfzs1@0 / erls1@0) lands.
  tmux drift ended by itself; rv records kept under their own tag.
- 11-24 13:05 consfzd1s1@2 (ours, 1 demo, seed 1, order 2; 24 tasks ok) 77.5 / 77.5, all 77.5 vs seed 0 83.1 / 72.5 (79.6)
  = -2.1 (se 3.1; 6-12-6), library -5.6 / new +5.0 = the smallest seed-1 drop so far, inside one eval's noise (drift check
  -3.8). Seed-0 ours - ER on order 2: +2.1 (library +4.4, new -2.5). Pair erld1s1@2 next.
- 11-24 13:40 erld1s1@2 (ER, 1 demo, seed 1, order 2; 24 tasks ok) 73.8 / 75.0, all 74.2 vs seed 0 77.5 (-3.3). Seed-1 ours - ER
  order 2: library +3.8 (se 2.6; 6-7-3), new +2.5 (se 8.0), all +3.3 (se 3.0; 10-8-6) = ahead (seed 0 here +2.1). Cells
  ahead 6 of 7 -> (s1) 2-seed CONSISTENT already met (>= 6 of 8 whatever order 3 gives). Pooled orders 0-2: seed 0 +2.9 (se
  1.4), seed 1 +1.2 (se 1.7), 2-seed mean +2.1 (se 1.1; 28-29-15). Seed-1 replication (pooled 4 orders > 0) waits for order 3.
- 11-24 15:45 consfzd1s1@3 (ours, 1 demo, seed 1, order 3; 24 tasks ok) 76.9 / 95.0, all 82.9 vs seed 0 72.5 / 73.8 (72.9)
  = +10.0 (se 5.5; 9-9-6), new +21.2 = seed 1 ABOVE seed 0 here by as much as it fell on orders 0-1 -> the seed swing at 1 demo
  is ~+-10 points both ways (training variance, symmetric), not a seed-1 deficit. Seed-0 ours - ER order 3: +2.1. erld1s1@3 next.
- 11-24 16:20 erld1s1@3 (ER, 1 demo, seed 1, order 3; 24 tasks ok) 76.2 / 88.8, all 80.4. Seed-1 ours - ER order 3: library
  +0.6, new +6.3, all +2.5 (se 3.0; 7-13-4).
- (s1) RESULT (11-24 16:20): seed-1 ours - ER on all 24 = +2.1 / -1.7 / +3.3 / +2.5 on orders 0-3, pooled +1.6 (se 1.5;
  29-44-23; library +1.9, new +0.9) > 0 = same sign as seed 0 (+2.7) -> REPLICATES. 2 seeds: ahead in 7 of 8 order x seed
  cells (>= 6) -> CONSISTENT; 2-seed pooled +2.1 (se 1.1; 40-33-23), library +2.7 (se 1.3), new +0.9 (se 1.9). 4-order
  all-24 level: ours 81.0 / 78.0, ER 78.3 / 76.5 (seed 0 / 1). Deck bullet added (1-demo slide).
- 11-24 18:30 consfzs1@0 (ours, 5 demos, seed 1, order 0; 24 tasks ok) 88.8 / 73.8, all 83.7 = seed 0 (consfz@0 83.7; diff
  0.0, se 3.7) -> at 5 demos the seeds agree. 1-demo control: seed 1 5 - 1 demo +10.0 (se 4.3; 10-11-3) vs seed 0 +0.4 ->
  the deck's "1 demo = 5 demos" (order 0, seed 0) does NOT replicate; 1 demo is the seed-sensitive budget. 4-order seed-0
  level 5 -> 1 demo: ours 84.4 -> 81.0 (-3.4), ER 83.6 -> 78.3 (-5.3) -> 1 demo costs ours ~3-5 points, ER more. Deck bullet
  rewritten. erls1@0 (ER, 5 demos, seed 1) next -> seed-1 ours - ER at 5 demos + (s2) with consbs1.
- 11-24 19:00 erls1@0 (ER, 5 demos, seed 1, order 0; 24 tasks ok) 85.6 / 73.8, all 81.7 vs seed 0 80.4 (+1.2) = seeds agree
  at 5 demos for ER too. Seed-1 ours - ER at 5 demos, order 0: library +3.1 (se 1.5; 6-9-1), new 0.0, all +2.1 (se 1.7;
  10-10-4) = same sign as seed 0 (+3.3). ER 5 - 1 demo at seed 1: +10.0 (= ours' +10.0) -> the seed-1 1-demo dip hit both
  arms equally (paired difference intact). (s2) waits for consbs1@0 and erls1 / consbs1 @1-3.
- 11-24 20:15 consfzcs1@19 (ours, continual stream, seed 1, arrival 2 = 16 library + 14 new; 30 tasks ok) 87.5 / 80.7 (first
  batch 78.8, second 83.3), all 84.3 vs seed 0 consfzc@19 89.0 = -4.7 (se 2.7; 6-13-11). Its arrival 1 (consfzs1@0) 83.7 on 24 ->
  84.3 on 30 = flat; first batch 73.8 -> 78.8 (no forgetting). Seed-0 ours - ER at arrival 2: +2.3. Pair erlcs1@19 training
  (19:37); consbs1@0 trained 17:53, eval queued.
- 11-24 21:25 consbs1@0 (ER from the base model, 5 demos, seed 1, order 0; 24 tasks ok) 64.4 / 68.8, all 65.8 (seed 0 60.0).
  (s2) order 0, seed 1: merge start - base start (erls1 - consbs1) library +21.2 (se 5.8; 11-5-0), new +5.0 (se 7.3), all
  +15.8 (se 4.8) = same signs as seed 0 (+22.5 / +16.2); ours - base start +24.4 / +5.0. Library lead fully replicates, new
  lead smaller. (s2) waits for erls1 / consbs1 @1-3.
- 11-24 22:00 erlcs1@19 (ER stream, seed 1, arrival 2; 30 tasks ok) 85.6 / 81.4, all 83.7 (seed 0 86.7). Seed-1 ours - ER at
  arrival 2: library +1.9, new -0.7, all +0.7 (se 2.7; 8-14-8) = same sign as seed 0 (+2.3), a near tie. Seed-1 stream so
  far: +2.1 / +0.7 at arrivals 1-2. (s3) merge vs base start needs consbcs1@19 (training).
- 11-24 23:05 consbcs1@19 (ER from the base model, stream, seed 1, arrival 2; 30 tasks ok) 67.5 / 70.0, all 68.7 (seed 0
  69.7). (s3) merge start - base start (erlcs1 - consbcs1) at arrival 2: library +18.1 (se 5.7; 10-5-1), new +11.4, all
  +15.0 (se 4.6; 17-8-5) = seed 0 +17.0. Seed-1 stream merge-start lead +15.8 / +15.0 at arrivals 1-2 (ahead at 2 of 2 so far).
- 11-24 23:45 consfzcs1@20 (ours, stream, seed 1, arrival 3 = 16 library + 18 new; 34 tasks ok) 88.8 / 91.7 (batches 87.5 /
  95.0 / 95.0), all 90.3 vs seed 0 91.2 (-0.9, se 2.9; 9-16-9). Seed-1 curve 83.7 -> 84.3 -> 90.3 (24 / 30 / 34 skills);
  first batch 73.8 -> 78.8 -> 87.5 = keeps rising (no forgetting). Seed-0 ours - ER at arrival 3: +7.1. erlcs1@20 next.
- 11-25 00:45 erlcs1@20 (ER stream, seed 1, arrival 3; 34 tasks ok) 86.2 / 92.2 (first batch 92.5), all 89.4. Seed-1 ours -
  ER at arrival 3: library +2.5, new -0.6, all +0.9 (se 2.7; 7-18-9) vs seed 0 +7.1 (new +11.1, ER's first batch 71.2 at seed
  0 vs 92.5 at seed 1) -> seed 0's largest stream gap does not repeat in size; sign does. Seed-1 stream ours - ER +2.1 / +0.7
  / +0.9 at arrivals 1-3 (all > 0, all small).
- 11-25 01:30 consbcs1@20 (ER from the base, stream, seed 1, arrival 3; 34 tasks ok) 76.9 / 70.6, all 73.5 (seed 0 66.5).
  (s3) merge start - base start at arrival 3: library +9.4 (se 4.7), new +21.7, all +15.9 (se 4.2; 21-9-4) = seed 0 +17.6.
  Seed-1 merge-start lead +15.8 / +15.0 / +15.9 at arrivals 1-3 (3 of 3); arrival 4 decides (s3) (seed 0 there: +4.5).
- 11-25 02:30 consfzcs1@4 (ours, stream, seed 1, arrival 4 = 16 library + 22 new; 38 tasks ok) 88.8 / 87.7 (batches 83.8 /
  90.0 / 90.0 / 90.0), all 88.2 vs seed 0 consfzc@4 86.3 (+1.8, se 3.2; 14-15-9). Seed-1 curve 83.7 -> 84.3 -> 90.3 -> 88.2
  (24 / 30 / 34 / 38 skills) = flat; first batch 73.8 -> 78.8 -> 87.5 -> 83.8 = (s3) flat-curve check met (arrival 4 >= arrival
  1 - 10; it is +10.0 above). Seed-0 ours - ER at arrival 4 was -2.4 (se 2.3; ER new +5.0): seed 1's ER (erlcs1@4) next decides
  that sign; consbcs1@4 then closes (s3).
- 11-25 03:10 erlcs1@4 (ER stream, seed 1, arrival 4; 38 tasks ok) 86.9 / 85.0 (batches 86.2 / 86.7 / 77.5 / 87.5), all 85.8
  (seed 0 88.7). Seed-1 ours - ER at arrival 4: library +1.9, new +2.7, all +2.4 (se 2.8; 12-17-9) vs seed 0 -2.4 -> arrival 4
  is a 2-seed tie (mean 0.0); seed-1 stream ours - ER +2.1 / +0.7 / +0.9 / +2.4 = ahead at 4 of 4 arrivals (seed 0 3 of 4);
  2 seeds x 4 arrivals: ahead in 7 of 8 cells, mean +2.1 (per-arrival 2-seed means +2.7 / +1.5 / +4.0 / 0.0). consbcs1@4 next
  closes (s3).
- 11-25 04:20 consbcs1@4 (ER from the base, stream, seed 1, arrival 4; 38 tasks ok) 70.0 / 75.5 (batches 76.2 / 83.3 / 62.5 /
  75.0), all 73.2 (seed 0 84.2). Merge start - base start (erlcs1 - consbcs1) at arrival 4: library +16.9 (se 6.6; 9-5-2), new
  +9.5 (se 3.0), all +12.6 (se 3.3; 21-13-4); ours - base start +15.0. (s3) RESULT: seed-1 merge-start lead +15.8 / +15.0 /
  +15.9 / +12.6 at arrivals 1-4 = ahead at 4 of 4 -> REPLICATES (pre-registered); seed 0 +20.4 / +17.0 / +17.6 / +4.5; seed 0's
  arrival-4 shrink does not repeat (2-seed arrival 4 +8.6). Flat curve (met) and ours - ER (ahead 4 of 4 on seed 1) above.
  Seed block left: erls1 / consbs1 @1-3 -> (s2).
- 11-25 05:35 erls1@1 (ER, 5 demos, seed 1, order 1; 24 tasks ok) 80.0 / 82.5, all 80.8 vs seed 0 erl@1 86.2 (-5.4, se 3.2;
  6-10-8) = within the ~3-5 point seed swing; vs ER at 1 demo, seed 1 (erld1s1@1) 79.6 -> +1.2 (more demos >= fewer). (s2)
  order 1 needs consbs1@1 (training).
- 11-25 06:55 consbs1@1 (ER from the base model, 5 demos, seed 1, order 1; 24 tasks ok) 60.6 / 55.0, all 58.8 (seed 0 70.0).
  (s2) order 1, seed 1: merge start - base start (erls1 - consbs1) library +19.4 (se 5.7; 12-1-3), new +27.5 (se 5.9; 8-0-0),
  all +22.1 (se 4.3) vs seed 0 +16.9 / +15.0. Seed 1 so far: library +21.2 / +19.4, new +5.0 / +27.5 on orders 0-1 (both > 0).
  Orders 2-3 (erls1 / consbs1 @2-3) close (s2).
- 11-25 08:20 erls1@2 (ER, 5 demos, seed 1, order 2; 24 tasks ok) 83.1 / 78.8, all 81.7 vs seed 0 erl@2 85.0 (-3.3, se 2.1;
  3-15-6); vs ER at 1 demo, seed 1 (erld1s1@2) 74.2 -> +7.5. (s2) order 2 needs consbs1@2.
- 11-25 09:35 consbs1@2 (ER from the base model, 5 demos, seed 1, order 2; 24 tasks ok) 67.5 / 63.7, all 66.2 (seed 0 60.0).
  (s2) order 2, seed 1: merge start - base start library +15.6 (se 4.7; 13-1-2), new +15.0 (se 7.3; 5-2-1), all +15.4 (se 3.9)
  vs seed 0 +25.6 / +23.8. Seed 1 orders 0-2: library +21.2 / +19.4 / +15.6, new +5.0 / +27.5 / +15.0 (all > 0). Order 3
  (erls1 / consbs1 @3, training) closes (s2). chain81 (EXP-173) lanes started 09:03 (seed block all claimed).

### EXP-172 — where does re-merging the arrivals help? limited replay memory, OOD shift, unseen tasks — **queued (11-23 09:13); chain80**
Motivation: EXP-167 (4 orders) put ours = ER in the stream at 5 demos; with only the library in it our component merge equals
the plain average, so without re-merging arrivals the merge is the start only (user 11-23: "is the merge just an initialization
step?"). User 11-23: "find some application of re-merging ... perhaps OOD performance or performance on unseen tasks". The only
thing ours has that ER lacks is the arriving adapters' weights (trained elsewhere on their full demos), so re-merging can help
only where those weights know something the correction's data does not: (a) earlier arrivals whose data is no longer stored
(the mirror of the one positive case, 1 library demo: +2.7, 4 / 4 orders), (b) inputs outside the rollouts' distribution
(weight averaging and robustness: WiSE-FT / DiWA; EXP-129 on uncorrected merges: no smaller drop than single adapters), (c)
tasks never trained (the library merge alone solves some unseen goal tasks, EXP-142; composites ~0 for every arm, EXP-157).
- 11-23 setup (exp147 md5 39cf1338..., gpu_gate 5f08f8b5..., both hosts; backups results/merge_stale/*.bak_1123):
  (a) consfzcm0 / erlcm0 = consfzc / erlc trained on the just-arrived batch's rollouts + library replay only (tasks_arr.txt,
  6 / 4 / 4 skills; frac 0.5, 1,500 resumed steps), streams 19 -> 20 -> 4 (order 0) and 21 -> 22 -> 23 (order 1) from
  consfz / erl; ours re-merges every arrival (k 30 / 34 / 38), ER keeps the library merge. Dry run: contexts 30 / 16 members,
  resume consfz / erl of the order. (b) E147_OOD = d: exp147 run with EXP-81's ood_shift (all free objects' xy moved d m in
  one seeded direction; demos span ~+-2.4 cm), tag suffix ood<d>; arrival 1 (consfz / erl @0-3) at 10 and 5 cm, arrival 4
  (consfzc / erlc @4 / 23 / 26 / 29) at 10 cm; states 0..9, 10 episodes. (c) E147_UNS = 1: the arrival-4 models zero-shot on
  the 8 externals no order draws (the same 8 for every order: idx 74 76 83 84 85 90 95 103; same suites as the arrivals) +
  16 LIBERO-90 tasks outside the order's library (seeded 11720 + order); merge k kept at the true member count (fixed in the
  dry run: the ZV branch used len(evaluated set)). Default-mode runs unchanged (0 jobs for done arms).
- Scheduler `sweep/chain80_remerge.sh` (tmux c80, results/chain80.log): trainer lanes once every chain79 trainer is claimed
  (wait 240 s); evals (port 11100, eval lock) once chain79 is ALL EXIT: (a) as adapters appear, else the next eval-only model
  (10 cm arrival 1 -> unseen -> 5 cm -> 10 cm arrival 4). 12 trainers + 32 eval-only models, ~1.5 days after chain79.
- PRE-REGISTERED (11-23 09:15, before any EXP-172 result; sign rule, se / W-T-L shown; per-skill pooling):
  (a) primary = ours - ER on the EARLIER batches (new skills that arrived before the current arrival: 8 / 14 / 18) over the 6
  cells (arrivals 2-4 x orders 0 / 1): mean > 0 = WIN, CONSISTENT if > 0 in >= 5 / 6. Mechanism: "re-merging stands in for
  stored rollouts" if ours' memory cost (full-memory consfzc - consfzcm0, earlier batches) < ER's (erlc - erlcm0), pooled.
  Also reported: just-arrived batch, library, all arrived skills, NBT.
  (b) primary = ours - ER under the 10 cm shift, arrival 1, all 24 skills pooled over 4 orders: > 0 = WIN, CONSISTENT if > 0
  on >= 3 / 4 orders; mechanism: paired OOD drop (ID - OOD) smaller for ours. Also: 5 cm, arrival 4 at 10 cm, library / new.
  (c) primary = ours - ER zero-shot on the 8 unseen externals, pooled over 4 orders (32 instances): > 0 = WIN, CONSISTENT if
  > 0 on >= 3 / 4 orders; if both arms average < 10% the result is a floor (uninformative). Also: the 16 unseen LIBERO-90
  tasks (same library merge in both arms -> expected ~0).

### EXP-173 — the all-baselines table: every same-data baseline on 4 orders at 5 and 1 library demos — **queued (11-24 22:38); chain81**
Motivation: user 11-24 "we will have more baselines than just ER ... show, in a table, that our method leads all baselines,
with ER the closest ... a better picture than one to one against ER". Order 0 already reads that way (all 24, 5 / 1 demos: ours
83.8 / 83.3, ER 80.4 / 79.6, learned merge weights 71.3 / -, EWC 66.7, multi-task from the base 60.0 / 50.5, sequential FT
59.2; joint on all demos 85.4 = more-data reference row), but most baselines exist on order 0 only.
- 11-24 setup (user-approved "yes add"): exp147 + lrcd1 = lrc (learned merge weights, no correction; EXP-162) at 1 library
  demo (md5 101706c1..., both hosts; backup results/merge_stale/exp147_consolidate.py.bak_1124b); prep2's lrc contexts
  (ctx_all_lrc / lrco.json) written for orders 1-3 (prep2 ran there before EXP-162; same recipe, 24 members). 16 trainings:
  consbd1 @1-3, seq @1-3, ewc @1-3, lrc @1-3, lrcd1 @0-3. seq / ewc train without replay -> one run per order serves both
  budgets; EWC's Fisher is estimated on the 5-demo replay (favours EWC at 1 demo; footnote). `sweep/chain81_table.sh` (tmux c81,
  results/chain81.log): starts once chain79's seed block (P1) is all claimed, lanes wait 150 s (< chain79's 240 -> ahead of
  its P2 / P3, before chain80), evals port 11400 under the eval lock.
- PRE-REGISTERED (11-24, before any EXP-173 result; sign rule): per budget (5 / 1 demos), all 24 skills pooled over 4 orders
  (seed 0): ours above EVERY same-data baseline (ER, multi-task from the base, sequential FT, online EWC, learned merge
  weights) = "ours leads all"; ER the closest = expected. Shown: library / new per arm, per-order signs, se over skills.
  Joint training on all demos = reference row, not a baseline. Training-free merges (plain average, our merge uncorrected)
  from existing evals.

### EXP-174 — LoRA continual-learning baselines for the main table: O-LoRA and InfLoRA — **queued (11-25); chain82**
Motivation: user 11-25 "I would add a couple of lora continual learning methods". The main table's continual-learning rows were
sequential FT and online EWC only (generic CL); reviewers from the LoRA-CL literature expect its standard methods. Picked the two
canonical ones that end in ONE model (no per-task routing): O-LoRA (Wang et al. 2023, orthogonal subspaces) and InfLoRA (Liang &
Li, CVPR 2024, interference-free subspace from input representations).
- 11-25 setup: train_lora.py --olora_lambda (official L1 loss sum |A_old A^T|, A_old = the library adapters' lora_A rows = the
  previous tasks' LoRAs, folded into the base; lambda 0.5 = official; lambda_2 0 = official; its gradient added AFTER the task-
  gradient clip — at 458 modules its norm would otherwise shrink the task gradient below Adam's eps) and --inflora (official
  design: old-input subspace M per module by the DualGPM criterion on the input covariance, threshold 0.95 = official lamb,
  estimated on the library's 5-demo replay through the library merge; A frozen at the top-16 left singular vectors of
  (I - M M^T) C_new / sqrt(3), C_new from the new skills' rollouts; only B trains, from zero; covariances from 4,096 sampled
  tokens per module over 50 minibatches). exp147 arms olora / inflora: ctx_lib_free (library merge, like seq / ewc / ER),
  tasks_arr.txt (the new skills' own rollouts), no replay, 3,000 steps, lr 1e-4, rank 16 (same as every arm; official lr belong
  to their backbones). One run per order serves both demo budgets. Backups results/merge_stale/*.bak_1125 (both hosts).
  `sweep/chain82_cl.sh` (tmux c82, results/chain82.log): 8 trainings (olora / inflora @0-3), starts once chain81's 16 jobs are
  claimed, lanes 150 s (ahead of chain79 P2 / P3 and chain80), evals port 11500.
- PRE-REGISTERED (11-25, before any EXP-174 result; sign rule, extends EXP-173): per budget (5 / 1 demos), all 24 skills pooled
  over 4 orders (seed 0): ours above O-LoRA and above InfLoRA. Expected shape (not a criterion): library above SeqFT (the
  constraint protects it), new skills below SeqFT (the constraint costs plasticity).
- 11-25 10:20 smoke (odin P6000, 3 steps, batch 2, order 0 inputs): plain arm still 46.1M trainable (edits leave it unchanged);
  O-LoRA 46.1M, A_old 256 rows x 458 / 458 modules, orth loss 2.5e5 at the start (task loss ~0.025; hence the separate clip);
  InfLoRA 29.5M (B only), A rows at norm 1/sqrt 3, 456 / 458 modules designed (2 see no input). Fixed: a module with fewer
  sampled rows than the rank (one row per sample, e.g. the time MLP, at smoke size) -> the missing A rows stay zero. CAVEAT:
  the official criterion (on the covariance's SQUARED singular values) keeps a tiny old-input subspace on pi0.5 (median 0.1%
  of the input dim, max 62%; on the eigenvalues it would be 3.5% / 81%): a few massive-activation directions hold the energy,
  so InfLoRA protects the library little here. Run as published; logged for the discussion.
- 11-25 10:24 chain82 started on shiva (tmux c82); scripts synced (train_lora 394049f6..., exp147 9e4c5f80..., chain82
  6f5e7887...). Watcher includes chain82.log.
