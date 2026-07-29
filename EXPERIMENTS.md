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
