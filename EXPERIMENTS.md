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

## EXP-3 · Multi-LoRA merging methods 🔵 running (scoping + sanity)

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

**Result (in progress).** On this pair, **average >> ties >> task_arithmetic(lambda=1)**. Mechanistic
read: average = task_arithmetic at lambda=0.5; summing both full task deltas (lambda=1) over-shoots and
tasks interfere (microwave collapses to 0%). **The merge scale lambda is the dominant knob.** Next: sweep
task_arithmetic lambda (e.g. 0.3/0.5/0.7) to confirm, then advance average/ties to 5 pairs.
