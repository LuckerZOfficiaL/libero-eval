# GUIDE — running multi-LoRA merge experiments

Pairwise LoRA **merging** (EXP-3): combine two task adapters into one policy and check it still solves
both tasks. Compared against the single-task **baseline** (EXP-2, mean 94.4%). Policy: pairwise,
**10 rollouts/eval**, staged **1 pair → 5 pairs → full set**.

> "apply" (single adapter → base) is a different op — see `AGENTS.md`. Here we always **merge** ≥2.

---

## Quick start — one command

```bash
bash sweep/run.sh
```
With no arguments this runs **exactly what's in [`configs/merge.yaml`](configs/merge.yaml)** — set
everything there and just launch. Any flag you add overrides the YAML for that run. Launches in a
persistent tmux session `merge`. Scheduling: up to **4 merges run at once (2 per P6000); extra
(pair, method) jobs queue**.

Override any default with flags:
```bash
bash sweep/run.sh --pairs 3:4 --method ties
bash sweep/run.sh --pairs 1:2,5:6 --method average --trials 20
bash sweep/run.sh --scope 5 --method task_arithmetic          # first 5 disjoint pairs
bash sweep/run.sh --scope all --workers 4                     # full set (38 pairs)
```

Or set everything in a YAML file and just launch:
```bash
bash sweep/run.sh --config configs/merge.yaml
```
Edit [`configs/merge.yaml`](configs/merge.yaml) — `pairs`/`scope`, `method`, `params`, `eval_tasks`, `trials`, `workers`.

### Sweep (cross product)
`--pairs` and `--method` both accept comma-separated **lists**; the run is their **cross product**
(every pair × every method), all in one parallel/resumable pass, one result file per combo:
```bash
bash sweep/run.sh --pairs "1:2,3:4,5:6" --method "task_arithmetic,ties,average"   # 3x3 = 9 jobs
```
→ `results/merge/task_1_2_task_arithmetic.jsonl`, `…task_1_2_ties.jsonl`, … (9 files).
In YAML: `pairs: "1:2,3:4"` and `method: [task_arithmetic, ties]`. Use `--workers 4` for big sweeps.

---

## The knobs

| Knob | Flag | YAML key | Default |
|---|---|---|---|
| Tasks to **merge** | `--pairs 1:2` (or `1:2,3:4`) | `pairs` | `1:2` |
| …or a scope | `--scope 5` / `--scope all` | `scope` | — |
| **Method** | `--method ties` | `method` | `task_arithmetic` |
| Method **params** | `--params '{"density":0.3}'` | `params` (per-method blocks) | see below |
| Tasks to **eval** | `--eval-tasks auto` (or `3,4` / scene names) | `eval_tasks` | `auto` |
| **Rollouts**/eval | `--trials 10` | `trials` | `10` |
| Parallel workers | `--workers 4` | `workers` | `2` |

**Methods & their hyperparameters:**
- `task_arithmetic` — ΔW = λ·Σ deltas. Params: `{lambda: 1.0}`.
- `average` — mean of deltas. No params.
- `ties` — trim each delta to its top-`density` entries, elect a per-param sign, merge the agreeing
  ones. Params: `{density: 0.2, lambda: 1.0}`.

In YAML, hyperparameters are **per-method blocks** — keep them all and just flip `method`:
```yaml
method: ties
params:
  task_arithmetic: {lambda: 1.0}
  ties: {density: 0.2, lambda: 1.0}
```
Defaults are kept out of the file name; changing one tags it (`task_1_2_ties_density0.3`).

**Tasks to evaluate on (`eval_tasks`):**
- `auto` (default) — the pair's own scenes (in-distribution **retention**).
- task indices — `--eval-tasks 5,6` or `eval_tasks: [5, 6]` → those tasks' scenes (held-out **transfer**).
- explicit scene names — `eval_tasks: ["KITCHEN_SCENE7_open_the_microwave"]`.

`task_index → scene` map: `sweep/plan.json`.

---

## Where the results are

One file **per pair + method combination**:
```
results/merge/task_<a>_<b>_<method>.jsonl        e.g. results/merge/task_1_2_ties.jsonl
```
Each row = one `(pair, scene)` eval: `{job, method, params, adapters, scene, success_rate, n_success, n_trials, ...}`.
Rollout videos: `results/merge_videos/<job>/<scene>/*.mp4`.

### Check progress (like `health.sh`)
```bash
bash sweep/merge_health.sh          # progress, servers, GPUs, per-pair results so far
tmux attach -t merge                # live feed  (detach: Ctrl-b then d)
```

### Read / compare results
```bash
python sweep/merge_summary.py       # every merged (pair,scene) vs its single-task baseline + retention
```

### Stop a run
```bash
bash sweep/stop.sh                  # guaranteed clean kill (orchestrator + build + servers + clients), frees GPUs/shm
```
Or **Ctrl+C** in the `merge` tmux pane (usually fine — everything is in that pane's process group).
Or just launch the next experiment — `bash sweep/run.sh ...` **auto-stops** the previous run first.
Completed `(pair,scene)` results in `results/merge/` are kept; a re-run resumes and skips them.

---

## Staged workflow (the policy)

```bash
# Stage 1 — one pair
bash sweep/run.sh --pairs 1:2 --method task_arithmetic
python sweep/merge_summary.py        # retain ~baseline? -> method holds for a pair

# Stage 2 — five pairs
bash sweep/run.sh --scope 5 --method task_arithmetic
python sweep/merge_summary.py        # still high across pairs? -> robust

# Stage 3 — full set
bash sweep/run.sh --scope all --workers 4 --method task_arithmetic
python sweep/merge_summary.py        # full retention curve — the number to report
```

Runs are **resumable**: results accumulate in `results/merge/`, and re-running skips `(pair, scene)`
evals already recorded. Different methods/params on the same pair get different file names, so they
never overwrite each other — `merge_summary.py` shows them side by side.

---

## Files

| Path | What |
|---|---|
| `sweep/run.sh` | one command: generate plan + launch |
| `configs/merge.yaml` | editable config (alternative to flags) |
| `sweep/make_merge_plan.py` | plan generator (flags/YAML → `sweep/merge_plan.json`) |
| `sweep/merge_loras.py` | the merge itself (N adapters + method → applied checkpoint) |
| `sweep/merge_orchestrator.py` | build → serve → eval → record, parallel + resumable |
| `sweep/merge_health.sh` | progress/health check |
| `sweep/merge_summary.py` | merged-vs-baseline retention table |
| `results/merge/` | per-pair-method result files |
| `results/baseline_summary.md` | single-task baseline (the comparison bar) |
