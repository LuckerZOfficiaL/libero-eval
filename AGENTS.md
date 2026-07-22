# AGENTS.md

## Purpose

This repository is used to run LIBERO simulations against a served OpenPI policy and to experiment with LoRA-based adaptation for LIBERO tasks.

The current practical workflow is:

1. start from a PI0.5 PyTorch backbone
2. apply a single-task LoRA adapter onto that backbone
3. serve the resulting merged model
4. evaluate selected LIBERO tasks in simulation

The larger research goal is not implemented here yet: develop model-merging methods that combine multiple task-specific LoRAs into one policy that can solve multiple tasks. At the moment, the LoRAs are single-task.

## Key Files

- [examples/libero/main.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/main.py): LIBERO evaluation entrypoint.
- [examples/libero/task_suite_task_map.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/task_suite_task_map.py): repo-local task selection map. Comment out tasks here to exclude them from evaluation.
- [examples/libero/libero90_tasks.jsonl](/media/ssd/usr/mehdi/libero-eval/examples/libero/libero90_tasks.jsonl): mapping between a `libero_90` task and its LoRA ID / task index.
- [scripts/apply_lora.py](/media/ssd/usr/mehdi/libero-eval/scripts/apply_lora.py): applies a LoRA adapter to the base PyTorch model and writes a merged checkpoint.
- [scripts/serve_policy.py](/media/ssd/usr/mehdi/libero-eval/scripts/serve_policy.py): websocket policy server used by the simulator client.
- [src/openpi/runtime.py](/media/ssd/usr/mehdi/libero-eval/src/openpi/runtime.py): runtime loader for LIBERO inference. Expects `model.safetensors` and `norm_stats.json`.
- [examples/libero/README.md](/media/ssd/usr/mehdi/libero-eval/examples/libero/README.md): LIBERO-specific local notes.

## External Assets

- Base backbone: `/media/pinas/Chris/models/pi05_base_pytorch/model.safetensors`
- Task-specific LoRAs: `/media/pinas/mehdi/libero90_lora`
- Local merged checkpoint example: `/media/pinas/mehdi/libero90_lora/task0/merged_model`
- Dedicated LIBERO-90 norm stats file: `/media/pinas/mehdi/libero90_lora/assets/libero90_norm_stats.json`

Do not assume these assets are copied into this repo. This codebase references them as external filesystem dependencies.

## Environment Setup

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
UV_CACHE_DIR=/media/ssd/usr/mehdi/.cache/uv uv sync
```

Install LIBERO and runtime dependencies:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/libero-local
uv pip install -e /tmp/libero-local
uv pip install bddl easydict cloudpickle gym==0.26.2
printf '%s\n' /tmp/libero-local > .venv/lib/python3.11/site-packages/libero_local_source.pth
```

Install the OpenPI transformer overrides:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Repo Conventions For LIBERO Evaluation

### Task selection

This repo now owns task selection locally through:

- [examples/libero/task_suite_task_map.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/task_suite_task_map.py)

To evaluate only a subset of tasks in a suite, comment out the unwanted entries in the corresponding list. The evaluator only runs uncommented tasks.

Example:

```python
"libero_90": [
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    # "KITCHEN_SCENE7_open_the_microwave",
]
```

### Task-to-LoRA mapping

For `libero_90`, the mapping between the human-readable task and the task-specific LoRA ID is stored in:

- [examples/libero/libero90_tasks.jsonl](/media/ssd/usr/mehdi/libero-eval/examples/libero/libero90_tasks.jsonl)

Use this file when selecting which LoRA checkpoint to apply for a specific evaluation task.

## Checkpoint Format Expectations

The served policy runtime does not consume the raw OpenPI orbax/JAX checkpoint layout. It expects a PyTorch checkpoint directory with:

```text
<checkpoint_dir>/
  model.safetensors
  assets/physical-intelligence/libero/norm_stats.json
```

This matters because:

- `scripts/serve_policy.py --env LIBERO` currently points at `gs://openpi-assets/checkpoints/pi05_libero`
- that default download does not contain `model.safetensors`
- so serving the default LIBERO checkpoint path is not sufficient for this workflow

In practice, serve a local merged PyTorch checkpoint via `policy:checkpoint`.

## Applying A Single-Task LoRA

Use [scripts/apply_lora.py](/media/ssd/usr/mehdi/libero-eval/scripts/apply_lora.py) to apply a task-specific LoRA to the base backbone.

Typical inputs:

- base model: `/media/pinas/Chris/models/pi05_base_pytorch/model.safetensors`
- LoRA checkpoint directory: something under `/media/pinas/mehdi/libero90_lora/taskX`
- output directory: a local merged checkpoint directory

Example:

```bash
ln -sfn task0_adapter.safetensors /media/pinas/mehdi/libero90_lora/task0/adapter_model.safetensors
uv run python scripts/apply_lora.py \
  --base-model-dir /media/pinas/Chris/models/pi05_base_pytorch/model.safetensors \
  --lora-checkpoint-dir /media/pinas/mehdi/libero90_lora/task0 \
  --output-dir /media/pinas/mehdi/libero90_lora/task0/merged_model
```

If the output checkpoint does not already contain the intended LIBERO norm stats, place them under the expected path. For LIBERO-90:

```bash
mkdir -p /media/pinas/mehdi/libero90_lora/task0/merged_model/assets/physical-intelligence/libero
ln -sfn /media/pinas/mehdi/libero90_lora/assets/libero90_norm_stats.json \
  /media/pinas/mehdi/libero90_lora/task0/merged_model/assets/physical-intelligence/libero/norm_stats.json
```

## Starting The Policy Server

Serve the merged model with an explicit checkpoint path:

```bash
source .venv/bin/activate
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /media/pinas/mehdi/libero90_lora/task0/merged_model
```

If you want to override norm stats via a separate directory, the path must be a directory that directly contains a file literally named `norm_stats.json`.

Example:

```bash
mkdir -p /tmp/libero90_norm_stats
ln -sfn /media/pinas/mehdi/libero90_lora/assets/libero90_norm_stats.json /tmp/libero90_norm_stats/norm_stats.json

uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /media/pinas/mehdi/libero90_lora/task0/merged_model \
  --policy.norm-stats-path /tmp/libero90_norm_stats
```

Expected success signal:

```text
INFO:websockets.server:server listening on 0.0.0.0:8000
```

## Running LIBERO Evaluation

In a second terminal:

```bash
source .venv/bin/activate
uv run python examples/libero/main.py --args.task-suite-name libero_90
```

Notes:

- The CLI flag is `--args.task-suite-name`, not `--task-suite-name`.
- Rollout videos are saved by default under [data/libero/videos](/media/ssd/usr/mehdi/libero-eval/data/libero/videos).
- If EGL fails, try:

```bash
MUJOCO_GL=glx uv run python examples/libero/main.py --args.task-suite-name libero_90
```

## Known Gotchas

- `scripts/serve_policy.py --env LIBERO` downloads a checkpoint that is incompatible with the PyTorch runtime expected by this repo.
- `src/openpi/shared/normalize.py` expects a directory containing `norm_stats.json`; passing an arbitrary differently named JSON file path does not work.
- `examples/libero/main.py` currently evaluates all uncommented tasks in the chosen suite. There is no separate per-task CLI selector yet.
- The LoRAs currently available are single-task. Multi-task merging is the open research direction, not a finished feature here.

## Recommended Workflow For A Coding Agent

When asked to work on LIBERO evaluation or LoRA merging in this repo:

1. Read [examples/libero/main.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/main.py), [examples/libero/task_suite_task_map.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/task_suite_task_map.py), [scripts/apply_lora.py](/media/ssd/usr/mehdi/libero-eval/scripts/apply_lora.py), and [src/openpi/runtime.py](/media/ssd/usr/mehdi/libero-eval/src/openpi/runtime.py) first.
2. Confirm whether the user is working on evaluation plumbing, LoRA application, or new model-merging research code.
3. For evaluation tasks, verify the selected suite and which tasks are commented out in the local mapping.
4. For LoRA tasks, use [examples/libero/libero90_tasks.jsonl](/media/ssd/usr/mehdi/libero-eval/examples/libero/libero90_tasks.jsonl) to identify the correct single-task adapter.
5. Verify checkpoint layout before serving: `model.safetensors` plus `assets/physical-intelligence/libero/norm_stats.json`.
6. Prefer explicit local checkpoint serving over the default `--env LIBERO` path.

## Current Research Direction

The intended next step for this codebase is to go beyond single-task LoRA application and implement novel merging methods that combine multiple task-specific adapters into one model that can perform multiple LIBERO tasks. Any new implementation work in this area should preserve the current evaluation loop so merged models can be tested under the same simulator setup.
