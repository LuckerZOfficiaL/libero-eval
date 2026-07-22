# LIBERO Evaluation

This directory runs the LIBERO MuJoCo benchmark against the slim `libero-eval` policy server.

## Prerequisites

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
UV_CACHE_DIR=/media/ssd/usr/mehdi/.cache/uv uv sync
```

Install LIBERO and the dependencies it needs at runtime:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/libero-local
uv pip install -e /tmp/libero-local
uv pip install bddl easydict cloudpickle gym==0.26.2
printf '%s\n' /tmp/libero-local > .venv/lib/python3.11/site-packages/libero_local_source.pth
```

The client in this repo now patches over several LIBERO compatibility issues automatically:

- it resets LIBERO's config path to the installed package location instead of relying on a stale `~/.libero/config.yaml`
- it forces the old `torch.load(..., weights_only=False)` behavior for LIBERO init-state files on PyTorch 2.6+
- it uses a websocket client configuration that tolerates long inference calls

Install the OpenPI transformer overrides:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Run

Terminal 1:

```bash
source .venv/bin/activate
uv run python scripts/serve_policy.py --env LIBERO
```

Terminal 2:

```bash
source .venv/bin/activate
uv run python examples/libero/main.py
```

If EGL fails:

```bash
MUJOCO_GL=glx uv run python examples/libero/main.py
```

## Selecting Tasks

The evaluator now uses the repo-local mapping file
[task_suite_task_map.py](/media/ssd/usr/mehdi/libero-eval/examples/libero/task_suite_task_map.py)
to decide which tasks to run inside each LIBERO suite.

Comment out any entries you want to skip, then run the suite as usual:

```python
"libero_goal": [
    "open_the_middle_drawer_of_the_cabinet",
    # "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
]
```

The `--task-suite-name` flag still chooses the suite, and the local mapping chooses the subset of tasks within it.

## Custom checkpoints

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir
```

If the norm stats are not under `assets/physical-intelligence/libero/norm_stats.json`, pass:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir \
  --policy.norm-stats-path /path/to/norm_stats_dir
```
