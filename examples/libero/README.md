# LIBERO Evaluation

This directory runs the LIBERO MuJoCo benchmark against the slim `libero-eval` policy server.

## Prerequisites

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
UV_CACHE_DIR=/media/ssd/usr/mehdi/.cache/uv uv sync
```

Install LIBERO separately:

```bash
uv pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Install the OpenPI transformer overrides:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Run

Terminal 1:

```bash
source .venv/bin/activate
python scripts/serve_policy.py --env LIBERO
```

Terminal 2:

```bash
source .venv/bin/activate
python examples/libero/main.py
```

If EGL fails:

```bash
MUJOCO_GL=glx python examples/libero/main.py
```

## Custom checkpoints

```bash
python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir
```
