# libero-eval

Minimal `uv` repo for running LIBERO MuJoCo evaluation with the OpenPI PyTorch `pi05_libero` inference path.

## Included

- LIBERO evaluation client: `examples/libero/main.py`
- Websocket policy server: `scripts/serve_policy.py`
- Minimal OpenPI PyTorch runtime for `pi05_libero`
- `openpi_client` for the websocket protocol
- A PEFT-style LoRA merge utility: `scripts/merge_lora.py`

## Not included

- Training code
- Dataset conversion scripts
- Other robot environments
- A vendored `third_party/libero` checkout

## Setup

Create the environment from the repo root:

```bash
cd /media/ssd/usr/mehdi/libero-eval
uv venv --python 3.11
source .venv/bin/activate
mkdir -p /media/ssd/usr/mehdi/.cache/uv
UV_CACHE_DIR=/media/ssd/usr/mehdi/.cache/uv uv sync
```

The explicit `UV_CACHE_DIR` matters on machines where the default cache lives on a `noexec` mount.

Install LIBERO separately:

```bash
uv pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Install the OpenPI transformer overrides into the environment:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Run evaluation

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

If MuJoCo EGL fails:

```bash
MUJOCO_GL=glx python examples/libero/main.py
```

More LIBERO-specific notes are in [examples/libero/README.md](/media/ssd/usr/mehdi/libero-eval/examples/libero/README.md).

## Custom checkpoints

To serve a local checkpoint directory containing `model.safetensors` and `assets/<asset_id>/norm_stats.json`:

```bash
python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir
```

## Merge a LoRA checkpoint

The merge script is modeled after the PEFT-based workflow used in `dragdrop-vlas`.

Inputs:

- base model directory containing `model.safetensors`, or a direct path to `model.safetensors`
- LoRA checkpoint directory containing `adapter_model.safetensors` and `adapter_config.json`, or a direct path to the adapter safetensors file
- output directory for the merged checkpoint

Usage:

```bash
python scripts/merge_lora.py \
  --base-model-dir /path/to/base_model_dir \
  --lora-checkpoint-dir /path/to/lora_checkpoint_dir \
  --output-dir /path/to/output_dir
```

Optional flags:

- `--action-horizon 10`
- `--discrete-state-input`
- `--dtype bfloat16`

Output:

- `output_dir/model.safetensors`
- `output_dir/adapter_config.json`
- `output_dir/assets/` if present in either input directory
