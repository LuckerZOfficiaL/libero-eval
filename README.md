# libero-eval

Minimal `uv` repo for running LIBERO MuJoCo evaluation with the OpenPI PyTorch `pi05_libero` inference path.

## Included

- LIBERO evaluation client: `examples/libero/main.py`
- Websocket policy server: `scripts/serve_policy.py`
- Minimal OpenPI PyTorch runtime for `pi05_libero`
- `openpi_client` for the websocket protocol
- A PEFT-style LoRA application utility: `scripts/apply_lora.py`

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

Install LIBERO and the extra benchmark dependencies:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/libero-local
uv pip install -e /tmp/libero-local
uv pip install bddl easydict cloudpickle gym==0.26.2
printf '%s\n' /tmp/libero-local > .venv/lib/python3.11/site-packages/libero_local_source.pth
```

Notes:

- On this machine, `uv pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git` created package metadata but did not expose an importable `libero` module. Installing from a local clone in editable mode and adding the `.pth` file above worked around that packaging issue.
- `examples/libero/main.py` now rewrites the LIBERO path config at startup so it points at the installed package instead of a stale `~/.libero/config.yaml`.
- The client also includes a PyTorch 2.6+ compatibility shim for LIBERO init-state files, because upstream LIBERO still calls `torch.load(...)` without setting `weights_only=False`.

Install the OpenPI transformer overrides into the environment:

```bash
cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## Run evaluation

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

If MuJoCo EGL fails:

```bash
MUJOCO_GL=glx uv run python examples/libero/main.py
```

Compatibility fixes included in this repo:

- `src/openpi/serving/websocket_policy_server.py` now runs `policy.infer(...)` in a worker thread so the websocket event loop can still service pings and handshakes during long inference calls.
- `src/openpi_client/websocket_client_policy.py` disables websocket ping timeouts on the client side, which avoids spurious disconnects when inference is slow.
- `src/openpi/runtime.py` only unnormalizes the leading LIBERO action dimensions covered by the norm stats, instead of trying to apply 7-D stats to the full 32-D model output.

More LIBERO-specific notes are in [examples/libero/README.md](/media/ssd/usr/mehdi/libero-eval/examples/libero/README.md).

## Custom checkpoints

To serve a local checkpoint directory containing `model.safetensors` and `assets/<asset_id>/norm_stats.json`:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir
```

If the norm stats live somewhere else, pass an explicit override. The value can be either the
directory that contains `norm_stats.json` or the `norm_stats.json` file itself:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /path/to/checkpoint_dir \
  --policy.norm-stats-path /path/to/norm_stats_dir
```

For the applied checkpoint created in this workspace:

```bash
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /media/pinas/mehdi/libero90_lora/task0/merged_model
```

## Apply a LoRA adapter

> Terminology: **apply** = fuse a *single* LoRA adapter into the base weights (this section).
> **merge** is reserved for the research operation of combining *multiple* LoRAs into one multi-task
> policy (not implemented yet). See `AGENTS.md`.

The apply script is modeled after the PEFT-based workflow used in `dragdrop-vlas`.

Inputs:

- base model directory containing `model.safetensors`, or a direct path to `model.safetensors`
- LoRA checkpoint directory containing `adapter_model.safetensors` and `adapter_config.json`, or a direct path to the adapter safetensors file
- output directory for the applied checkpoint

Usage:

```bash
uv run python scripts/apply_lora.py \
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

If your LoRA directory uses a different filename such as `task0_adapter.safetensors`, either:

- rename or symlink it to `adapter_model.safetensors` before running apply with the directory path, or
- pass the adapter `.safetensors` file path directly instead of the directory.

The apply command used for the local checkpoint was:

```bash
ln -sfn task0_adapter.safetensors /media/pinas/mehdi/libero90_lora/task0/adapter_model.safetensors
uv run python scripts/apply_lora.py \
  --base-model-dir /media/pinas/Chris/models/pi05_base_pytorch/model.safetensors \
  --lora-checkpoint-dir /media/pinas/mehdi/libero90_lora/task0 \
  --output-dir /media/pinas/mehdi/libero90_lora/task0/merged_model
mkdir -p /media/pinas/mehdi/libero90_lora/task0/merged_model/assets/physical-intelligence/libero
cp /media/pinas/mehdi/libero90_lora/assets/libero90_norm_stats.json \
  /media/pinas/mehdi/libero90_lora/task0/merged_model/assets/physical-intelligence/libero/norm_stats.json
rm /media/pinas/mehdi/libero90_lora/task0/adapter_model.safetensors
```
