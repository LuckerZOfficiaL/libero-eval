from __future__ import annotations

import argparse
import dataclasses
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import safetensors.torch
import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import PretrainedConfig

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.runtime import Pi05LiberoConfig


@dataclass
class LoRAGroup:
    stem: str
    a: torch.Tensor | None = None
    b: torch.Tensor | None = None
    alpha: float | None = None
    source_keys: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a PI0.5 base model.")
    parser.add_argument(
        "--base-model-dir",
        required=True,
        type=Path,
        help="Base model directory containing `model.safetensors`, or a direct path to `model.safetensors`.",
    )
    parser.add_argument(
        "--lora-checkpoint-dir",
        required=True,
        type=Path,
        help="LoRA checkpoint directory containing `adapter_model.safetensors` and `adapter_config.json`, or a direct path to an adapter `.safetensors` file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the merged model should be written.",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=10,
        help="Action horizon for the PI0.5 model config. Default: 10.",
    )
    parser.add_argument(
        "--discrete-state-input",
        action="store_true",
        help="Enable discrete state input in the PI0.5 model config.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Model dtype to use while loading and merging. Default: bfloat16.",
    )
    parser.add_argument(
        "--default-lora-scale",
        type=float,
        default=1.0,
        help="Fallback LoRA scale used by raw safetensors merge when no adapter config is available.",
    )
    return parser.parse_args()


def _looks_like_safetensors(path: Path) -> bool:
    return path.suffix == ".safetensors"


def _suggest_matching_files(path: Path) -> str | None:
    if not _looks_like_safetensors(path):
        return None
    parent = path.parent
    if not parent.exists():
        return None
    matches = sorted(parent.rglob(path.name))
    if not matches:
        return None
    preview = "\n".join(str(match) for match in matches[:10])
    more = "" if len(matches) <= 10 else f"\n... and {len(matches) - 10} more"
    return f"Path does not exist: {path}\nFound files with the same name under {parent}:\n{preview}{more}"


def resolve_base_model_path(base_model: Path) -> tuple[Path, Path]:
    if _looks_like_safetensors(base_model):
        if not base_model.exists():
            raise FileNotFoundError(_suggest_matching_files(base_model) or f"Base model weights not found: {base_model}")
        if base_model.name != "model.safetensors":
            raise ValueError(f"Base model file must be named `model.safetensors`, got: {base_model}")
        return base_model.parent, base_model

    base_model_weights_path = base_model / "model.safetensors"
    if not base_model_weights_path.exists():
        raise FileNotFoundError(f"Base model weights not found: {base_model_weights_path}")
    return base_model, base_model_weights_path


def resolve_lora_paths(lora_checkpoint: Path) -> tuple[Path, Path, Path | None]:
    if _looks_like_safetensors(lora_checkpoint):
        if not lora_checkpoint.exists():
            raise FileNotFoundError(_suggest_matching_files(lora_checkpoint) or f"LoRA adapter weights not found: {lora_checkpoint}")
        lora_dir = lora_checkpoint.parent
        adapter_model_path = lora_checkpoint
    else:
        lora_dir = lora_checkpoint
        adapter_model_path = lora_checkpoint / "adapter_model.safetensors"
        if not adapter_model_path.exists():
            raise FileNotFoundError(f"LoRA adapter weights not found: {adapter_model_path}")

    adapter_config_path = lora_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        adapter_config_path = None

    return lora_dir, adapter_model_path, adapter_config_path


def build_model_config(args: argparse.Namespace) -> Pi05LiberoConfig:
    return Pi05LiberoConfig(
        action_horizon=args.action_horizon,
        discrete_state_input=args.discrete_state_input,
        dtype=args.dtype,
    )


def attach_hf_config(model: PI0Pytorch, model_cfg: Pi05LiberoConfig) -> None:
    config_dict = dataclasses.asdict(model_cfg)
    if "tie_word_embeddings" not in config_dict:
        config_dict["tie_word_embeddings"] = False
    model.config = PretrainedConfig.from_dict(config_dict)


def copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def extract_lora_groups(lora_state: dict[str, torch.Tensor]) -> dict[str, LoRAGroup]:
    groups: dict[str, LoRAGroup] = {}
    for key, value in lora_state.items():
        parsed = parse_lora_key(key)
        if parsed is None:
            continue
        stem, kind = parsed
        group = groups.setdefault(stem, LoRAGroup(stem=stem))
        group.source_keys.append(key)
        if kind == "a":
            group.a = value
        elif kind == "b":
            group.b = value
        elif kind == "alpha":
            group.alpha = float(value.item())
    return groups


def parse_lora_key(key: str) -> tuple[str, str] | None:
    patterns = [
        (".lora_A.weight", "a"),
        (".lora_B.weight", "b"),
        (".lora_down.weight", "a"),
        (".lora_up.weight", "b"),
        (".alpha", "alpha"),
        (".lora_alpha", "alpha"),
    ]
    for suffix, kind in patterns:
        if key.endswith(suffix):
            return key[: -len(suffix)], kind
    return None


def resolve_base_key(base_state: dict[str, torch.Tensor], stem: str) -> str | None:
    candidates = [stem, f"{stem}.weight"]
    prefixes = ["base_model.model.", "base_model.", "model."]
    for prefix in prefixes:
        if stem.startswith(prefix):
            trimmed = stem[len(prefix) :]
            candidates.extend([trimmed, f"{trimmed}.weight"])
    for candidate in candidates:
        if candidate in base_state:
            return candidate
    return None


def infer_rank(a: torch.Tensor) -> int:
    if a.ndim != 2:
        raise ValueError(f"Only 2D LoRA matrices are supported, got shape {tuple(a.shape)}")
    return min(a.shape)


def compute_delta(base: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Only 2D LoRA matrices are supported, got A={tuple(a.shape)} B={tuple(b.shape)}")
    if base.ndim < 2:
        raise ValueError(f"Target tensor must be at least 2D, got shape {tuple(base.shape)}")

    out_dim = base.shape[0]
    in_dim = math.prod(base.shape[1:])
    target_shape = (out_dim, in_dim)

    candidates: list[torch.Tensor] = []
    if (b.shape[0], a.shape[1]) == target_shape and b.shape[1] == a.shape[0]:
        candidates.append(b @ a)
    if (a.shape[0], b.shape[1]) == target_shape and a.shape[1] == b.shape[0]:
        candidates.append(a @ b)
    if (a.shape[1], b.shape[0]) == target_shape and a.shape[0] == b.shape[1]:
        candidates.append((b @ a).T)
    if (b.shape[1], a.shape[0]) == target_shape and b.shape[0] == a.shape[1]:
        candidates.append((a @ b).T)

    if not candidates:
        raise ValueError(
            f"Could not infer LoRA multiplication order for base={tuple(base.shape)} A={tuple(a.shape)} B={tuple(b.shape)}"
        )
    return candidates[0].reshape(base.shape)


def load_metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def manual_merge_lora(
    base_model_path: Path,
    lora_model_path: Path,
    output_path: Path,
    *,
    default_scale: float,
) -> int:
    base_state = safetensors.torch.load_file(str(base_model_path))
    lora_state = safetensors.torch.load_file(str(lora_model_path))
    merged = {k: v.clone() for k, v in base_state.items()}
    merged_keys = 0

    for stem, group in sorted(extract_lora_groups(lora_state).items()):
        if group.a is None or group.b is None:
            continue
        base_key = resolve_base_key(merged, stem)
        if base_key is None:
            raise KeyError(f"Could not resolve base tensor for LoRA stem '{stem}'")
        scale = group.alpha / infer_rank(group.a) if group.alpha is not None else default_scale
        base_tensor = merged[base_key]
        delta = compute_delta(base_tensor, group.a.to(torch.float32), group.b.to(torch.float32))
        merged[base_key] = (base_tensor.to(torch.float32) + scale * delta).to(base_tensor.dtype)
        merged_keys += 1

    if merged_keys == 0:
        raise ValueError("No LoRA tensor pairs were found in the adapter checkpoint.")

    metadata = {
        **load_metadata(base_model_path),
        "merged_from_base": str(base_model_path),
        "merged_from_lora": str(lora_model_path),
        "merged_lora_tensors": str(merged_keys),
    }
    safetensors.torch.save_file(merged, str(output_path), metadata=metadata)
    return merged_keys


def main() -> None:
    args = parse_args()
    base_model_dir, base_model_weights_path = resolve_base_model_path(args.base_model_dir)
    lora_checkpoint_dir, adapter_model_path, adapter_config_path = resolve_lora_paths(args.lora_checkpoint_dir)

    output_model_path = args.output_dir / "model.safetensors"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if adapter_config_path is None:
        print(f"No adapter_config.json found next to {adapter_model_path}. Falling back to raw safetensors merge.")
        merged_keys = manual_merge_lora(
            base_model_weights_path,
            adapter_model_path,
            output_model_path,
            default_scale=args.default_lora_scale,
        )
        copy_tree_if_exists(base_model_dir / "assets", args.output_dir / "assets")
        copy_tree_if_exists(lora_checkpoint_dir / "assets", args.output_dir / "assets")
        print(f"Merged {merged_keys} tensors into {output_model_path}")
        return

    print(f"Loading base model from: {base_model_weights_path}")
    model_cfg = build_model_config(args)
    base_model = PI0Pytorch(model_cfg)
    attach_hf_config(base_model, model_cfg)

    safetensors.torch.load_model(base_model, base_model_weights_path)
    base_model.to(dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    print("Base model loaded successfully.")

    print(f"Loading LoRA adapter from: {lora_checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir, is_trainable=False)

    print("Merging LoRA weights into the base model...")
    model = model.merge_and_unload()
    print("Merge complete.")

    print(f"Saving merged model to: {args.output_dir}")
    model.config.save_pretrained(args.output_dir)
    safetensors.torch.save_model(model, output_model_path)

    copy_tree_if_exists(base_model_dir / "assets", args.output_dir / "assets")
    copy_tree_if_exists(lora_checkpoint_dir / "assets", args.output_dir / "assets")
    shutil.copy2(adapter_config_path, args.output_dir / "adapter_config.json")
    print(f"Merged model written to: {args.output_dir}")


if __name__ == "__main__":
    main()
