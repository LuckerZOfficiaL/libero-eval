#!/usr/bin/env python
"""Merge MULTIPLE task-specific LoRA adapters into one multi-task applied checkpoint.

Distinct from scripts/apply_lora.py, which *applies* a SINGLE adapter. Here we *merge*: combine the
per-module deltas of several adapters via a merge method, bake the combined delta into the base, and
write one applied checkpoint the existing serve/eval pipeline can serve directly.

All LIBERO adapters here are homogeneous (r=16, alpha=16, scale=alpha/r=1.0, same target modules), so
a merge is a per-module combine over a stack of compatible deltas dW_i = B_i @ A_i.

Processed **module-by-module** (base weights are updated in place; only the current module's deltas
live in float32 at once) to keep memory bounded. Reuses the verified delta math from apply_lora.py.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import safetensors.torch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apply_lora as al  # noqa: E402  (reuse extract_lora_groups / resolve_base_key / compute_delta / infer_rank)


def adapter_scale(adapter_file: Path) -> float:
    """LoRA scale (alpha / r, or alpha / sqrt(r) for rsLoRA) read from the adapter's config."""
    cfg = adapter_file.parent / "adapter_config.json"
    if not cfg.exists():
        return 1.0
    c = json.loads(cfg.read_text())
    r, alpha = c.get("r"), c.get("lora_alpha")
    if not r or alpha is None:
        return 1.0
    return alpha / math.sqrt(r) if c.get("use_rslora") else alpha / r


def combine(deltas: list[torch.Tensor], method: str, params: dict) -> torch.Tensor:
    """Combine per-module deltas (list of same-shape float32 tensors) into one delta.

    Processed WITHOUT stacking (a (k, n) stack + its temporaries blow up memory for the huge modules
    like lm_head/embeddings); we accumulate incrementally instead.
    """
    if method == "average":
        out = deltas[0].clone()
        for d in deltas[1:]:
            out += d
        return out / len(deltas)
    if method == "task_arithmetic":
        lam = float(params.get("lambda", 1.0))
        out = deltas[0].clone()
        for d in deltas[1:]:
            out += d
        return out.mul_(lam)
    if method == "ties":
        # trim each task delta to its top-`density` entries by |magnitude| (fast kthvalue threshold),
        # elect a per-param sign from the summed trimmed deltas, average the agreeing entries, x lambda.
        density = float(params.get("density", 0.2))
        lam = float(params.get("lambda", 1.0))
        trimmed = []
        for d in deltas:
            f = d.reshape(-1)
            n = f.numel()
            keep = max(1, int(density * n))
            if keep < n:
                thr = f.abs().kthvalue(n - keep).values
                trimmed.append(torch.where(d.abs() > thr, d, torch.zeros((), dtype=d.dtype)))
            else:
                trimmed.append(d)
        s = trimmed[0].clone()
        for t in trimmed[1:]:
            s += t
        elected = torch.sign(s)
        del s
        num = torch.zeros_like(trimmed[0])
        den = torch.zeros_like(trimmed[0])
        for t in trimmed:
            ag = (torch.sign(t) == elected) & (t != 0)
            num += torch.where(ag, t, torch.zeros((), dtype=t.dtype))
            den += ag.to(t.dtype)
            del ag
        return num.div_(den.clamp_(min=1)).mul_(lam)
    raise ValueError(f"unknown merge method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multiple LoRA adapters into one multi-task applied checkpoint.")
    ap.add_argument("--base-model-dir", required=True, type=Path, help="dir with model.safetensors, or the file itself")
    ap.add_argument("--adapters", required=True, nargs="+", type=Path, help="adapter .safetensors files to merge")
    ap.add_argument("--method", default="task_arithmetic", choices=["average", "task_arithmetic", "ties"])
    ap.add_argument("--params", default="{}", help='JSON method params, e.g. \'{"lambda":1.0}\'')
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--norm-stats", type=Path, default=None, help="norm_stats.json to embed under assets/physical-intelligence/libero/")
    args = ap.parse_args()
    params = json.loads(args.params)

    base_dir, base_weights = al.resolve_base_model_path(args.base_model_dir)
    merged = safetensors.torch.load_file(str(base_weights))     # updated in place -> the output

    # load every adapter's LoRA groups once (small), and map each targeted base weight -> contributors
    adapters = [(al.extract_lora_groups(safetensors.torch.load_file(str(f))), adapter_scale(f)) for f in args.adapters]
    contributors: dict[str, list[tuple[int, str]]] = {}
    for ai, (groups, _scale) in enumerate(adapters):
        for stem, g in groups.items():
            if g.a is None or g.b is None:
                continue
            base_key = al.resolve_base_key(merged, stem)
            if base_key is None:
                raise KeyError(f"could not resolve base tensor for LoRA stem '{stem}' ({args.adapters[ai]})")
            contributors.setdefault(base_key, []).append((ai, stem))

    if not contributors:
        raise ValueError("no LoRA deltas found across the given adapters")

    # one module at a time: gather its deltas across adapters, combine, add to base, free
    for base_key, lst in contributors.items():
        deltas = []
        for ai, stem in lst:
            groups, scale = adapters[ai]
            g = groups[stem]
            s = g.alpha / al.infer_rank(g.a) if g.alpha is not None else scale
            deltas.append(s * al.compute_delta(merged[base_key], g.a.to(torch.float32), g.b.to(torch.float32)))
        combined = combine(deltas, args.method, params)
        merged[base_key] = (merged[base_key].to(torch.float32) + combined).to(merged[base_key].dtype)
        del deltas, combined

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"merge_method": args.method, "merge_params": json.dumps(params),
            "merged_adapters": ";".join(str(a) for a in args.adapters), "merged_modules": str(len(contributors))}
    safetensors.torch.save_file(merged, str(args.output_dir / "model.safetensors"), metadata=meta)
    if (base_dir / "config.json").exists():
        shutil.copy2(base_dir / "config.json", args.output_dir / "config.json")
    if args.norm_stats:
        ns = args.output_dir / "assets" / "physical-intelligence" / "libero"
        ns.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.norm_stats, ns / "norm_stats.json")
    print(f"merged {len(args.adapters)} adapters via {args.method} {params} across {len(contributors)} modules -> {args.output_dir}")


if __name__ == "__main__":
    main()
