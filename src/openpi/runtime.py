from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import numpy as np
import safetensors.torch
import torch

from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
from openpi.shared import normalize as normalize_lib


@dataclasses.dataclass(frozen=True)
class Pi05LiberoConfig:
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 10
    max_token_len: int = 200
    pi05: bool = True
    discrete_state_input: bool = False
    pytorch_compile_mode: str | None = "max-autotune"


def checkpoint_asset_id() -> str:
    return "physical-intelligence/libero"


def resolve_checkpoint_dir(checkpoint_dir: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir)
    if path.exists():
        return path
    from openpi.shared import download

    return download.maybe_download(str(checkpoint_dir))


def load_norm_stats(checkpoint_dir: pathlib.Path) -> dict[str, normalize_lib.NormStats]:
    return normalize_lib.load(checkpoint_dir / "assets" / checkpoint_asset_id())


def normalize_inputs(data: dict[str, Any], norm_stats: dict[str, normalize_lib.NormStats]) -> dict[str, Any]:
    state_stats = norm_stats["state"]
    data["state"] = (data["state"] - state_stats.mean[..., : data["state"].shape[-1]]) / (
        state_stats.std[..., : data["state"].shape[-1]] + 1e-6
    )
    return data


def unnormalize_outputs(data: dict[str, Any], norm_stats: dict[str, normalize_lib.NormStats]) -> dict[str, Any]:
    action_stats = norm_stats["actions"]
    dim = action_stats.mean.shape[-1]
    data["actions"] = data["actions"] * (action_stats.std[..., :dim] + 1e-6) + action_stats.mean[..., :dim]
    return data


class LiberoPolicy:
    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        *,
        default_prompt: str | None = None,
        pytorch_device: str | None = None,
    ):
        self._checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
        self._default_prompt = default_prompt
        self._input_transform = LiberoInputs()
        self._output_transform = LiberoOutputs()
        self._tokenizer = PaligemmaTokenizer(Pi05LiberoConfig().max_token_len)
        self._norm_stats = load_norm_stats(self._checkpoint_dir)
        self._device = pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = PI0Pytorch(Pi05LiberoConfig())
        safetensors.torch.load_model(model, str(self._checkpoint_dir / "model.safetensors"))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        self._model = model.to(self._device).eval()
        self.metadata = {"checkpoint_dir": str(self._checkpoint_dir)}

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        if self._default_prompt is not None and "prompt" not in obs:
            obs = {**obs, "prompt": self._default_prompt}
        inputs = self._input_transform(dict(obs))
        inputs = normalize_inputs(inputs, self._norm_stats)
        tokens, token_mask = self._tokenizer.tokenize(str(inputs.pop("prompt")))
        inputs["tokenized_prompt"] = tokens
        inputs["tokenized_prompt_mask"] = token_mask
        inputs["state"] = np.pad(inputs["state"], (0, Pi05LiberoConfig().action_dim - inputs["state"].shape[-1]))

        tensor_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, dict):
                tensor_inputs[key] = {k: torch.from_numpy(np.asarray(v)).to(self._device)[None, ...] for k, v in value.items()}
            else:
                tensor_inputs[key] = torch.from_numpy(np.asarray(value)).to(self._device)[None, ...]
        observation = Observation.from_dict(tensor_inputs)

        sample_kwargs = {}
        if noise is not None:
            noise_tensor = torch.from_numpy(noise).to(self._device)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor[None, ...]
            sample_kwargs["noise"] = noise_tensor

        actions = self._model.sample_actions(self._device, observation, **sample_kwargs)
        outputs = {"state": tensor_inputs["state"], "actions": np.asarray(actions[0].detach().cpu())}
        outputs = unnormalize_outputs(outputs, self._norm_stats)
        outputs = self._output_transform(outputs)
        return outputs
