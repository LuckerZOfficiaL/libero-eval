from __future__ import annotations

import dataclasses
from typing import Generic, TypeVar

import numpy as np
import torch


ArrayT = TypeVar("ArrayT", bound=torch.Tensor | np.ndarray)


@dataclasses.dataclass
class Observation(Generic[ArrayT]):
    images: dict[str, ArrayT]
    image_masks: dict[str, ArrayT]
    state: ArrayT
    tokenized_prompt: ArrayT | None = None
    tokenized_prompt_mask: ArrayT | None = None
    token_ar_mask: ArrayT | None = None
    token_loss_mask: ArrayT | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Observation":
        images = {}
        for key, value in data["image"].items():
            if hasattr(value, "dtype") and value.dtype == torch.uint8:
                images[key] = value.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                images[key] = value.astype(np.float32) / 255.0 * 2.0 - 1.0
            else:
                images[key] = value
        return cls(
            images=images,
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )
