from __future__ import annotations

import dataclasses

from openpi.runtime import Pi05LiberoConfig


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    model: Pi05LiberoConfig
    policy_metadata: dict


def get_config(name: str) -> TrainConfig:
    if name != "pi05_libero":
        raise ValueError(f"Unsupported config: {name}")
    return TrainConfig(name="pi05_libero", model=Pi05LiberoConfig(), policy_metadata={"env": "libero"})
