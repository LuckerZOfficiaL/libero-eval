from __future__ import annotations

from openpi.runtime import LiberoPolicy


def create_trained_policy(train_config, checkpoint_dir, *, default_prompt=None, pytorch_device=None, **_kwargs):
    del train_config
    return LiberoPolicy(checkpoint_dir, default_prompt=default_prompt, pytorch_device=pytorch_device)
