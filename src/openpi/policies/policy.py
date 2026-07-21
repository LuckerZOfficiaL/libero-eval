from __future__ import annotations

from openpi.runtime import LiberoPolicy as Policy


class PolicyRecorder:
    def __init__(self, policy, _record_dir: str):
        self._policy = policy

    def infer(self, obs: dict):
        return self._policy.infer(obs)

    @property
    def metadata(self):
        return self._policy.metadata


__all__ = ["Policy", "PolicyRecorder"]
