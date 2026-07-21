from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: dict) -> dict: ...
