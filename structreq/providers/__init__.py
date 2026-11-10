"""Interchangeable model providers. All of them take (system, messages) and return raw text."""
from __future__ import annotations

from .base import Provider


def make_provider(name: str, **kwargs) -> Provider:
    if name == "local":
        from .local_hf import LocalHFProvider
        return LocalHFProvider(**kwargs)
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(**kwargs)
    if name == "replay":
        from .replay import ReplayProvider
        return ReplayProvider(**kwargs)
    raise ValueError(f"unknown provider {name!r} (local, claude, replay)")
