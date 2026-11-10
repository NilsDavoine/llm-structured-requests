"""Replays answers recorded from a real model, so tests and the benchmark run offline."""
from __future__ import annotations

import json
from pathlib import Path

from .base import request_key


class ReplayMiss(KeyError):
    pass


class ReplayProvider:
    def __init__(self, path: str | Path, **_):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                self.records[r["key"]] = r
        models = {r.get("model") for r in self.records.values()}
        self.name = "replay:" + ",".join(sorted(m for m in models if m))

    def complete(self, system: str, messages: list[dict]) -> str:
        key = request_key(system, messages)
        if key not in self.records:
            raise ReplayMiss(f"no recorded answer for this request ({key[:12]})")
        return self.records[key]["output"]


class RecordingProvider:
    """Wraps a live provider and appends every exchange to a JSONL file."""

    def __init__(self, inner, path: str | Path):
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def complete(self, system: str, messages: list[dict]) -> str:
        out = self.inner.complete(system, messages)
        rec = {"key": request_key(system, messages), "model": self.inner.name, "output": out}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out
