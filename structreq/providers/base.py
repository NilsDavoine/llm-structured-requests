from __future__ import annotations

import hashlib
import json
from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict]) -> str:
        """Return the model's raw text answer. `messages` alternate user/assistant."""
        ...


def request_key(system: str, messages: list[dict]) -> str:
    """Stable key of one call, used to record and replay answers."""
    blob = json.dumps({"system": system, "messages": messages}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
