"""Claude through the official Anthropic SDK. Optional: only used when a key is provided."""
from __future__ import annotations

import os

from ..schema import json_schema

DEFAULT_MODEL = "claude-haiku-5-5"   # small and cheap; the original system used a Haiku model


class ClaudeProvider:
    def __init__(self, model: str | None = None, **_):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY to use the Claude provider")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install -r requirements-claude.txt") from e
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("STRUCTREQ_CLAUDE_MODEL", DEFAULT_MODEL)
        self.name = f"claude:{self.model}"

    def complete(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=messages,
            # Constrained decoding on the server; our Pydantic layer re-validates anyway.
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": json_schema()}},
        )
        if response.stop_reason == "refusal":
            return '{"status": "refused", "refusal_reason": "off_topic"}'
        return next((b.text for b in response.content if b.type == "text"), "")
