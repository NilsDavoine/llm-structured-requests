"""Claude through the official Anthropic SDK. Optional: only used when a key is provided."""
from __future__ import annotations

import os

from ..schema import json_schema

DEFAULT_MODEL = "claude-haiku-5-5"   # small and cheap; the original system used a Haiku model
MAX_TOKENS = 4000


class ClaudeProvider:
    """No temperature/top_p/top_k and no assistant prefill: current models reject both
    with a 400. Structured outputs (constrained decoding) replace the prefill trick."""

    def __init__(self, model: str | None = None, client=None, **_):
        if client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("set ANTHROPIC_API_KEY to use the Claude provider")
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError("pip install -r requirements-claude.txt") from e
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("STRUCTREQ_CLAUDE_MODEL", DEFAULT_MODEL)
        self.name = f"claude:{self.model}"
        self.input_tokens = self.output_tokens = 0   # measured, for the real cost

    def complete(self, system: str, messages: list[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            # Constrained decoding on the server; our Pydantic layer re-validates anyway.
            output_config={"effort": "low",
                           "format": {"type": "json_schema", "schema": json_schema()}},
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += usage.input_tokens
            self.output_tokens += usage.output_tokens
        if response.stop_reason == "refusal":
            return '{"status": "refused", "refusal_reason": "off_topic"}'
        return next((b.text for b in response.content if b.type == "text"), "")
