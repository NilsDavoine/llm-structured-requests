"""sentence -> redact -> guard -> model -> parse -> validate -> (one repair) -> result.

Two guards are never optional:
  1. nothing containing personal data leaves the process (assert_no_pii on the payload);
  2. nothing outside the nomenclature, and no personal data, comes back out
     (StructuredRequest validation).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from . import prompt
from .catalog import load_catalog
from .pii import Finding, assert_no_pii, redact
from .providers.base import Provider
from .schema import StructuredRequest

MAX_REPAIRS = 1   # one new attempt at most, then a clean failure


class InvalidOutput(ValueError):
    pass


def parse_json(raw: str) -> dict:
    """Extract the single JSON object of the answer. Tolerates a code fence, nothing more."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise InvalidOutput("no JSON object found")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise InvalidOutput(f"invalid JSON ({e.msg} at character {e.pos})") from None
    if not isinstance(obj, dict):
        raise InvalidOutput("the answer must be a JSON object")
    return obj


@dataclass
class Outcome:
    status: str                       # "request" | "refused" | "failed"
    result: Optional[StructuredRequest]
    attempts: int
    first_try_valid: bool
    errors: list[str]
    latency_s: float
    redactions: list[Finding]
    sent: list[dict] = field(default_factory=list)   # every payload actually sent (for audits)

    def client_view(self, lang: str, today: Optional[dt.date] = None) -> dict:
        """What the client would see: labels from the catalog, a real due date."""
        if self.status != "request" or self.result is None:
            return {"status": self.status, "reason": self.result.refusal_reason if self.result else None,
                    "errors": self.errors}
        cat, r = load_catalog(), self.result
        today = today or dt.date.today()
        return {
            "status": "request",
            "title": r.title,
            "procedure": r.procedure,
            "due_date": (today + dt.timedelta(days=r.due_in_days)).isoformat() if r.due_in_days else None,
            "documents": [{"code": d.code, "label": cat.label(d.code, lang), "note": d.note}
                          for d in r.documents],
        }


def _send(provider: Provider, system: str, messages: list[dict], sent: list[dict]) -> str:
    assert_no_pii(*(m["content"] for m in messages))     # guard 1: the payload is checked, not trusted
    sent.append({"system": system, "messages": [dict(m) for m in messages]})
    return provider.complete(system, messages)


def _validate(raw: str) -> StructuredRequest:
    return StructuredRequest.model_validate(parse_json(raw))   # guard 2


def run(sentence: str, provider: Provider, max_repairs: int = MAX_REPAIRS) -> Outcome:
    t0 = time.perf_counter()
    clean, findings = redact(sentence)
    system = prompt.system_prompt()
    messages = [{"role": "user", "content": prompt.user_message(clean)}]
    sent: list[dict] = []
    errors: list[str] = []

    for attempt in range(1 + max_repairs):
        raw = _send(provider, system, messages, sent)
        try:
            result = _validate(raw)
        except (InvalidOutput, ValidationError) as e:
            errors.append(prompt.describe_error(e))
            # The previous answer goes back to the model redacted: if it invented a
            # name or a number, that must not be sent either.
            messages = messages + [
                {"role": "assistant", "content": redact(raw)[0]},
                {"role": "user", "content": prompt.repair_message(errors[-1])},
            ]
            continue
        return Outcome(result.status, result, attempt + 1, attempt == 0, errors,
                       time.perf_counter() - t0, findings, sent)

    return Outcome("failed", None, 1 + max_repairs, False, errors,
                   time.perf_counter() - t0, findings, sent)


_LANG_HINTS = {
    "fr": {"le", "la", "les", "des", "du", "de", "pour", "il", "me", "faut", "et", "un", "une", "dans", "sous", "pièces"},
    "de": {"der", "die", "das", "und", "für", "ich", "brauche", "bitte", "mit", "den", "eine", "einer", "innerhalb", "von"},
    "en": {"the", "and", "for", "we", "need", "please", "with", "of", "a", "within", "to", "from"},
}


def guess_lang(sentence: str) -> str:
    words = re.findall(r"[a-zà-öø-ÿß]+", sentence.lower())
    scores = {lang: sum(w in hints for w in words) for lang, hints in _LANG_HINTS.items()}
    return max(scores, key=scores.get) if any(scores.values()) else "en"
