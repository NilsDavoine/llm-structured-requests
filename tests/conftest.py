import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ScriptedProvider:
    """Test double: returns the given answers in order and records what it was sent."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append([dict(m) for m in messages])
        a = self.answers.pop(0)
        return a if isinstance(a, str) else json.dumps(a)


@pytest.fixture
def scripted():
    return ScriptedProvider


VALID = {"status": "request", "refusal_reason": None, "procedure": "vat_return",
         "title": "VAT return Q3", "due_in_days": 14,
         "documents": [{"code": "SALES_INVOICES", "note": "All sales invoices of July to September."},
                       {"code": "PURCHASE_INVOICES", "note": "All purchase invoices of July to September."}]}


@pytest.fixture
def valid():
    return json.loads(json.dumps(VALID))
