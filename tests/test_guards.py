"""The two guards that must never let anything through. Checked by mutation (scripts/mutation_check.py)."""
import copy

import pytest
from pydantic import ValidationError

from structreq import pipeline
from structreq.pii import PIILeak
from structreq.schema import StructuredRequest

SENTENCE_WITH_PII = ("VAT return Q3 for Mr. David Brown, usual documents, call him on "
                     "+352 621 123 456 or d.brown@example.com")
SECRETS = ["David Brown", "Brown", "621 123 456", "d.brown@example.com"]


# ---- guard 1: no document outside the nomenclature ever passes ----

def test_unknown_code_is_rejected_by_the_schema(valid):
    bad = copy.deepcopy(valid)
    bad["documents"][0]["code"] = "TAX_RETURN_OF_THE_DOG"
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(bad)


def test_unknown_code_never_reaches_the_result(scripted, valid):
    bad = copy.deepcopy(valid)
    bad["documents"].append({"code": "INVENTED_DOCUMENT", "note": "Something."})
    p = scripted(bad, bad)
    out = pipeline.run("VAT return Q3, usual documents", p)
    assert out.status == "failed" and out.result is None
    assert len(p.calls) == 2                       # one repair, no more


def test_unknown_code_is_repaired_once(scripted, valid):
    bad = copy.deepcopy(valid)
    bad["documents"][0]["code"] = "SALES_INVOICE"   # near miss
    p = scripted(bad, valid)
    out = pipeline.run("VAT return Q3, usual documents", p)
    assert out.status == "request" and out.attempts == 2 and not out.first_try_valid
    assert "unknown document code" in p.calls[1][-1]["content"]


# ---- guard 2: no personal data ever leaves, and none comes back ----

def test_personal_data_is_never_sent(scripted, valid):
    p = scripted(valid)
    pipeline.run(SENTENCE_WITH_PII, p)
    sent = str(p.calls)
    for s in SECRETS:
        assert s not in sent


def test_send_guard_blocks_even_if_redaction_is_bypassed(scripted, valid, monkeypatch):
    monkeypatch.setattr(pipeline, "redact", lambda text: (text, []))
    p = scripted(valid)
    with pytest.raises(PIILeak) as e:
        pipeline.run(SENTENCE_WITH_PII, p)
    assert p.calls == []                           # the provider was never called
    for s in SECRETS:
        assert s not in str(e.value)               # the error does not carry the data


def test_personal_data_in_the_answer_is_rejected(scripted, valid):
    leaky = copy.deepcopy(valid)
    leaky["documents"][0]["note"] = "Ask Sophie Lambert, phone 691 234 567."
    p = scripted(leaky, leaky)
    out = pipeline.run("VAT return Q3, usual documents", p)
    assert out.status == "failed" and out.result is None
    # the model's own invented data is redacted before being sent back for repair
    assert "691 234 567" not in str(p.calls[1]) and "Sophie Lambert" not in str(p.calls[1])


def test_personal_data_in_title_is_rejected(valid):
    leaky = copy.deepcopy(valid)
    leaky["title"] = "VAT for jean.martin@example.lu"
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(leaky)


# ---- repair is bounded ----

def test_repair_is_bounded_to_one_attempt(scripted):
    p = scripted("not json", "still not json", "never asked")
    out = pipeline.run("VAT return Q3", p)
    assert out.status == "failed" and out.attempts == 2 and len(p.calls) == 2
