import copy

import pytest
from pydantic import ValidationError

from structreq.catalog import load_catalog
from structreq.schema import StructuredRequest, json_schema


def test_valid_request(valid):
    r = StructuredRequest.model_validate(valid)
    assert [d.code for d in r.documents] == ["SALES_INVOICES", "PURCHASE_INVOICES"]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra_field=1),
    lambda d: d["documents"][0].update(extra=1),
    lambda d: d.update(status="maybe"),
    lambda d: d.update(procedure="tax_evasion"),
    lambda d: d.update(due_in_days=0),
    lambda d: d.update(title="x" * 81),
    lambda d: d.update(documents=[]),
    lambda d: d["documents"].append(dict(d["documents"][0])),          # duplicate code
    lambda d: d.update(refusal_reason="off_topic"),                     # request + reason
    lambda d: d["documents"][0].update(note=""),
])
def test_invalid_outputs_are_rejected(valid, mutate):
    d = copy.deepcopy(valid)
    mutate(d)
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(d)


def test_refusal_shapes():
    StructuredRequest.model_validate({"status": "refused", "refusal_reason": "too_vague"})
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate({"status": "refused"})
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate({"status": "refused", "refusal_reason": "off_topic",
                                          "documents": [{"code": "CASH_BOOK", "note": "x"}]})


def test_json_schema_enum_is_the_catalog():
    enum = json_schema()["properties"]["documents"]["items"]["properties"]["code"]["enum"]
    assert set(enum) == load_catalog().codes
