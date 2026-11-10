"""The strict output contract.

The model must return exactly one JSON object. Pydantic validates it; anything
else (extra keys, a document code outside the nomenclature, personal data in a
text field, an inconsistent refusal) is rejected and may be repaired once.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import load_catalog
from .pii import find_pii

MAX_DOCUMENTS = 12


def _no_personal_data(value: Optional[str]) -> Optional[str]:
    if value and find_pii(value):
        raise ValueError("contains personal data (name, phone, e-mail, address or identifier); remove it")
    return value


class DocumentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    note: str = Field(min_length=1, max_length=200)

    @field_validator("code")
    @classmethod
    def code_in_nomenclature(cls, v: str) -> str:
        if v not in load_catalog().codes:
            raise ValueError(f"unknown document code {v!r}: pick a code from the list")
        return v

    @field_validator("note")
    @classmethod
    def note_without_pii(cls, v: str) -> str:
        return _no_personal_data(v)


class StructuredRequest(BaseModel):
    """What the model returns: either a request to send to the client, or a refusal."""
    model_config = ConfigDict(extra="forbid")

    status: Literal["request", "refused"]
    refusal_reason: Optional[Literal["off_topic", "too_vague"]] = None
    procedure: Optional[str] = None
    title: Optional[str] = Field(default=None, max_length=80)
    due_in_days: Optional[int] = Field(default=None, ge=1, le=365)
    documents: list[DocumentItem] = Field(default_factory=list, max_length=MAX_DOCUMENTS)

    @field_validator("procedure")
    @classmethod
    def procedure_known(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "other" and v not in load_catalog().procedures:
            raise ValueError(f"unknown procedure {v!r}: use one of the listed procedures or 'other'")
        return v

    @field_validator("title")
    @classmethod
    def title_without_pii(cls, v: Optional[str]) -> Optional[str]:
        return _no_personal_data(v)

    @model_validator(mode="after")
    def consistent(self) -> "StructuredRequest":
        if self.status == "refused":
            if self.refusal_reason is None:
                raise ValueError("a refusal needs refusal_reason 'off_topic' or 'too_vague'")
            if self.documents:
                raise ValueError("a refusal must not list documents")
        else:
            if self.refusal_reason is not None:
                raise ValueError("refusal_reason must be null when status is 'request'")
            if not self.title or len(self.title.strip()) < 3:
                raise ValueError("a request needs a short title")
            if not self.documents:
                raise ValueError("a request needs at least one document")
            codes = [d.code for d in self.documents]
            if len(codes) != len(set(codes)):
                raise ValueError("each document code may appear only once")
        return self


def json_schema() -> dict:
    """The same contract as a strict JSON Schema, with the nomenclature as an enum.

    Used by providers that support constrained decoding (Claude structured outputs).
    Pydantic still re-validates the answer: the schema is a help, not the guard.
    """
    cat = load_catalog()
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "refusal_reason", "procedure", "title", "due_in_days", "documents"],
        "properties": {
            "status": {"type": "string", "enum": ["request", "refused"]},
            "refusal_reason": {"anyOf": [{"type": "string", "enum": ["off_topic", "too_vague"]},
                                         {"type": "null"}]},
            "procedure": {"anyOf": [{"type": "string", "enum": sorted(cat.procedures) + ["other"]},
                                    {"type": "null"}]},
            "title": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "due_in_days": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "documents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "note"],
                    "properties": {
                        "code": {"type": "string", "enum": sorted(cat.codes)},
                        "note": {"type": "string"},
                    },
                },
            },
        },
    }
