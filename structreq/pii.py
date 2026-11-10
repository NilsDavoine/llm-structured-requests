"""Personal-data filter applied BEFORE anything is sent to a model.

Deterministic and explainable on purpose: regular expressions for contact
details and identifiers, plus two rules for people's names (a courtesy title
followed by a capitalised word, or a known first name). It is a filter for
short business sentences, not a general-purpose anonymiser: see README limits.

`redact()` replaces every finding with a typed placeholder such as [PHONE].
`find_pii()` is reused by two guards: one on the outgoing payload, one on the
model's answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FIRST_NAMES_PATH = Path(__file__).parent / "data" / "first_names.txt"


def _load_first_names() -> frozenset[str]:
    names = set()
    for line in _FIRST_NAMES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.update(line.split())
    return frozenset(names)


FIRST_NAMES = _load_first_names()

_CAP = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿß]+(?:-[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿß]+)?"
_PARTICLE = r"(?:(?:de|du|da|dos|van|von|der|le|la)\s+)?"
_TITLES = (r"M\.|Mme|Mlle|Monsieur|Madame|Mademoiselle|Me|Maître|Mr\.?|Mrs\.?|Ms\.?|Miss|"
           r"Dr\.?|Herr|Herrn|Frau|Sr\.?|Sra\.?|Senhor|Senhora|Dona")

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b")),
    # +352 621 123 456, 00352 26 12 34 56, 621-123-456, 691123456, 1985041212345 (13-digit ID)
    ("PHONE_OR_ID", re.compile(r"(?<![\w.,])(?:\+|00)?\d(?:[ .\-/]?\(?\d\)?){7,15}(?![\w])")),
    ("ADDRESS", re.compile(
        r"\b\d{1,4}(?:bis|ter|[a-z])?,?\s+(?i:rue|avenue|av\.|boulevard|bd|route|place|chemin|allée|"
        r"impasse|street|st\.|road|lane|Straße|Strasse|Str\.|Weg|Gasse|Platz|Allee)"
        r"(?:\s+(?:de\s+la|de|du|des|d'|der|the))?\s*[A-ZÀ-Þa-zà-ÿß][a-zà-ÿß']+(?:-[A-ZÀ-Þa-zà-ÿß][a-zà-ÿß']+)*"
        r"(?:\s+[A-ZÀ-Þ][a-zà-ÿß']+(?:-[A-ZÀ-Þ][a-zà-ÿß']+)*)*")),
    ("ADDRESS", re.compile(
        r"\b\d{1,4}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St\.|Road|Rd\.|Avenue|Lane|Drive|Way|"
        r"Straße|Strasse|Weg|Gasse|Allee)\b")),
    ("ADDRESS", re.compile(
        r"\b[A-ZÄÖÜ][a-zäöüß]+(?:straße|strasse|weg|gasse|platz|allee)\s+\d{1,4}[a-z]?\b")),
    ("POSTCODE", re.compile(r"\b(?:L|F|D|B)-\d{4,5}\b")),
    ("PERSON", re.compile(rf"(?<!\w)(?:{_TITLES})\s+(?:(?:{_TITLES})\s+)?{_PARTICLE}{_CAP}(?:\s+{_PARTICLE}{_CAP})?")),
]

_YEARISH = re.compile(r"^(?:19|20)\d\d$")
_DATE = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")


@dataclass(frozen=True)
class Finding:
    kind: str
    text: str
    start: int
    end: int


def _is_false_phone(s: str) -> bool:
    """Dates (31.12.2025) and runs of years (2024 2025) are not phone numbers."""
    s = s.strip()
    if _DATE.match(s):
        return True
    groups = re.split(r"[ .\-/]+", s)
    return all(_YEARISH.match(g) for g in groups)


def _name_findings(text: str) -> list[Finding]:
    """A known first name, plus the capitalised surname that follows it, if any."""
    out = []
    for m in re.finditer(rf"(?<!\w){_CAP}", text):
        if m.group(0) in FIRST_NAMES:
            tail = re.match(rf"(?:\s+{_PARTICLE}{_CAP})?", text[m.end():])
            end = m.end() + (tail.end() if tail else 0)
            out.append(Finding("PERSON", text[m.start():end], m.start(), end))
    return out


def find_pii(text: str) -> list[Finding]:
    found: list[Finding] = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            if kind == "PHONE_OR_ID" and _is_false_phone(m.group(0)):
                continue
            found.append(Finding(kind, m.group(0), m.start(), m.end()))
    found.extend(_name_findings(text))
    # keep the longest non-overlapping spans
    found.sort(key=lambda f: (f.start, -(f.end - f.start)))
    kept: list[Finding] = []
    for f in found:
        if kept and f.start < kept[-1].end:
            continue
        kept.append(f)
    return kept


_PLACEHOLDER = {"EMAIL": "[EMAIL]", "IBAN": "[IBAN]", "PHONE_OR_ID": "[NUMBER]",
                "ADDRESS": "[ADDRESS]", "POSTCODE": "[POSTCODE]", "PERSON": "[NAME]"}


def redact(text: str) -> tuple[str, list[Finding]]:
    findings = find_pii(text)
    out, last = [], 0
    for f in findings:
        out.append(text[last:f.start])
        out.append(_PLACEHOLDER[f.kind])
        last = f.end
    out.append(text[last:])
    return "".join(out), findings


class PIILeak(RuntimeError):
    """Raised when personal data is about to leave the process. Never carries the data itself."""


def assert_no_pii(*texts: str) -> None:
    for t in texts:
        found = find_pii(t)
        if found:
            kinds = sorted({f.kind for f in found})
            raise PIILeak(f"outgoing payload still contains personal data ({', '.join(kinds)})")
