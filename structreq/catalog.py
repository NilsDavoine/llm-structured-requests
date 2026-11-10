"""The closed nomenclature: the only document codes a request may contain."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).parent / "data" / "catalog.yml"
LANGS = ("en", "fr", "de")


@dataclass(frozen=True)
class Catalog:
    documents: dict[str, dict[str, str]]   # code -> {en, fr, de}
    procedures: dict[str, dict]            # name -> {label, usual: [codes]}

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(self.documents)

    def label(self, code: str, lang: str = "en") -> str:
        return self.documents[code].get(lang) or self.documents[code]["en"]


@lru_cache(maxsize=1)
def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cat = Catalog(documents=data["documents"], procedures=data["procedures"])
    for name, proc in cat.procedures.items():
        unknown = set(proc["usual"]) - cat.codes
        if unknown:
            raise ValueError(f"procedure {name} uses unknown codes {sorted(unknown)}")
    return cat
