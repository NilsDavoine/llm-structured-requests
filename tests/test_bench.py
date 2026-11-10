"""The benchmark data itself, and the recorded answers of the local model."""
from pathlib import Path

import pytest

from bench.run import load_cases, run_case, score
from structreq.pii import redact
from structreq.providers.replay import ReplayProvider

REPLAYS = sorted((Path(__file__).resolve().parent.parent / "bench" / "replays").glob("*.jsonl"))


def test_cases_cover_the_brief():
    cases = load_cases()
    assert len(cases) >= 60
    assert {c["lang"] for c in cases} == {"fr", "en", "de"}
    assert sum(c["expect"]["status"] == "refused" for c in cases) >= 12
    assert sum(bool(c["pii"]) for c in cases) >= 10


def test_every_pii_string_of_the_bench_is_redacted():
    for c in load_cases():
        clean, _ = redact(c["text"])
        for s in c["pii"]:
            assert s not in clean, (c["id"], s)


@pytest.mark.parametrize("path", REPLAYS, ids=[p.stem for p in REPLAYS])
def test_recorded_local_answers_respect_the_guards(path):
    cases, provider = load_cases(), ReplayProvider(path)
    runs = [run_case(c, provider) for c in cases]
    s = score(cases, runs)
    assert s["pii_leaks"] == 0
    assert s["unknown_codes_in_output"] == 0
