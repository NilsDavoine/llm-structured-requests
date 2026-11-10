import json

import pytest

from structreq import pipeline
from structreq.pipeline import InvalidOutput, parse_json
from structreq.providers.replay import RecordingProvider, ReplayMiss, ReplayProvider


def test_valid_first_try(scripted, valid):
    out = pipeline.run("VAT return Q3, the usual documents, within two weeks", scripted(valid))
    assert out.status == "request" and out.first_try_valid and out.attempts == 1
    view = out.client_view("fr")
    assert view["documents"][0]["label"] == "Factures de vente" and view["due_date"]


def test_code_fence_is_tolerated(scripted, valid):
    out = pipeline.run("VAT Q3", scripted("```json\n" + json.dumps(valid) + "\n```"))
    assert out.status == "request"


@pytest.mark.parametrize("raw", ["", "Sure! Here it is.", "[1, 2]", "{'status': 'request'}", '{"status": '])
def test_parse_json_rejects_non_objects(raw):
    with pytest.raises(InvalidOutput):
        parse_json(raw)


def test_refusal_passes_through(scripted):
    out = pipeline.run("Write me a poem", scripted({"status": "refused", "refusal_reason": "off_topic"}))
    assert out.status == "refused" and out.result.refusal_reason == "off_topic"


def test_record_then_replay(tmp_path, scripted, valid):
    path = tmp_path / "rec.jsonl"
    live = RecordingProvider(scripted(valid), path)
    a = pipeline.run("VAT return Q3", live)
    b = pipeline.run("VAT return Q3", ReplayProvider(path))
    assert a.result == b.result
    with pytest.raises(ReplayMiss):
        pipeline.run("Something never recorded", ReplayProvider(path))


@pytest.mark.parametrize("text,lang", [("Il me faut les pièces habituelles pour la TVA", "fr"),
                                       ("Bitte die üblichen Unterlagen für die Gründung", "de"),
                                       ("We need the usual documents for the VAT return", "en")])
def test_guess_lang(text, lang):
    assert pipeline.guess_lang(text) == lang
