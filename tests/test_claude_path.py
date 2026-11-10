"""The Claude path end to end, with a fake client: no key, no network, no cost."""
import json
import sys
import types
from types import SimpleNamespace

import pytest

import bench.run as bench_run
from structreq.providers.claude import DEFAULT_MODEL, ClaudeProvider

FORBIDDEN = {"temperature", "top_p", "top_k"}


class FakeMessages:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=json.dumps(self.answer))],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=200))


class FakeClient:
    def __init__(self, answer):
        self.messages = FakeMessages(answer)


def check_request(kw):
    assert not FORBIDDEN & kw.keys(), "current models reject sampling parameters"
    assert kw["messages"][-1]["role"] == "user", "no assistant prefill"
    assert kw["output_config"]["format"]["type"] == "json_schema"


def test_provider_request_shape(valid):
    client = FakeClient(valid)
    p = ClaudeProvider(client=client)
    assert p.model == DEFAULT_MODEL == "claude-haiku-5-5"
    out = p.complete("system", [{"role": "user", "content": "VAT Q3"}])
    assert json.loads(out) == valid
    check_request(client.messages.calls[0])
    assert (p.input_tokens, p.output_tokens) == (1500, 200)


def test_refusal_stop_reason_is_a_refusal():
    client = FakeClient({})
    client.messages.create = lambda **kw: SimpleNamespace(stop_reason="refusal", content=[], usage=None)
    out = ClaudeProvider(client=client).complete("s", [{"role": "user", "content": "x"}])
    assert json.loads(out)["status"] == "refused"


def test_cost_estimate_is_shown_and_small():
    e = bench_run.estimate_claude_cost(bench_run.load_cases(), max_tokens=4000)
    assert e["calls"] == 60
    assert 0 < e["expected_usd"] < e["worst_usd"] < 1.0


def test_bench_command_adds_the_claude_line(tmp_path, monkeypatch, valid, capsys):
    """`./run.sh claude --yes` with a fake SDK: the Claude row lands in results.md and README."""
    fake = FakeClient(valid)
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=lambda: fake))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.delenv("STRUCTREQ_CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(bench_run, "ROOT", tmp_path)
    (tmp_path / "README.md").write_text("x\n<!-- results:start -->\n<!-- results:end -->\n")

    assert bench_run.main(["--provider", "claude", "--yes", "--limit", "5"]) == 0

    out = capsys.readouterr().out
    assert "Cost estimate for 5 cases on claude-haiku-5-5" in out
    assert len(fake.messages.calls) >= 5
    for kw in fake.messages.calls:
        check_request(kw)
    results = (tmp_path / "results.md").read_text()
    assert "## claude:claude-haiku-5-5" in results and "measured cost" in results
    assert "| claude: claude-haiku-5-5 |" in (tmp_path / "README.md").read_text()


def test_bench_command_stops_without_confirmation(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert bench_run.main(["--provider", "claude", "--limit", "2"]) == 1
    assert "nothing was sent" in capsys.readouterr().out
