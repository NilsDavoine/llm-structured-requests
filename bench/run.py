"""Run the benchmark and write results.md (generated, never edited by hand).

    python -m bench.run --provider local                       # measure the local model
    python -m bench.run --provider local --record              # ... and record its answers
    python -m bench.run --provider replay --replay-file F      # re-score recorded answers offline
    python -m bench.run --provider claude                      # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import statistics
import time
from pathlib import Path

import yaml

from structreq import pipeline
from structreq.catalog import load_catalog
from structreq.pii import PIILeak
from structreq.providers import make_provider
from structreq.providers.replay import RecordingProvider

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "bench" / "cases.yml"


def load_cases(path: Path = CASES) -> list[dict]:
    cat = load_catalog()
    cases = yaml.safe_load(path.read_text(encoding="utf-8"))
    for c in cases:
        exp = c["expect"]
        docs = exp.get("documents", [])
        if isinstance(docs, str):
            assert docs.startswith("usual:"), docs
            docs = list(cat.procedures[docs.split(":", 1)[1]]["usual"])
        exp["documents"] = sorted(set(docs) | set(exp.get("plus", [])))
        unknown = set(exp["documents"]) - cat.codes
        assert not unknown, (c["id"], unknown)
        c.setdefault("pii", [])
    return cases


def score(cases: list[dict], runs: list[dict]) -> dict:
    n = len(cases)
    first = sum(r["first_try_valid"] for r in runs)
    valid = sum(r["status"] != "failed" for r in runs)
    exp_ref = [r for c, r in zip(cases, runs) if c["expect"]["status"] == "refused"]
    exp_req = [(c, r) for c, r in zip(cases, runs) if c["expect"]["status"] == "request"]
    tp = fp = fn = exact = 0
    for c, r in exp_req:
        want, got = set(c["expect"]["documents"]), set(r["documents"])
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)
        exact += want == got
    due_cases = [(c, r) for c, r in exp_req if "due_in_days" in c["expect"]]
    lat = [r["latency_s"] for r in runs]
    return {
        "cases": n,
        "valid_first_try": first,
        "valid_after_repair": valid,
        "failed": n - valid,
        "correct_refusals": sum(r["status"] == "refused" for r in exp_ref),
        "expected_refusals": len(exp_ref),
        "refusal_reason_ok": sum(r["status"] == "refused" and r["refusal_reason"] == c["expect"]["reason"]
                                 for c, r in zip(cases, runs) if c["expect"]["status"] == "refused"),
        "false_refusals": sum(r["status"] == "refused" for _, r in exp_req),
        "request_cases": len(exp_req),
        "doc_precision": tp / (tp + fp) if tp + fp else 0.0,
        "doc_recall": tp / (tp + fn) if tp + fn else 0.0,
        "exact_document_sets": exact,
        "due_ok": sum(r["due_in_days"] == c["expect"]["due_in_days"] for c, r in due_cases),
        "due_cases": len(due_cases),
        "pii_cases": sum(bool(c["pii"]) for c in cases),
        "pii_leaks": sum(r["pii_leaks"] for r in runs),
        "unknown_codes_in_output": sum(r["unknown_codes"] for r in runs),
        "latency_p50": statistics.median(lat),
        "latency_p95": sorted(lat)[max(0, int(round(0.95 * len(lat))) - 1)],
        "latency_mean": statistics.fmean(lat),
    }


def run_case(case: dict, provider) -> dict:
    try:
        out = pipeline.run(case["text"], provider)
    except PIILeak as e:      # the guard stopped the call: counted, never silent
        return {"id": case["id"], "status": "failed", "first_try_valid": False, "attempts": 0,
                "documents": [], "refusal_reason": None, "due_in_days": None, "latency_s": 0.0,
                "errors": [str(e)], "pii_leaks": 0, "unknown_codes": 0, "blocked_by_send_guard": True}
    sent_blob = json.dumps([s["messages"] for s in out.sent], ensure_ascii=False)
    leaks = sum(s in sent_blob for s in case["pii"])
    r = out.result
    docs = [d.code for d in r.documents] if r and r.status == "request" else []
    codes = load_catalog().codes
    return {
        "id": case["id"], "status": out.status, "first_try_valid": out.first_try_valid,
        "attempts": out.attempts, "documents": docs,
        "refusal_reason": r.refusal_reason if r else None,
        "due_in_days": r.due_in_days if r else None,
        "latency_s": round(out.latency_s, 3), "errors": out.errors,
        "pii_leaks": leaks, "unknown_codes": sum(d not in codes for d in docs),
        "redacted": sorted({f.kind for f in out.redactions}),
    }


def pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"


def render(provider_name: str, s: dict, cases: list[dict], runs: list[dict], live: bool) -> str:
    lat_note = ("" if live else
                " — replayed answers: latency here is only the pipeline overhead, not the model")
    lines = [
        f"## {provider_name}",
        "",
        f"Generated by `python -m bench.run` on {dt.date.today().isoformat()}, "
        f"{platform.python_version()}, {len(cases)} cases.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Valid on first try | {pct(s['valid_first_try'], s['cases'])} |",
        f"| Valid after at most one repair | {pct(s['valid_after_repair'], s['cases'])} |",
        f"| Clean failures (still invalid after repair) | {s['failed']} |",
        f"| Correct refusals (off-topic or too vague) | {pct(s['correct_refusals'], s['expected_refusals'])} |",
        f"| ... with the right reason | {pct(s['refusal_reason_ok'], s['expected_refusals'])} |",
        f"| False refusals of real requests | {pct(s['false_refusals'], s['request_cases'])} |",
        f"| Document precision (micro) | {s['doc_precision']:.2f} |",
        f"| Document recall (micro) | {s['doc_recall']:.2f} |",
        f"| Exact document set | {pct(s['exact_document_sets'], s['request_cases'])} |",
        f"| Delay in days correct | {pct(s['due_ok'], s['due_cases'])} |",
        f"| Personal data sent to the model | {s['pii_leaks']} leaks over {s['pii_cases']} cases with personal data |",
        f"| Document codes outside the nomenclature in validated output | {s['unknown_codes_in_output']} |",
        f"| Latency p50 / p95 / mean (s){lat_note} | {s['latency_p50']:.2f} / {s['latency_p95']:.2f} / {s['latency_mean']:.2f} |",
        "",
        "<details><summary>Per-case results</summary>",
        "",
        "| Case | Expected | Got | Attempts | s |",
        "|---|---|---|---|---|",
    ]
    for c, r in zip(cases, runs):
        e = c["expect"]
        exp = (f"refused ({e['reason']})" if e["status"] == "refused"
               else ", ".join(e["documents"]) + (f"; {e['due_in_days']} d" if "due_in_days" in e else ""))
        got = (f"refused ({r['refusal_reason']})" if r["status"] == "refused"
               else "FAILED" if r["status"] == "failed"
               else ", ".join(sorted(r["documents"])) + (f"; {r['due_in_days']} d" if r["due_in_days"] else ""))
        mark = "" if (e["status"] == r["status"] and
                      (e["status"] == "refused" or set(e["documents"]) == set(r["documents"]))) else " ✗"
        lines.append(f"| {c['id']}{mark} | {exp} | {got} | {r['attempts']} | {r['latency_s']:.1f} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="local", choices=["local", "claude", "replay"])
    ap.add_argument("--model")
    ap.add_argument("--replay-file")
    ap.add_argument("--record", action="store_true", help="record live answers to bench/replays/")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    cases = load_cases()[: args.limit] if args.limit else load_cases()
    kwargs = {"model": args.model} if args.model else {}
    if args.provider == "replay":
        kwargs["path"] = args.replay_file
    t0 = time.perf_counter()
    provider = make_provider(args.provider, **kwargs)
    load_s = time.perf_counter() - t0
    slug = provider.name.replace("/", "_").replace(":", "-")
    if args.record and args.provider != "replay":
        rec_path = ROOT / "bench" / "replays" / f"{slug}.jsonl"
        rec_path.unlink(missing_ok=True)
        provider = RecordingProvider(provider, rec_path)

    runs = []
    for c in cases:
        r = run_case(c, provider)
        runs.append(r)
        print(f"{c['id']:7} {r['status']:8} attempts={r['attempts']} {r['latency_s']:.1f}s {r['documents']}", flush=True)
    s = score(cases, runs)
    s["model_load_s"] = round(load_s, 1)

    # Replays only re-score recorded answers: kept apart so results.md shows live runs.
    out_dir = ROOT / "results" / ("replay" if args.provider == "replay" else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.json").write_text(json.dumps({"summary": s, "runs": runs}, ensure_ascii=False, indent=1))
    (out_dir / f"{slug}.md").write_text(render(provider.name, s, cases, runs, args.provider != "replay"))
    build_results_md(ROOT / "results")
    print(json.dumps(s, indent=1))
    return 0


def build_results_md(out_dir: Path) -> None:
    """results.md = header + every live provider section in results/ (all generated).
    The summary table in README.md, between the two markers, is regenerated too."""
    sections = sorted(out_dir.glob("*.md"))
    head = ("# Benchmark results\n\n"
            "This file is generated by `python -m bench.run`; do not edit it by hand.\n"
            "Each section is one provider run over `bench/cases.yml`.\n\n")
    (ROOT / "results.md").write_text(head + "\n".join(p.read_text() for p in sections))

    rows = ["| Provider | Valid 1st try | Valid after repair | Correct refusals | False refusals "
            "| Doc precision | Doc recall | Exact sets | PII sent | Latency p50 / p95 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for js in sorted(out_dir.glob("*.json")):
        s = json.loads(js.read_text())["summary"]
        name = js.stem.replace("local-", "local: ").replace("claude-", "claude: ").replace("_", "/")
        rows.append(f"| {name} | {pct(s['valid_first_try'], s['cases'])} | {pct(s['valid_after_repair'], s['cases'])} "
                    f"| {pct(s['correct_refusals'], s['expected_refusals'])} | {pct(s['false_refusals'], s['request_cases'])} "
                    f"| {s['doc_precision']:.2f} | {s['doc_recall']:.2f} | {pct(s['exact_document_sets'], s['request_cases'])} "
                    f"| {s['pii_leaks']} | {s['latency_p50']:.1f} s / {s['latency_p95']:.1f} s |")
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        a, b = "<!-- results:start -->", "<!-- results:end -->"
        if a in text and b in text:
            text = text[:text.index(a) + len(a)] + "\n" + "\n".join(rows) + "\n" + text[text.index(b):]
            readme.write_text(text)


if __name__ == "__main__":
    raise SystemExit(main())
