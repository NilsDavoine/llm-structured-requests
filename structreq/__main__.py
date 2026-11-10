"""Command line.

    python -m structreq "Constitution d'une SARL-S, il me faut les pièces habituelles"
    python -m structreq --provider claude "VAT return Q3, usual documents"
    python -m structreq serve --port 8000
"""
from __future__ import annotations

import argparse
import json
import sys

from . import pipeline
from .providers import make_provider


def _provider(args):
    kwargs = {"model": args.model} if args.model else {}
    if args.provider == "replay":
        kwargs["path"] = args.replay_file
    return make_provider(args.provider, **kwargs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="structreq", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="+", help="the accountant's sentence, or 'serve'")
    ap.add_argument("--provider", default="local", choices=["local", "claude", "replay"])
    ap.add_argument("--model", help="model id (local: Hugging Face id; claude: model name)")
    ap.add_argument("--replay-file", default="bench/replays/local-Qwen_Qwen2.5-1.5B-Instruct.jsonl")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--json", action="store_true", help="print the raw validated JSON")
    args = ap.parse_args(argv)

    if args.sentence == ["serve"]:
        from .web import serve
        serve(_provider(args), port=args.port)
        return 0

    sentence = " ".join(args.sentence)
    provider = _provider(args)
    out = pipeline.run(sentence, provider)
    lang = pipeline.guess_lang(sentence)

    redacted = sorted({f.kind for f in out.redactions})
    print(f"sent to {provider.name}: {out.sent[0]['messages'][0]['content'].splitlines()[0]}", file=sys.stderr)
    if redacted:
        print(f"removed before sending: {', '.join(redacted)}", file=sys.stderr)
    print(f"attempts: {out.attempts}  latency: {out.latency_s:.1f}s", file=sys.stderr)
    if args.json and out.result is not None:
        print(out.result.model_dump_json(indent=2))
    else:
        print(json.dumps(out.client_view(lang), ensure_ascii=False, indent=2))
    return 0 if out.status != "failed" else 2


if __name__ == "__main__":
    sys.exit(main())
