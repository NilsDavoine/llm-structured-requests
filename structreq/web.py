"""A minimal local page to try the pipeline. Standard library only; binds to 127.0.0.1."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import pipeline
from .pii import PIILeak

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Structured client requests</title>
<style>
 :root { --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --line:#deded8; --accent:#1f5f8b; --bad:#a33; }
 @media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --line:#33332f; --accent:#7fb6dd; --bad:#e88; } }
 body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; }
 main { max-width:720px; margin:0 auto; padding:32px 16px; }
 h1 { font-size:1.4rem; margin:0 0 4px; } p.sub { color:var(--muted); margin:0 0 24px; }
 textarea { width:100%; box-sizing:border-box; min-height:84px; padding:10px; font:inherit;
            background:transparent; color:inherit; border:1px solid var(--line); border-radius:6px; }
 button { margin-top:8px; padding:8px 16px; font:inherit; border:0; border-radius:6px;
          background:var(--accent); color:var(--bg); cursor:pointer; }
 button:disabled { opacity:.5; }
 .meta { color:var(--muted); font-size:.9rem; margin-top:20px; }
 ol { padding-left:20px; } li { margin:8px 0; } li small { display:block; color:var(--muted); }
 .bad { color:var(--bad); } code { font-size:.85rem; }
 details { margin-top:16px; } pre { overflow-x:auto; font-size:.8rem; }
</style></head><body><main>
<h1>Describe what the client must send</h1>
<p class="sub">One sentence, in French, English or German. Personal data is removed before the model sees it.</p>
<textarea id="s">Constitution d'une SARL-S, il me faut les pièces habituelles, sous 15 jours.</textarea>
<button id="go">Prepare the request</button>
<div id="out"></div>
<script>
const out = document.getElementById('out'), go = document.getElementById('go');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
go.onclick = async () => {
  go.disabled = true; out.innerHTML = '<p class="meta">Working…</p>';
  try {
    const r = await (await fetch('/api/request', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify({sentence: document.getElementById('s').value})})).json();
    let h = `<p class="meta">Sent to ${esc(r.provider)}: <code>${esc(r.sent)}</code><br>`
          + `${r.redacted.length ? 'Removed: ' + esc(r.redacted.join(', ')) + ' · ' : ''}`
          + `attempts ${r.attempts} · ${r.latency_s.toFixed(1)} s</p>`;
    const v = r.view;
    if (v.status === 'request') {
      h += `<h2>${esc(v.title)}</h2>` + (v.due_date ? `<p>Due ${esc(v.due_date)}</p>` : '')
        + '<ol>' + v.documents.map(d => `<li>${esc(d.label)}<small>${esc(d.note)}</small></li>`).join('') + '</ol>';
    } else if (v.status === 'refused') {
      h += `<p class="bad">Refused: ${esc(v.reason === 'too_vague' ? 'too vague to know what to ask for' : 'not an accounting document request')}.</p>`;
    } else {
      h += `<p class="bad">No valid answer after one repair.</p><pre>${esc((v.errors||[]).join('\\n'))}</pre>`;
    }
    h += `<details><summary>Validated JSON</summary><pre>${esc(JSON.stringify(r.result, null, 2))}</pre></details>`;
    out.innerHTML = h;
  } catch (e) { out.innerHTML = `<p class="bad">${esc(e)}</p>`; }
  go.disabled = false;
};
</script></main></body></html>"""


def serve(provider, host: str = "127.0.0.1", port: int = 8000) -> None:
    lock = threading.Lock()   # one generation at a time on a small GPU

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/request":
                return self._send(404, b"not found", "text/plain")
            length = min(int(self.headers.get("Content-Length", 0)), 4000)
            try:
                sentence = str(json.loads(self.rfile.read(length) or b"{}").get("sentence", ""))[:1000]
                with lock:
                    out = pipeline.run(sentence, provider)
            except PIILeak as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            except ValueError:
                return self._send(400, b'{"error": "bad request"}', "application/json")
            body = {
                "provider": provider.name,
                "sent": out.sent[0]["messages"][0]["content"] if out.sent else "",
                "redacted": sorted({f.kind for f in out.redactions}),
                "attempts": out.attempts, "latency_s": out.latency_s,
                "view": out.client_view(pipeline.guess_lang(sentence)),
                "result": out.result.model_dump() if out.result else None,
            }
            self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json")

        def log_message(self, *args):   # keep sentences out of the console log
            pass

    print(f"open http://{host}:{port}  (provider: {provider.name})")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
