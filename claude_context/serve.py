#!/usr/bin/env python3
"""Static server for the Newton context site + a /branch endpoint (git auto-detect).

Serves from the **repo root** (this file's grandparent) so the Docs tab can reach
`claude_context/*.md`, the top-level `context.md`/`TODOs.md`, and `genpipe/*.md`. The page
is at `/claude_context/index.html`; open that path.

    python3 serve.py            # :8091
    PORT=9000 python3 serve.py  # override

All responses are no-store so edits show on a plain reload.
"""
import http.server
import socketserver
import subprocess
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PORT = int(os.environ.get("PORT", "8091"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        # /branch (reachable as /branch or /claude_context/branch)
        if self.path.split("?")[0].rstrip("/").endswith("/branch") or self.path.split("?")[0] == "/branch":
            try:
                branch = subprocess.check_output(
                    ["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                branch = "mike_2026_7_21_newton"
            body = json.dumps({"branch": branch, "repo": "yanhr21/Curiosity"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


os.chdir(ROOT)
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print("serving %s on 127.0.0.1:%d  →  open /claude_context/index.html  (/branch = git in %s)"
          % (ROOT, PORT, ROOT), flush=True)
    httpd.serve_forever()
