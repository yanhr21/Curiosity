#!/usr/bin/env python3
"""Static server for the Robot Baby context page + a /branch endpoint that
auto-detects the current git branch (and repo dir name) of whatever checkout it
is served from, so the top bar reflects the real branch instead of a hard-coded
one. If the folder is not a git repo, /branch falls back to sensible defaults.

All responses are sent no-store so edits show up on a normal reload.

Serves the claude_context/ dir (this file's dir) on :8090 by default.
    python3 serve.py            # :8090
    PORT=9000 python3 serve.py  # override
"""
import http.server
import socketserver
import subprocess
import os
import json

ROOT = os.path.dirname(os.path.abspath(__file__))   # .../claude_context
REPO = os.path.dirname(ROOT)                          # repo root (parent, i.e. robot_baby/)
PORT = int(os.environ.get("PORT", "8090"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Defeat browser caching so page/markdown edits always show on reload.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/branch":
            try:
                branch = subprocess.check_output(
                    ["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                branch = "main"
            body = json.dumps({
                "branch": branch,
                "repo": "users/shengzew/" + os.path.basename(REPO),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()


# Serve from claude_context/ so the page is at / and every doc/markdown fetch is
# same-dir (no `..` traversal, which the static handler blocks).
os.chdir(ROOT)
# Bind loopback (not 0.0.0.0): the dev use case is local browsing or `ssh -L`
# port-forwarding, both of which hit 127.0.0.1 on this host. Override with HOST=0.0.0.0.
HOST = os.environ.get("HOST", "127.0.0.1")
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
    print("serving %s on %s:%d  (/branch auto-detects git branch in %s)"
          % (ROOT, HOST, PORT, REPO), flush=True)
    httpd.serve_forever()
