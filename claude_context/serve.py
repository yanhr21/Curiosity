#!/usr/bin/env python3
"""Static server for the project context page.

Serves this file's directory, plus the two things the page cannot do itself:

  /branch          JSON {branch, repo, head} read live from git, so the top bar
                   shows the real checkout instead of a hard-coded string.
  /docs/<name>     resolves a markdown file by name — the site's own docs/ dir,
                   then the site dir itself, then the repo root, then git's index
                   (submodules included).
                   That is why the docs sidebar can use bare filenames and needs
                   no symlink farm. A bare /<name>.md that is not a file here
                   resolves the same way, so sibling pages that fetch an absolute
                   markdown path keep working.

Everything is sent no-store, so edits show up on a plain reload.

    python3 serve.py                # first free port from 8082
    PORT=9000 python3 serve.py      # pin a port
    HOST=127.0.0.1 python3 serve.py # bind loopback only (default 0.0.0.0)
"""
import http.server
import json
import os
import socket
import socketserver
import subprocess
import urllib.parse

ROOT = os.path.dirname(os.path.abspath(__file__))     # the served site dir
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8082"))
PORT_TRIES = int(os.environ.get("PORT_TRIES", "20"))


def _git(*args, **kw):
    try:
        return subprocess.check_output(["git", "-C", kw.get("cwd", ROOT)] + list(args),
                                       text=True, stderr=subprocess.DEVNULL,
                                       timeout=20).strip()
    except Exception:
        return ""


def _repo_root():
    """The git toplevel, not just the parent — the site dir may be nested."""
    return os.environ.get("REPO_ROOT") or _git("rev-parse", "--show-toplevel") or os.path.dirname(ROOT)


REPO = os.path.abspath(_repo_root())


def resolve_doc(name):
    """site docs/ → repo root → git index. Absolute path, or None.

    The git lookup reads the index rather than walking the tree: instant even on
    a huge working copy, and it never stats an outputs/ directory. Untracked
    files outside the site dir therefore will not resolve — `git add` them.
    """
    name = urllib.parse.unquote(name).lstrip("/")
    if ".." in name.split("/"):
        return None
    # ROOT itself first: the skill writes operations.md into the site dir, so a
    # sidebar entry for it must resolve without a docs/ symlink existing.
    for cand in (os.path.join(ROOT, "docs", name), os.path.join(ROOT, name),
                 os.path.join(REPO, name)):
        if os.path.isfile(cand):
            return cand
    base = os.path.basename(name)
    for extra in (["--recurse-submodules"], []):     # older git lacks the flag
        listing = _git("ls-files", "--cached", *(extra + [base, "*/" + base]), cwd=REPO)
        for rel in listing.splitlines():
            p = os.path.join(REPO, rel)
            if os.path.isfile(p):
                return p
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Defeat browser caching so page edits always show on reload.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send(self, body, ctype, status=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/branch":
            return self._send(json.dumps({
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
                "repo": os.path.basename(REPO),
                "head": _git("rev-parse", "--short", "HEAD"),
            }), "application/json")

        # /docs/<name>, plus a bare /<name>.md that is not a real file in the
        # site dir. The bare form matters for sibling pages written against an
        # absolute markdown path (rebuttal.html fetching /rebuttal.md): they keep
        # working without being edited.
        if path.startswith("/docs/") or (
                path.endswith(".md")
                and not os.path.isfile(os.path.join(ROOT, path.lstrip("/")))):
            name = path[len("/docs/"):] if path.startswith("/docs/") else path.lstrip("/")
            target = resolve_doc(name)
            if target is None:
                return self._send(
                    "# Not found\n\n`%s` did not resolve in `%s/docs/`, `%s`, or git's "
                    "index. If the file is untracked, `git add` it or symlink it into "
                    "the site's `docs/`.\n" % (name, ROOT, REPO),
                    "text/markdown; charset=utf-8", 404)
            try:
                with open(target, "rb") as f:
                    return self._send(f.read(), "text/markdown; charset=utf-8")
            except OSError as e:
                return self._send("# Read error\n\n%s\n" % e,
                                  "text/markdown; charset=utf-8", 500)

        return super().do_GET()

    def log_message(self, fmt, *args):      # one tidy line, no noise
        print("  %s" % (fmt % args), flush=True)


def bind():
    socketserver.TCPServer.allow_reuse_address = True
    last = None
    for p in range(PORT, PORT + PORT_TRIES):
        try:
            return socketserver.TCPServer((HOST, p), Handler), p
        except OSError as e:
            last = e
    raise SystemExit("no free port in %d..%d (%s)" % (PORT, PORT + PORT_TRIES - 1, last))


os.chdir(ROOT)
httpd, port = bind()
print("context page   http://%s:%d/" % (socket.gethostname(), port), flush=True)
print("  serving      %s" % ROOT, flush=True)
print("  repo         %s (%s)" % (REPO, _git("rev-parse", "--abbrev-ref", "HEAD") or "no git"), flush=True)
httpd.serve_forever()
