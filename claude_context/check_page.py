#!/usr/bin/env python3
"""Static contract check for a context-site page.

Catches the failures that render fine and are silently dead: a diagram whose
nodes cannot be clicked, a panel map pointing at an id that does not exist, an
unbalanced tag that swallows half the page, a metric column with no ↑/↓ so it
never ranks, or the example's content left in place describing the wrong repo.

    python3 check_page.py claude_context/index.html

Exit 0 = clean. Everything it reports is a real defect, not a style opinion.
"""
import html.parser
import os
import re
import sys

VOID = {"meta", "link", "br", "img", "input", "hr", "col", "source", "area", "base", "wbr"}
SVG_SELF = {"path", "polygon", "polyline", "circle", "ellipse", "line", "rect", "use", "stop"}

# Distinctive strings from the shipped example. Their presence means content was
# copied but not rewritten — a page confidently describing someone else's repo.
EXAMPLE_TOKENS = ["Chirp", "Conformer-CTC", "conformer_960h", "lora_r16_960h",
                  "log-mel", "test-clean", "finetune_lora"]

fails = []
warns = []


def ok(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def warn(cond, msg):
    print(("  ok    " if cond else "  warn  ") + msg)
    if not cond:
        warns.append(msg)


class Balance(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.bad = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID and tag not in SVG_SELF:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID or tag in SVG_SELF:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.bad.append("unclosed <%s> before </%s>" % (self.stack.pop(), tag))
            if self.stack:
                self.stack.pop()
        else:
            self.bad.append("stray </%s>" % tag)


def main(path, is_example=False):
    raw = open(path).read()
    base = os.path.dirname(os.path.abspath(path))

    # Analyse the page WITHOUT its comments. The template's instructional
    # comments legitimately contain `<col>`, `{{PLACEHOLDER}}` and sample markup;
    # counting those produces false failures, and nothing in a comment renders.
    src = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    # The entry point lives one level up from the site dir, at the repo root —
    # that is where /recall-context looks, so that is where it gets checked.
    root_ctx = os.path.join(os.path.dirname(os.path.abspath(base)), "context.md")

    print("checking %s" % path)

    # --- leftover example content ----------------------------------------
    # The worst thing this page can be is a confident description of the wrong
    # repo. The example carries a sentinel attribute; removing it is the
    # deliberate act of saying "this is mine now".
    if is_example:
        ok("data-example=" in src, "example sentinel present (self-test mode)")
    else:
        ok("data-example=" not in src,
           "example sentinel removed from <body> (page converted to this repo)")
        stale = sorted({t for t in EXAMPLE_TOKENS if re.search(re.escape(t), src, re.I)})
        warn(not stale, "no example text left behind%s"
             % (" — found: " + ", ".join(stale[:8]) if stale else ""))
        # Step 1 copies the example's markdown in alongside the page, so a
        # half-converted site can ship someone else's operating notes while
        # index.html looks perfectly clean. Scan the siblings too — and the
        # repo-root context.md, which lives one level UP and is therefore
        # invisible to a scan of the site dir alone.
        scan = [os.path.join(base, f)
                for f in sorted(os.listdir(base)) if f.endswith(".md")]
        if os.path.exists(root_ctx):
            scan.append(root_ctx)
        for md in scan:
            body = open(md, errors="replace").read()
            hit = sorted({t for t in EXAMPLE_TOKENS if re.search(re.escape(t), body, re.I)})
            warn(not hit, "%s carries no example text%s"
                 % (os.path.relpath(md, base), " — found: " + ", ".join(hit[:6]) if hit else ""))

    # --- tag balance ------------------------------------------------------
    p = Balance()
    p.feed(src)
    problems = p.bad + ["unclosed <%s>" % t for t in p.stack]
    ok(not problems, "tag balance (%s)" % ("; ".join(problems)[:200] or "clean"))

    ids = set(re.findall(r'\bid="([^"]+)"', src))

    # --- ids app.js reaches for by name -----------------------------------
    for r in ["view-main", "view-findings", "docs-content",
              "docs-pathbar-text", "docs-raw-link", "findings-content",
              "findings-toolbar", "flow-pop", "repo-path", "repo-branch"]:
        ok(r in ids, "app.js target #%s present" % r)

    ok('data-tab="findings"' in src, 'nav has data-tab="findings"')
    ok('class="tab-count"' in src, "findings nav has a .tab-count badge")
    # Docs lives in the main flow and is reached by anchor, not by tab switch.
    ok('id="docs"' in src and 'href="#docs"' in src,
       "docs section is anchored from the nav (#docs)")

    # --- operations ------------------------------------------------------
    # The most-opened part of the page, and the easiest to skip: a page can be
    # architecturally perfect and still leave a reader unable to launch anything.
    warn('id="run"' in src,
         "page has a Run section (#run) — how to launch, on this cluster")
    ops_path = os.path.join(base, "operations.md")
    ops = os.path.exists(ops_path)
    warn(ops, "operating notes exist (%s)" % ops_path)
    warn(not ops or 'data-file="operations.md"' in src,
         "operations.md is linked from the docs sidebar")
    if ops:
        ops_src = open(ops_path, errors="replace").read()
        generated = "BEGIN GENERAL" in ops_src
        warn(generated,
             "operations.md carries the generated general block "
             "(make_operations.py <site>; --check tells you if it is stale)")
        if generated:
            # Only meaningful once the block exists — a hand-written file that
            # predates the generator has no "## This repo" heading and the
            # warning above already covers it. Three or more TODOs means the
            # skeleton was copied and never filled, which is the failure that
            # actually happens.
            tail = ops_src.split("END GENERAL -->", 1)[-1]
            warn("## This repo" in tail and tail.count("TODO") < 3,
                 "the repo-specific half of operations.md is written, not skeleton TODOs")

    # --- the /recall-context contract -------------------------------------
    # These two filenames are read BY NAME in a future session. A repo that
    # invents its own gets neither loaded, and nobody finds out until it costs
    # something. context.md belongs at the repo ROOT, which is why it is checked
    # there rather than in the site dir.
    # Skipped in self-test mode: the skill directory is not a repo, so warning
    # that it has no entry point is noise, and noise is how warnings stop
    # getting read.
    if not is_example:
        ctx = root_ctx if os.path.exists(root_ctx) else os.path.join(base, "context.md")
        have_ctx = os.path.exists(ctx)
        warn(have_ctx, "entry point context.md exists (%s)" % (ctx if have_ctx else root_ctx))
        if have_ctx:
            warn("operations.md" in open(ctx, errors="replace").read(),
                 "context.md points at operations.md — otherwise the manual is unreachable")

    # --- diagram wiring ---------------------------------------------------
    block = re.search(r"window\.FLOW_DIAGRAMS\s*=\s*\[(.*?)\];", src, re.S)
    ok(block is not None, "FLOW_DIAGRAMS declared")
    wired_svgs = set()
    if block:
        body = block.group(1)
        for svg_id in re.findall(r"svg:\s*'([^']+)'", body):
            wired_svgs.add(svg_id)
            ok(svg_id in ids, "FLOW_DIAGRAMS svg id '%s' exists" % svg_id)
        for panel in re.findall(r"'[^']+'\s*:\s*'([^']+)'", body):
            ok(("panel-" + panel) in ids, "panel map target #panel-%s exists" % panel)

    # every node in a wired SVG needs a <title>, or it is not clickable
    for m in re.finditer(r'<svg\b[^>]*id="([^"]+)"(.*?)</svg>', src, re.S):
        svg_id, svg_body = m.group(1), m.group(2)
        nodes = re.findall(r'<g[^>]*class="node"[^>]*>(.*?)</g>', svg_body, re.S)
        if not nodes:
            continue
        missing = sum(1 for n in nodes if "<title>" not in n)
        if svg_id in wired_svgs:
            ok(not missing, "svg #%s: %d/%d nodes carry <title>"
               % (svg_id, len(nodes) - missing, len(nodes)))
        else:
            warn(False, "svg #%s is not in FLOW_DIAGRAMS — nothing in it is clickable" % svg_id)

    # --- diagram geometry -------------------------------------------------
    # Two silent killers. A node whose box lies outside the viewBox is CLIPPED:
    # the reader sees a stage that stops mid-air, or an edge that ends nowhere,
    # and reads it as a real dead end — the diagram asserts something false.
    # Two nodes drawn on top of each other OCCLUDE: whichever is painted second
    # simply hides the first, and nobody notices a stage went missing.
    # app.js re-fits the viewBox at runtime, but that only helps where JS runs;
    # authored coordinates should be right on their own.
    for m in re.finditer(r'<svg\b[^>]*id="([^"]+)"[^>]*viewBox="([^"]+)"(.*?)</svg>',
                         src, re.S):
        svg_id, vb, svg_body = m.group(1), m.group(2), m.group(3)
        try:
            vx, vy, vw, vh = (float(t) for t in vb.split())
        except ValueError:
            continue
        # Graphviz wraps the drawing in <g class="graph" transform="scale(..) rotate(..)
        # translate(..)"> and emits node coordinates in the PRE-transform space (negative
        # y). Comparing those raw against the viewBox reports every node as clipped — a
        # false alarm that is worse than no check, because it trains you to ignore it.
        # Apply the graph transform first. If the graph is rotated, skip containment
        # rather than guess; overlap is unaffected by a uniform transform.
        sx = sy = 1.0
        tx = ty = rot = 0.0
        gm = re.search(r'<g[^>]*class="graph"[^>]*transform="([^"]+)"', svg_body)
        if gm:
            tf = gm.group(1)
            ms = re.search(r"scale\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)", tf)
            if ms:
                sx, sy = float(ms.group(1)), float(ms.group(2))
            else:
                ms = re.search(r"scale\(\s*(-?[\d.]+)\s*\)", tf)
                if ms:
                    sx = sy = float(ms.group(1))
            mr = re.search(r"rotate\(\s*(-?[\d.]+)", tf)
            if mr:
                rot = float(mr.group(1))
            mt = re.search(r"translate\(\s*(-?[\d.]+)[ ,]+(-?[\d.]+)\s*\)", tf)
            if mt:
                tx, ty = float(mt.group(1)), float(mt.group(2))
        boxes = []
        for g in re.findall(r'<g[^>]*class="node"[^>]*>(.*?)</g>', svg_body, re.S):
            t = re.search(r"<title>([^<]*)</title>", g)
            xs, ys = [], []
            for pts in re.findall(r'points="([^"]+)"', g):
                v = [float(p) for p in re.split(r"[ ,]+", pts.strip()) if p]
                xs += v[0::2]; ys += v[1::2]
            for d in re.findall(r' d="([^"]+)"', g):
                # absolute M/L/H/V only; an arc's radii are not coordinates
                for cmd, args in re.findall(r"([MLHVA])([^A-Za-z]*)", d):
                    v = [float(n) for n in re.findall(r"-?\d+\.?\d*", args)]
                    if cmd in "ML":
                        xs += v[0::2]; ys += v[1::2]
                    elif cmd == "H":
                        xs += v
                    elif cmd == "V":
                        ys += v
                    elif cmd == "A" and len(v) >= 7:
                        xs.append(v[5]); ys.append(v[6])
            if xs and ys:
                xs = [sx * (x + tx) for x in xs]
                ys = [sy * (y + ty) for y in ys]
                boxes.append((t.group(1) if t else "?",
                              min(xs), min(ys), max(xs), max(ys)))
        if not boxes:
            continue
        outside = [] if rot else [n for n, a, b, c, d in boxes
                   if a < vx - 0.5 or c > vx + vw + 0.5
                   or b < vy - 0.5 or d > vy + vh + 0.5]
        ok(not outside, "svg #%s: every node is inside the viewBox%s"
           % (svg_id, " (clipped: " + ", ".join(outside[:4]) + ")" if outside else ""))
        hits = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                n1, a1, b1, c1, d1 = boxes[i]
                n2, a2, b2, c2, d2 = boxes[j]
                if min(c1, c2) - max(a1, a2) > 1 and min(d1, d2) - max(b1, b2) > 1:
                    hits.append("%s∩%s" % (n1, n2))
        ok(not hits, "svg #%s: no node overlaps another%s"
           % (svg_id, " (" + ", ".join(hits[:4]) + ")" if hits else ""))

    # panels that exist but nothing points at them
    if block:
        mapped = set(re.findall(r"'[^']+'\s*:\s*'([^']+)'", block.group(1)))
        orphan = sorted(i[len("panel-"):] for i in ids
                        if i.startswith("panel-") and i[len("panel-"):] not in mapped)
        warn(not orphan, "every detail panel is reachable from a diagram node%s"
             % (" (unreachable: " + ", ".join(orphan[:6]) + ")" if orphan else ""))

    # --- metric tables ----------------------------------------------------
    for m in re.finditer(r'<table class="variants">(.*?)</table>', src, re.S):
        head = re.search(r"<thead>(.*?)</thead>", m.group(1), re.S)
        if not head:
            continue
        headers = re.findall(r"<th[^>]*>(.*?)</th>", head.group(1), re.S)
        ranked = [h for h in headers if "↑" in h or "↓" in h]
        ok(bool(ranked), "variant table has >=1 ranked column (↑/↓ in the header) — %d of %d"
           % (len(ranked), len(headers)))
        rows = re.findall(r"<tr[^>]*>", m.group(1))
        warn(len(rows) >= 3, "variant table has >=2 data rows (ranking needs a comparison)")

    # --- grid arity: silent layout breakage, invisible in a diff -----------
    for m in re.finditer(r'<table class="diff">(.*?)</table>', src, re.S):
        cols = len(re.findall(r"<col\b", m.group(1)))
        head = re.search(r"<thead>(.*?)</thead>", m.group(1), re.S)
        ths = len(re.findall(r"<th\b", head.group(1))) if head else 0
        if cols:
            ok(cols == ths, "comparison table: %d <col> vs %d <th>" % (cols, ths))

    for m in re.finditer(r'<div class="strip">(.*?)</div>\s*</header>', src, re.S):
        cells = len(re.findall(r'<div class="cell', m.group(1)))
        ok(cells == 4, "cover strip has 4 cells (3 facts + legend), got %d — the "
                       "grid is fixed at 4 columns" % cells)

    for m in re.finditer(r'style="--ncol:(\d+)"', src):
        n = int(m.group(1))
        warn(1 <= n <= 6, "--ncol=%d is in range" % n)

    # --- assets -----------------------------------------------------------
    for asset in re.findall(r'(?:src|href)="((?!http|#|mailto)[^"]+)"', src):
        if asset.startswith("docs/") or asset == "branch":
            continue
        ok(os.path.exists(os.path.join(base, asset)), "asset %s exists" % asset)

    for extra in ("findings.md", "serve.py"):
        ok(os.path.exists(os.path.join(base, extra)), "%s present in the site dir" % extra)

    print()
    if fails:
        print("%d FAILED, %d warning(s)" % (len(fails), len(warns)))
        return 1
    print("all contract checks passed%s" % (", %d warning(s)" % len(warns) if warns else ""))
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--example"]
    if len(args) != 1:
        sys.exit("usage: check_page.py [--example] <index.html>\n"
                 "  --example  self-test the shipped example, which still carries its sentinel")
    sys.exit(main(args[0], is_example="--example" in sys.argv))
