/* app.js — wiring for the project context page.
 *
 *   1. TABS      main / docs / findings. Only one .view is visible; the docs and
 *                findings views are lazy — nothing is fetched until first click.
 *   2. DOCS      sidebar links load markdown through marked.js, then medalRank().
 *   3. FINDINGS  renders findings.md, groups it into one card per `### ` heading,
 *                colours each by verdict, and adds a verdict filter bar.
 *   4. FLOW      clicking a node in an architecture SVG opens a popover holding
 *                that stage's detail panel. Contract: the SVG's <g class="node">
 *                must contain a <title> whose text is a key in the diagram's
 *                panel map (Graphviz emits these automatically; hand-written SVGs
 *                must add them — a node with no <title> is simply not clickable).
 *   5. REPO      top-bar path + branch come from serve.py's /branch endpoint.
 *
 * No build step, no framework. Everything degrades to "still readable" if a
 * fetch fails.
 */
(function () {
  'use strict';

  var MD_OPTS = { gfm: true, breaks: false, headerIds: true };
  function renderMd(md) {
    if (window.marked && window.marked.setOptions) window.marked.setOptions(MD_OPTS);
    return window.marked ? window.marked.parse(md) : '<pre>' + md.replace(/[<&]/g, function (c) {
      return c === '<' ? '&lt;' : '&amp;';
    }) + '</pre>';
  }
  function get(id) { return document.getElementById(id); }

  /* ============================================================ */
  /* 1. TABS                                                      */
  /* ============================================================ */
  /* Docs is a section inside the main view, not a tab — Findings is the only
     thing hidden behind a click. */
  var TABS = ['main', 'findings'];
  var loaded = {};

  function showTab(name, push) {
    if (TABS.indexOf(name) === -1) name = 'main';
    TABS.forEach(function (t) {
      var v = get('view-' + t);
      if (v) v.hidden = (t !== name);
    });
    document.querySelectorAll('nav.topbar a[data-tab]').forEach(function (a) {
      a.classList.toggle('on', a.getAttribute('data-tab') === name && name !== 'main');
    });
    if (name === 'findings' && !loaded.findings) { loaded.findings = true; initFindings(); }
    if (push) {
      var h = (name === 'main') ? (location.pathname + location.search) : '#' + name;
      if (window.history && history.replaceState) history.replaceState(null, '', h);
      window.scrollTo(0, 0);   // only on an explicit tab click — never on load,
    }                          // or a deep link like #map would lose its anchor
  }

  document.querySelectorAll('nav.topbar a[data-tab]').forEach(function (a) {
    a.addEventListener('click', function (ev) {
      ev.preventDefault();
      showTab(a.getAttribute('data-tab'), true);
    });
  });

  /* In-page anchors belong to the main view — jump back to it first. */
  document.querySelectorAll('nav.topbar a[href^="#"]:not([data-tab])').forEach(function (a) {
    a.addEventListener('click', function () {
      var main = get('view-main');
      if (main && main.hidden) showTab('main', false);
    });
  });

  /* ============================================================ */
  /* 2. DOCS TAB                                                  */
  /* ============================================================ */
  function initDocs() {
    var sidebar = document.querySelector('.docs-sidebar');
    var content = get('docs-content');
    var pathbar = get('docs-pathbar-text');
    var rawLink = get('docs-raw-link');
    if (!sidebar || !content) return;

    function loadDoc(file) {
      content.innerHTML = '<div class="docs-loading">Loading ' + file + '&hellip;</div>';
      if (pathbar) pathbar.textContent = 'docs/' + file;
      if (rawLink) rawLink.setAttribute('href', 'docs/' + file);
      fetch('docs/' + file, { cache: 'no-store' })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); })
        .then(function (md) {
          content.innerHTML = renderMd(md);
          if (window.medalRank) window.medalRank(content);
        })
        .catch(function (err) {
          content.innerHTML = '<div class="docs-error">Failed to load ' + file + ': ' + err.message +
            '<br>Is serve.py running, and does the file exist in the repo?</div>';
        });
    }

    sidebar.querySelectorAll('a.doc-link').forEach(function (link) {
      link.addEventListener('click', function (ev) {
        ev.preventDefault();
        var file = link.getAttribute('data-file');
        if (!file) return;
        sidebar.querySelectorAll('a.doc-link').forEach(function (a) { a.classList.remove('active'); });
        link.classList.add('active');
        loadDoc(file);
      });
    });

    var initial = sidebar.querySelector('a.doc-link.active') || sidebar.querySelector('a.doc-link');
    if (initial) loadDoc(initial.getAttribute('data-file'));
  }

  /* ============================================================ */
  /* 3. FINDINGS TAB                                              */
  /* ============================================================ */
  /* Each finding is a `### F-0007 · CONFIRMED · <variant> · <module>` heading
     plus everything up to the next `###`. Verdict = 2nd bullet-separated field. */
  var VERDICTS = ['confirmed', 'corrected', 'refuted', 'unverified', 'open'];

  function groupFindings(root) {
    var cards = [];
    var kids = Array.prototype.slice.call(root.children);
    var cur = null;
    kids.forEach(function (el) {
      if (el.tagName === 'H3') {
        cur = document.createElement('div');
        cur.className = 'finding';
        root.insertBefore(cur, el);
        cur.appendChild(el);
        cards.push(cur);
      } else if (el.tagName === 'H1' || el.tagName === 'H2' || el.tagName === 'HR') {
        cur = null;    // a rule or a round heading ends the card; without this,
      } else if (cur) { // later blocks jump *above* the rule into the card
        cur.appendChild(el);
      }
    });
    return cards;
  }

  function decorate(card) {
    var h = card.querySelector('h3');
    if (!h) return null;
    var parts = h.textContent.split('·').map(function (s) { return s.trim(); });
    var id = parts[0] || '';
    var verdict = (parts[1] || '').toLowerCase();
    if (VERDICTS.indexOf(verdict) === -1) {
      /* Separator or field order not as documented — recover the verdict from
         anywhere in the heading rather than dumping everything into "open". */
      var whole = h.textContent.toLowerCase();
      verdict = VERDICTS.filter(function (v) {
        return new RegExp('\\b' + v + '\\b').test(whole);
      })[0] || 'open';
    }
    var rest = parts.slice(2).join(' · ');

    card.classList.add('f-' + verdict);
    card.dataset.verdict = verdict;
    h.innerHTML = '<span class="fid">' + id + '</span>' +
      '<span class="verdict-chip v-' + verdict + '">' + verdict + '</span>' +
      (rest ? '<span class="fmeta">' + rest + '</span>' : '');
    return verdict;
  }

  /* One fetch, shared by the badge and the tab. */
  var findingsMd = fetch('findings.md', { cache: 'no-store' })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); });

  function initFindings() {
    var content = get('findings-content');
    var bar = get('findings-toolbar');
    if (!content) return;
    findingsMd
      .then(function (md) {
        content.innerHTML = renderMd(md);
        var cards = groupFindings(content);
        var counts = {};
        cards.forEach(function (c) {
          var v = decorate(c);
          if (v) counts[v] = (counts[v] || 0) + 1;
        });
        if (window.medalRank) window.medalRank(content);
        buildFilter(bar, cards, counts);
        setBadge(cards.length);
      })
      .catch(function (err) {
        content.innerHTML = '<div class="docs-error">No findings log yet (' + err.message +
          '). The audit pass writes <code>findings.md</code> next to this page.</div>';
      });
  }

  function buildFilter(bar, cards, counts) {
    if (!bar) return;
    bar.innerHTML = '<span class="fk">Verdict</span>';
    var total = cards.length;
    var buttons = [];

    function apply(key) {
      cards.forEach(function (c) { c.hidden = (key !== 'all' && c.dataset.verdict !== key); });
      buttons.forEach(function (b) { b.classList.toggle('on', b.dataset.key === key); });
      var shown = cards.filter(function (c) { return !c.hidden; }).length;
      var cnt = bar.querySelector('.count');
      if (cnt) cnt.textContent = shown + ' of ' + total + ' shown';
    }

    ['all'].concat(VERDICTS.filter(function (v) { return counts[v]; })).forEach(function (key) {
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.key = key;
      b.textContent = key === 'all' ? 'all ' + total : key + ' ' + counts[key];
      b.addEventListener('click', function () { apply(key); });
      bar.appendChild(b);
      buttons.push(b);
    });
    var c = document.createElement('span');
    c.className = 'count';
    bar.appendChild(c);
    apply('all');
  }

  /* the nav badge is the only hint the tab exists before you click it */
  function setBadge(n) {
    document.querySelectorAll('nav.topbar a[data-tab="findings"] .tab-count').forEach(function (el) {
      el.textContent = n ? String(n) : '';
    });
  }

  /* Populate the badge on load without rendering the tab — the tab itself stays
     lazy. Fenced code blocks must be stripped first: findings.md documents its
     own `### F-0001 · …` format inside a fence, and counting that would make the
     badge disagree with the number of cards the tab actually shows. */
  findingsMd
    .then(function (md) {
      var m = md.replace(/```[\s\S]*?```/g, '').match(/^###\s+/gm);
      setBadge(m ? m.length : 0);
    })
    .catch(function () {});

  /* ============================================================ */
  /* 4. ARCHITECTURE FLOW DIAGRAMS                                */
  /* ============================================================ */
  /* window.FLOW_DIAGRAMS is declared in the page:
       [{ svg: 'ltxflow', accent: 'a3', panels: { 'SA': 'ltx-dit', ... } }]
     key   = the <title> text of a node in that SVG
     value = the id suffix of the detail panel (<details id="panel-ltx-dit">) */
  var pop = get('flow-pop');

  function hidePop() { if (pop) { pop.hidden = true; pop.innerHTML = ''; } }

  function showPop(panelId, nodeG, accent) {
    var panel = get('panel-' + panelId);
    if (!panel || !pop) return;
    pop.className = 'flow-pop' + (accent ? ' ' + accent : '');
    var sum = panel.querySelector('summary');
    var body = panel.querySelector('.nb');
    var title = sum ? (sum.childNodes[0] ? sum.childNodes[0].textContent.trim() : sum.textContent.trim()) : '';
    pop.innerHTML =
      '<button class="flow-pop-x" aria-label="Close">&times;</button>' +
      '<div class="flow-pop-h">' + title + '</div>' +
      '<div class="flow-pop-b">' + (body ? body.innerHTML : '') + '</div>';
    /* reveal off-screen to measure, then place beside the node, clamped to the viewport */
    pop.style.visibility = 'hidden';
    pop.hidden = false;
    var r = nodeG.getBoundingClientRect();
    var pw = pop.offsetWidth, ph = pop.offsetHeight, M = 8;
    var x = r.right + 12;
    if (x + pw > window.innerWidth - M) x = r.left - pw - 12;   // flip left if no room
    if (x < M) x = M;
    var y = r.top + r.height / 2 - ph / 2;
    if (y + ph > window.innerHeight - M) y = window.innerHeight - ph - M;
    if (y < M) y = M;
    pop.style.left = x + 'px';
    pop.style.top = y + 'px';
    pop.style.visibility = 'visible';
    pop.querySelector('.flow-pop-x').addEventListener('click', hidePop);
  }

  /* ---------------------------------------------------------------- */
  /* DIAGRAM SIZING — fit the frame to the drawing, never the reverse   */
  /* ---------------------------------------------------------------- */
  /* Hand-authored SVG coordinates drift as nodes and edges get added, and a
     viewBox that no longer contains them clips SILENTLY. That is the worst
     failure mode on this page: a truncated edge reads as a real dead end, so
     the diagram asserts something false rather than merely looking broken.

     getBBox() returns the union of what is actually rendered, so the frame can
     follow the drawing instead of the drawing having to fit a number someone
     typed. Two rules:

       1. viewBox comes from the content, always.
       2. The RENDER SCALE is fixed. Labels are authored assuming one SVG unit
          ≈ one px, so with width:100% a wider viewBox shrinks the type —
          widening a frame to stop a clip would make it unreadable. An explicit
          pixel width decouples the two: a wider viewBox gives a wider BOX,
          never a smaller drawing. Give the column room (or let it scroll);
          do not scale the diagram down to fit. */
  var FLOW_SCALE = 1.3;   /* css px per SVG unit */
  var FLOW_PAD = 6;

  function fitFlowSvg(svg) {
    try {
      var g = svg.querySelector('g.graph') || svg;
      var b = g.getBBox();
      if (!b || !b.width || !b.height) return false;   /* not laid out yet */
      var w = b.width + FLOW_PAD * 2, h = b.height + FLOW_PAD * 2;
      svg.setAttribute('viewBox', (b.x - FLOW_PAD).toFixed(1) + ' ' +
        (b.y - FLOW_PAD).toFixed(1) + ' ' + w.toFixed(1) + ' ' + h.toFixed(1));
      svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
      svg.style.width = (w * FLOW_SCALE).toFixed(0) + 'px';
      svg.style.height = (h * FLOW_SCALE).toFixed(0) + 'px';
      svg.style.maxWidth = 'none';
      return true;
    } catch (e) { return false; }
  }

  function initFlowSizing() {
    var svgs = document.querySelectorAll('svg.flow-svg');
    if (!svgs.length) return;
    function fitAll() { svgs.forEach(fitFlowSvg); }
    fitAll();
    window.addEventListener('load', fitAll);
    window.addEventListener('resize', fitAll);
    /* A carousel column that is display:none reports a 0×0 bbox; re-fit the
       moment it becomes visible. */
    if (window.IntersectionObserver) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) fitFlowSvg(e.target); });
      });
      svgs.forEach(function (s) { io.observe(s); });
    }
  }

  function initFlow() {
    if (!pop || !window.FLOW_DIAGRAMS) return;
    window.FLOW_DIAGRAMS.forEach(function (d) {
      var svg = get(d.svg);
      if (!svg) { console.warn('[flow] no SVG with id ' + d.svg); return; }
      var wired = 0;
      svg.querySelectorAll('g.node').forEach(function (g) {
        var t = g.querySelector('title');
        var panel = t && d.panels[t.textContent.trim()];
        if (!panel) return;
        if (!get('panel-' + panel)) { console.warn('[flow] ' + d.svg + ': no #panel-' + panel); return; }
        g.classList.add('gv-clickable');
        g.addEventListener('click', function (ev) { ev.stopPropagation(); showPop(panel, g, d.accent); });
        wired++;
      });
      if (!wired) console.warn('[flow] ' + d.svg + ': 0 clickable nodes — do its <g class="node"> have <title> children?');
    });
    document.addEventListener('click', function (ev) { if (!pop.hidden && !pop.contains(ev.target)) hidePop(); });
    document.addEventListener('keydown', function (ev) { if (ev.key === 'Escape') hidePop(); });
    window.addEventListener('scroll', hidePop, true);
    window.addEventListener('resize', hidePop);
  }

  /* open a detail panel directly (used by prose links) */
  window.flowDetail = function (id) {
    var el = get('panel-' + id);
    if (!el) return;
    el.open = true;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  /* ============================================================ */
  /* 5. REPO IDENTITY                                             */
  /* ============================================================ */
  fetch('branch', { cache: 'no-store' })
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (d) {
      if (!d) return;
      var b = get('repo-branch'); if (b && d.branch) b.textContent = d.branch;
      var p = get('repo-path');   if (p && d.repo)   p.textContent = d.repo;
    })
    .catch(function () {});

  /* ============================================================ */
  /* THEME — one attribute on <html>; the skin is pure CSS.       */
  /* The saved value is applied by an inline script in <head>, so */
  /* this only has to keep the control in sync and persist picks. */
  (function () {
    var picks = document.querySelectorAll('.theme-pick button[data-theme-set]');
    if (!picks.length) return;
    function sync() {
      var cur = document.documentElement.getAttribute('data-theme') || '';
      picks.forEach(function (b) { b.classList.toggle('on', b.getAttribute('data-theme-set') === cur); });
    }
    picks.forEach(function (b) {
      b.addEventListener('click', function () {
        var t = b.getAttribute('data-theme-set');
        if (t) document.documentElement.setAttribute('data-theme', t);
        else document.documentElement.removeAttribute('data-theme');
        try { localStorage.setItem('site-theme', t); } catch (e) {}
        sync();
      });
    });
    sync();
  })();

  initFlow();
  initFlowSizing();   // frame follows the drawing; scale stays fixed
  initDocs();     // in the main flow now, so it loads with the page

  /* The variant selection table lives in the main view, so it needs its own
     medalRank() call — the docs and findings tabs rank their own content when
     they load. Without this the table renders but never ranks or toggles.
     onlyRanked here, and deliberately NOT in the tabs: this view also holds
     prose comparison tables that have nothing to rank, whereas a markdown page
     mixes metric tables with legend tables that must keep their × toggles. */
  if (window.medalRank) window.medalRank(get('view-main'), { onlyRanked: true });

  showTab((location.hash || '').replace('#', ''), false);
  window.addEventListener('hashchange', function () {
    var h = (location.hash || '').replace('#', '');
    if (TABS.indexOf(h) !== -1) showTab(h, false);
  });
})();
