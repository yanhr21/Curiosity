/* medal_rank.js — rank each metric column of a rendered markdown table, tint the
 * best / 2nd / 3rd cells gold / silver / bronze, and give every row a "deactivate"
 * × / + toggle that excludes it from the ranking.
 *
 * Direction comes from the HEADER ARROW, the only reliable signal in these tables:
 * "IoU ↑" = larger is better, "FVD ↓" = smaller is better. A header with neither arrow
 * is skipped, which automatically excludes the label column and any Step/cfg/N column.
 *
 * Deactivating a row (click the × in the label cell; it becomes + to restore):
 *   - greys the row out,
 *   - moves it to the bottom of the table (deactivated rows keep their relative order),
 *   - re-ranks every column over the REMAINING active rows only.
 * Unchecking restores the row to its original position and re-ranks again.
 *
 * The toggle button is injected INTO the first cell rather than as a new column, so the
 * column count, the header arrows and all existing table CSS stay untouched.
 *
 * Details that matter for these tables:
 *   - cells may be **bold** or *italic*, so rank on textContent;
 *   - "-", "--", "–" and empty cells are skipped, never read as 0 (0 would otherwise win
 *     every "lower is better" column);
 *   - ties share a medal and the next DISTINCT value takes the next medal down.
 */
(function (global) {
  var CLASSES = ['medal-gold', 'medal-silver', 'medal-bronze'];

  function parseCell(td) {
    var t = (td.textContent || '').trim();
    if (!t) return null;
    if (/^[-–—]+$/.test(t)) return null;
    var m = t.replace(/,/g, '').match(/-?\d+(\.\d+)?/);
    if (!m) return null;
    return parseFloat(m[0]);
  }

  function bodyRows(table) {
    var out = [];
    for (var b = 0; b < table.tBodies.length; b++) {
      for (var r = 0; r < table.tBodies[b].rows.length; r++) out.push(table.tBodies[b].rows[r]);
    }
    return out;
  }

  /* clear medals, re-rank over active rows only */
  function rank(table) {
    var head = table.tHead;
    if (!head || !head.rows.length) return;
    var hdr = head.rows[head.rows.length - 1].cells;
    var all = bodyRows(table);
    var rows = all.filter(function (tr) { return !tr.classList.contains('row-off'); });

    all.forEach(function (tr) {
      for (var i = 0; i < tr.cells.length; i++) tr.cells[i].classList.remove.apply(
        tr.cells[i].classList, CLASSES);
    });
    if (rows.length < 2) return;

    for (var c = 0; c < hdr.length; c++) {
      var h = hdr[c].textContent || '';
      var hi = h.indexOf('↑') !== -1, lo = h.indexOf('↓') !== -1;
      if (!hi && !lo) continue;

      var vals = [];
      for (var i = 0; i < rows.length; i++) {
        var td = rows[i].cells[c];
        if (!td) continue;
        var v = parseCell(td);
        if (v === null || !isFinite(v)) continue;
        vals.push({ v: v, td: td });
      }
      if (vals.length < 2) continue;

      var distinct = vals.map(function (o) { return o.v; })
        .filter(function (v, i, a) { return a.indexOf(v) === i; })
        .sort(function (x, y) { return hi ? y - x : x - y; })
        .slice(0, CLASSES.length);

      for (var d = 0; d < distinct.length; d++) {
        for (var k = 0; k < vals.length; k++) {
          if (vals[k].v === distinct[d]) vals[k].td.classList.add(CLASSES[d]);
        }
      }
    }
  }

  /* active rows in original order, then deactivated rows in original order */
  function reorder(table) {
    var tb = table.tBodies[0];
    if (!tb) return;
    var rows = bodyRows(table).slice().sort(function (a, b) {
      return (+a.dataset.origIndex) - (+b.dataset.origIndex);
    });
    var on = rows.filter(function (r) { return !r.classList.contains('row-off'); });
    var off = rows.filter(function (r) { return r.classList.contains('row-off'); });
    on.concat(off).forEach(function (r) { tb.appendChild(r); });
  }

  function enhance(table) {
    var rows = bodyRows(table);
    if (!rows.length) return;
    rows.forEach(function (tr, i) {
      if (tr.dataset.origIndex !== undefined) return;   // already enhanced
      tr.dataset.origIndex = i;
      var first = tr.cells[0];
      if (!first) return;
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'row-toggle';
      btn.textContent = '×';                       // × = deactivate
      btn.title = 'Deactivate this row (exclude from ranking, move to bottom)';
      btn.addEventListener('click', function () {
        var off = !tr.classList.contains('row-off');
        tr.classList.toggle('row-off', off);
        btn.textContent = off ? '+' : '×';    // + = restore
        btn.title = off ? 'Restore this row' : 'Deactivate this row (exclude from ranking, move to bottom)';
        reorder(table);
        rank(table);
      });
      first.insertBefore(btn, first.firstChild);
    });
  }

  function hasRankedColumn(table) {
    var head = table.tHead;
    if (!head || !head.rows.length) return false;
    var hdr = head.rows[head.rows.length - 1].cells;
    for (var c = 0; c < hdr.length; c++) {
      var h = hdr[c].textContent || '';
      if (h.indexOf('↑') !== -1 || h.indexOf('↓') !== -1) return true;
    }
    return false;
  }

  /* medalRank(root)                     — enhance EVERY table (the default)
     medalRank(root, {onlyRanked: true}) — skip tables with no ↑/↓ header

     The default must stay "every table". A rendered-markdown page mixes metric
     tables with companion legend tables that carry no arrows — which variant has
     which feature — and deactivating a row there is still how you mute a variant
     across the whole set. Those tables keep their × toggles. Any caller that
     passes no options (a standalone markdown page, the docs tab, the findings
     tab) therefore behaves exactly as it always has.

     onlyRanked is for hand-authored views that sit prose comparison tables next
     to the metric table, where a toggle on a prose table is pure clutter. */
  global.medalRank = function (root, opts) {
    if (!root) return;
    var onlyRanked = !!(opts && opts.onlyRanked);
    var tables = root.querySelectorAll('table');
    for (var i = 0; i < tables.length; i++) {
      try {
        if (onlyRanked && !hasRankedColumn(tables[i])) continue;
        enhance(tables[i]);
        rank(tables[i]);
      } catch (e) { /* never break rendering */ }
    }
  };
})(window);
