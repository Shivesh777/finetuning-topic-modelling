/* Layout check for the pre-read: open /?still=1&qa=1 (every slide at its last step), or /?qa=1#12.3
   (the live deck held at slide 12, step 3: the slide on screen is checked at that step).
   Lists, per slide: text that overlaps other text, notes and sources that touch a chart, text that
   lies on top of a chart shape it does not belong to (a bar, a dot, a data line, a connector), and
   anything past the margins (left edge label, right rail, bottom). The result is printed into
   <pre id="qa-out"> at the top of the page and to the console. */
(function () {
  'use strict';
  var MARGIN = {left: 120, right: 1728, top: 290, bottom: 1036};

  function shown(e, stop) {
    for (var n = e; n && n !== stop; n = n.parentNode) {
      if (n.nodeType !== 1) continue;
      var cs = getComputedStyle(n);
      if (cs.display === 'none' || cs.visibility === 'hidden' || +cs.opacity === 0) return false;
    }
    return true;
  }
  function textBox(e) {
    var r = document.createRange(); r.selectNodeContents(e);
    var b = r.getBoundingClientRect();
    return b.width && b.height ? b : null;
  }
  function rel(b, o) { return {l: b.left - o.left, t: b.top - o.top, r: b.right - o.left, b: b.bottom - o.top}; }
  function hit(a, b, pad) { pad = pad || 2; return a.l < b.r - pad && b.l < a.r - pad && a.t < b.b - pad && b.t < a.b - pad; }
  function name(e) { var s = (e.textContent || '').replace(/\s+/g, ' ').trim(); return '"' + (s.length > 38 ? s.slice(0, 36) + '…' : s) + '"'; }

  function check(sl, num) {
    var o = sl.getBoundingClientRect(), out = [], texts = [], notes = [], shapes = [];
    sl.querySelectorAll('svg text').forEach(function (t) {
      if (!t.textContent.trim() || !shown(t, sl)) return;
      var b = t.getBoundingClientRect(); if (!b.width) return;
      texts.push({e: t, b: rel(b, o), svg: true});
    });
    sl.querySelectorAll('.title, .src, .xnote, .list, ol.next li, .who-list, .takeaway, .meta, .panel-dark .cap, .tags').forEach(function (e) {
      if (!shown(e, sl)) return;
      var b = textBox(e); if (!b) return;
      var it = {e: e, b: rel(b, o), note: /\b(src|xnote|list)\b/.test(e.className) || e.tagName === 'LI'};
      texts.push(it); if (it.note) notes.push(it);
    });
    sl.querySelectorAll('svg rect, svg circle, svg path, svg line').forEach(function (e) {
      if (e.closest('defs, pattern') || !shown(e, sl)) return;
      var b = e.getBoundingClientRect(); if (!b.width && !b.height) return;
      shapes.push({e: e, b: rel(b, o)});
    });
    // text on top of a shape it does not belong to: the shape is not the text's own box (one that contains it),
    // not a rule or axis line, not a hairline, and the text has no white halo (paint-order) that makes it legible there
    function inside(a, b, pad) { pad = pad || 0; return a.l >= b.l - pad && a.r <= b.r + pad && a.t >= b.t - pad && a.b <= b.b + pad; }
    function area(b) { return Math.max(0, b.r - b.l) * Math.max(0, b.b - b.t); }
    // does the segment (x1,y1)-(x2,y2), in slide coordinates, cross the box b?
    function segHits(x1, y1, x2, y2, b) {
      var pad = 1, l = b.l + pad, r = b.r - pad, tp = b.t + pad, bt = b.b - pad;
      if (Math.max(x1, x2) < l || Math.min(x1, x2) > r || Math.max(y1, y2) < tp || Math.min(y1, y2) > bt) return false;
      var t0 = 0, t1 = 1, dx = x2 - x1, dy = y2 - y1, edges = [[-dx, x1 - l], [dx, r - x1], [-dy, y1 - tp], [dy, bt - y1]];
      for (var i = 0; i < 4; i++) { var p = edges[i][0], q = edges[i][1]; if (p === 0) { if (q < 0) return false; } else { var u = q / p; if (p < 0) { if (u > t1) return false; if (u > t0) t0 = u; } else { if (u < t0) return false; if (u < t1) t1 = u; } } }
      return true;
    }
    // the segments of a straight path (M, L, H, V, absolute) in slide coordinates; null when it has curves
    function pathSegs(e) {
      var d = e.getAttribute('d') || '', m = e.getScreenCTM(); if (!m || /[CSQTAcsqta]/.test(d)) return null;
      var toks = d.replace(/([MLHVZmlhvz])/g, ' $1 ').trim().split(/[\s,]+/), x = 0, y = 0, cmd = '', segs = [], i = 0;
      function pt(px, py) { return [m.a * px + m.c * py + m.e - o.left, m.b * px + m.d * py + m.f - o.top]; }
      while (i < toks.length) {
        var tk = toks[i];
        if (/^[MLHVZ]$/.test(tk)) { cmd = tk; i++; if (cmd === 'Z') continue; }
        var nx = x, ny = y;
        if (cmd === 'H') { nx = +toks[i++]; } else if (cmd === 'V') { ny = +toks[i++]; } else { nx = +toks[i++]; ny = +toks[i++]; }
        if (cmd !== 'M') { var a = pt(x, y), b2 = pt(nx, ny); segs.push([a[0], a[1], b2[0], b2[1]]); }
        x = nx; y = ny;
      }
      return segs;
    }
    texts.forEach(function (t) {
      if (!t.svg || t.e.getAttribute('paint-order')) return;
      shapes.forEach(function (sh) {
        var e = sh.e, cls = e.getAttribute('class') || '', tag = e.tagName.toLowerCase();
        if (/ln-rule|ln-axis|ln-strike/.test(cls) || e.contains(t.e) || t.e.contains(e)) return;
        if (tag === 'line' && +(e.getAttribute('stroke-width') || 1) <= 1.2) return;
        if (tag === 'rect' && inside(t.b, sh.b, 4)) return;                       // its own box, or a panel behind it
        if ((tag === 'rect' || tag === 'circle') && e.getAttribute('fill') === 'none') return;   // outlines
        if (!hit(t.b, sh.b, 1)) return;
        if (tag === 'line') {                                                      // exact: does the segment cross the text?
          var m = e.getScreenCTM(), p1 = [m.a * +e.getAttribute('x1') + m.c * +e.getAttribute('y1') + m.e - o.left, m.b * +e.getAttribute('x1') + m.d * +e.getAttribute('y1') + m.f - o.top],
              p2 = [m.a * +e.getAttribute('x2') + m.c * +e.getAttribute('y2') + m.e - o.left, m.b * +e.getAttribute('x2') + m.d * +e.getAttribute('y2') + m.f - o.top];
          if (!segHits(p1[0], p1[1], p2[0], p2[1], t.b)) return;
        } else if (tag === 'path' && (e.getAttribute('fill') === 'none' || !e.getAttribute('fill'))) {
          var segs = pathSegs(e);
          if (segs) { if (!segs.some(function (sg) { return segHits(sg[0], sg[1], sg[2], sg[3], t.b); })) return; }
          else if (area(sh.b) > 60000) return;                                      // a big curve: its box says little
        }
        out.push('text over ' + tag + (cls ? '.' + cls.split(' ')[0] : '') + ': ' + name(t.e));
      });
    });
    var cover = sl.classList.contains('cover');
    texts.concat(shapes).forEach(function (it) {
      var b = it.b, inPanel = it.e.closest && it.e.closest('.panel-dark');
      if (cover && inPanel) return;
      if (b.r > MARGIN.right) out.push('past right margin (' + Math.round(b.r) + '): ' + (it.e.tagName === 'text' || !it.svg ? name(it.e) : it.e.tagName));
      if (b.l < MARGIN.left) out.push('past left margin (' + Math.round(b.l) + '): ' + name(it.e));
      if (b.b > MARGIN.bottom) out.push('below the bottom (' + Math.round(b.b) + '): ' + name(it.e));
      if (it.svg && b.t < MARGIN.top) out.push('above the exhibit (' + Math.round(b.t) + '): ' + name(it.e));
    });
    for (var i = 0; i < texts.length; i++) for (var j = i + 1; j < texts.length; j++) {
      var a = texts[i], c = texts[j];
      if (a.e.contains(c.e) || c.e.contains(a.e)) continue;
      if (hit(a.b, c.b)) out.push('text overlaps text: ' + name(a.e) + ' / ' + name(c.e));
    }
    notes.forEach(function (n) {
      shapes.forEach(function (s) { if (hit(n.b, s.b, 1)) out.push('note touches chart: ' + name(n.e) + ' / ' + s.e.tagName); });
    });
    var seen = {};
    return out.filter(function (x) { if (seen[x]) return false; seen[x] = 1; return true; }).map(function (x) { return 's' + num + '  ' + x; });
  }

  function run() {
    var all = [], still = document.documentElement.classList.contains('still');
    document.querySelectorAll('.slide').forEach(function (sl, i) { if (still || sl.classList.contains('on')) all = all.concat(check(sl, i + 1)); });
    var pre = document.createElement('pre'); pre.id = 'qa-out';
    pre.style.cssText = 'position:relative;z-index:9;margin:0 auto 40px;width:1920px;padding:24px;background:#fff;font:14px/1.5 monospace;white-space:pre-wrap';
    if (!still) pre.style.cssText += ';position:absolute;left:0;top:0;width:auto;max-width:1920px;opacity:0.001';   // live mode: keep the stage visible, the text is for the DOM dump
    pre.textContent = 'QA ' + (all.length ? all.length + ' issue(s)\n' + all.join('\n') : 'clean');
    document.body.insertBefore(pre, document.body.firstChild);
    console.log(pre.textContent);
    document.title = 'QA done';
  }
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(function () { setTimeout(run, 50); });
})();
