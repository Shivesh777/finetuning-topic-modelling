/* Layout check for the pre-read: open /?still=1&qa=1.
   Lists, per slide: text that overlaps other text, notes and sources that touch a chart,
   and anything past the margins (left edge label, right rail, bottom). The result is printed
   into <pre id="qa-out"> at the top of the page and to the console. */
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
    sl.querySelectorAll('.title, .src, .xnote, .list, ol.next li, .who-list, .meta, .panel-dark .cap, .tags').forEach(function (e) {
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
    var all = [];
    document.querySelectorAll('.slide').forEach(function (sl, i) { all = all.concat(check(sl, i + 1)); });
    var pre = document.createElement('pre'); pre.id = 'qa-out';
    pre.style.cssText = 'position:relative;z-index:9;margin:0 auto 40px;width:1920px;padding:24px;background:#fff;font:14px/1.5 monospace;white-space:pre-wrap';
    pre.textContent = 'QA ' + (all.length ? all.length + ' issue(s)\n' + all.join('\n') : 'clean');
    document.body.insertBefore(pre, document.body.firstChild);
    console.log(pre.textContent);
    document.title = 'QA done';
  }
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(function () { setTimeout(run, 50); });
})();
