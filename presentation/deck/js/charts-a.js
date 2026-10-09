/* Exhibits, part A: helpers, the news map hook, section 1 and the backups B8, B9 and B12 (sections 2 and 3 are in part D).
   Every number is read from window.DATA (tools/build_data.py). Each chart has
   enter(box, step), step(box, k), leave(box) and still(box): still draws the last frame. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var C = window.CHARTS = window.CHARTS || {};
  var D = function () { return window.DATA; };

  // ------------------------------------------------------------------ helpers (shared with part B)
  function el(tag, a, parent) { var e = document.createElementNS(NS, tag); for (var k in a) if (a[k] != null) e.setAttribute(k, a[k]); if (parent) parent.appendChild(e); return e; }
  function tx(parent, x, y, s, a) { var t = el('text', Object.assign({x: x, y: y}, a || {}), parent); t.textContent = s; return t; }
  function surface(box) { box.innerHTML = ''; var w = box.offsetWidth || 1380, h = box.offsetHeight || 660; return {s: el('svg', {width: w, height: h, viewBox: '0 0 ' + w + ' ' + h}, box), w: w, h: h}; }
  function lin(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); }; }
  function ease(u) { return 1 - Math.pow(1 - u, 3); }
  var CALM = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function tween(box, ms, fn, done) {
    if (CALM) { fn(1); if (done) done(); return function () {}; }   // reduced motion: the final state at once
    var t0 = null, id, stopped = false;
    function f(t) { if (stopped) return; if (t0 === null) t0 = t; var u = Math.min(1, (t - t0) / ms); fn(u); if (u < 1) id = requestAnimationFrame(f); else if (done) done(); }
    id = requestAnimationFrame(f);
    var stop = function () { stopped = true; cancelAnimationFrame(id); };
    (box._stops = box._stops || []).push(stop);
    return stop;
  }
  function halt(box) { (box._stops || []).forEach(function (f) { f(); }); box._stops = []; }
  // an element first shown at step `at`: hidden before it, faded in when the step being played reveals it
  function reveal(box, e, k, at, animate, ms) {
    e.setAttribute('opacity', k >= at ? 1 : 0);
    if (animate && k === at) { e.setAttribute('opacity', 0); tween(box, ms || 500, function (u) { e.setAttribute('opacity', ease(u).toFixed(3)); }); }
    return e;
  }
  function pct(v, k) { return (v * 100).toFixed(k == null ? 0 : k) + '%'; }
  function signed(v, k) { return (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(k == null ? 2 : k); }
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function day(s) { return Date.parse(s) / 864e5; }
  var COL = {ink: '#161616', grey: '#6F6F6F', mist: '#A8A8A6', rule: '#DEDEDE', wash: '#F3F3F1', cobalt: '#2C4BEF', red: '#E0442B'};
  window.DECKUTIL = {el: el, tx: tx, surface: surface, lin: lin, ease: ease, tween: tween, halt: halt, reveal: reveal, pct: pct, signed: signed, day: day, MON: MON, COL: COL};

  // a chart built from a draw(state) function: steps re-draw, still draws the last state
  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); box._k = -1; build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; var from = box._k; box._k = k; build(box, k, true, from); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k, false); }
    };
  }

  // ------------------------------------------------------------------ the bead grid (title and closing)
  C.beads = {
    enter: function (box) { window.BEADS.enter(box); },
    leave: function (box) { window.BEADS.leave(box); },
    still: function (box) { window.BEADS.still(box); }
  };

  // ------------------------------------------------------------------ 1.1 three stations, then attention
  C.stations = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, y = 110, xs = [240, 700, 1160];
    var names = ['230P', '230ZA', '230ZB'];
    var lines = [['Bybee, Kelly & Su pipeline on FNSPID news', 'LDA vs MiniLM topics'],
                 ['Multi-horizon exposures and spherical clustering', 'Paper replication using supplied LDA outputs'],
                 ['Rebuilt under train, validation', 'and test']];
    var line = el('line', {x1: 60, y1: y, x2: 60, y2: y, stroke: COL.mist, 'stroke-width': 2}, s);
    var nodes = [];
    xs.forEach(function (x, i) {
      var gg = el('g', {opacity: 0}, s);
      el('circle', {cx: x, cy: y, r: 13, fill: i === 2 ? COL.cobalt : COL.ink}, gg);
      tx(gg, x, y - 38, names[i], {'text-anchor': 'middle', 'font-size': 40, class: 't-serif', 'font-weight': 500});
      tx(gg, x, y + 58, lines[i][0], {'text-anchor': 'middle', 'font-size': 20, class: 't-quiet'});
      tx(gg, x, y + 88, lines[i][1], {'text-anchor': 'middle', 'font-size': 20, class: 't-quiet'});
      nodes.push(gg);
    });
    function showStations(u) {
      var x = 60 + (1320 - 60) * u;
      line.setAttribute('x2', x);
      nodes.forEach(function (n, i) { n.setAttribute('opacity', x >= xs[i] ? 1 : 0); });
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 1400, showStations); else showStations(1); }

    // attention: every topic's share of each month's news, stacked; on the highlighted topic, the months that are beads on the cover
    var M = D().map, months = M.months, ti = 0, top = 300, bot = 600, left = 60, right = 1320;
    var gA = reveal(box, el('g', {}, s), k, 2, animate), X = lin(0, months.length - 1, left, right), Y = lin(0, 1, bot, top);
    var order = [ti].concat(M.shares.map(function (_, i) { return i; }).filter(function (i) { return i !== ti; }));
    var base = months.map(function () { return 0; }), topEdge = null;
    var defs = el('defs', {}, s), clip = el('clipPath', {id: 'att-clip'}, defs), cr = el('rect', {x: left, y: top - 20, width: 0, height: bot - top + 40}, clip);
    [0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', {x1: left, y1: Y(v), x2: right, y2: Y(v), class: 'ln-rule'}, gA);
      tx(gA, left - 12, Y(v) + 7, pct(v), {'text-anchor': 'end', 'font-size': 20, class: 't-mono t-quiet'});
    });
    var gS = el('g', {'clip-path': 'url(#att-clip)'}, gA);
    order.forEach(function (t, j) {
      var ser = M.shares[t], up = ser.map(function (v, i) { return base[i] + v; }), d = '', i;
      for (i = 0; i < months.length; i++) d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(up[i]).toFixed(1);
      for (i = months.length - 1; i >= 0; i--) d += 'L' + X(i).toFixed(1) + ' ' + Y(base[i]).toFixed(1);
      el('path', {d: d + 'Z', fill: t === ti ? 'rgba(44,75,239,.62)' : (j % 2 ? '#E6E6E3' : '#D3D3D0'), stroke: '#fff', 'stroke-width': 0.7}, gS);
      if (t === ti) topEdge = up;
      base = up;
    });
    var B = M.beads, i0 = months.indexOf(B.months[0]);
    B.rows.forEach(function (row, r) { if (row[ti]) el('circle', {cx: X(i0 + r), cy: Y(topEdge[i0 + r]) - 8, r: 4, fill: COL.cobalt}, gS); });
    el('line', {x1: left, y1: bot, x2: right, y2: bot, class: 'ln-axis', 'stroke-width': 1.5}, gA);
    for (var yy = 2013; yy <= 2024; yy += 2) {
      var ix = months.indexOf(yy + '-01'); if (ix < 0) continue;
      tx(gA, X(ix), bot + 30, String(yy), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    }
    tx(gA, left, top - 34, '20 NoLBERT topics: each one’s share of the month’s news, stacked, 2013–2024', {'font-size': 26});
    tx(gA, left, top - 6, 'blue: “' + M.terms[ti] + '” · dots on it: the months that are beads on the cover · topics fitted on text to 2019', {'font-size': 19, class: 't-mono t-quiet'});
    if (k >= 2) { if (animate) tween(box, 2400, function (u) { cr.setAttribute('width', (right - left) * u); }); else cr.setAttribute('width', right - left); }
  });

  // ------------------------------------------------------------------ 1.2 the delta
  C.delta = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    var rows = [
      ['Short test period', 'Complete-return test: 21 months, still short', null, 'open'],
      ['Look-ahead bias', 'Topics and vocabulary fit on text to 2019', 'still not point-in-time: encoders, news source', 'partly', true],
      ['Model chosen after testing', 'Model selection on 2020–2021 validation', 'Test period revisited after fixes', 'partly'],
      ['Unequal comparison', 'Same articles, one top-word rule', 'text spans and encoder training still differ', 'partly'],
      ['Limited testing', 'Two topic counts for NoLBERT; seed reruns', null, 'partly'],
      ['No time-of-day check', 'News used from the next trading day', null, 'open']];
    var xl = 470, xr = 740, xt = 774, y0 = 64, dy = 86;
    tx(s, xl + 13, 22, '230ZA limitation', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    tx(s, xt, 22, '230ZB changes', {'font-size': 18, class: 't-mono t-quiet'});
    rows.forEach(function (r, i) {
      var y = y0 + i * dy, shown = k >= Math.floor(i / 2) + 1, open = r[3] === 'open';
      el('circle', {cx: xl, cy: y, r: 10, fill: COL.ink}, s);
      tx(s, xl - 26, y + 9, r[0], {'text-anchor': 'end', 'font-size': 27});
      var gl = reveal(box, el('g', {}, s), k, Math.floor(i / 2) + 1, animate);
      var ln = el('line', {x1: xl + 12, y1: y, x2: xl + 12, y2: y, stroke: open ? COL.mist : COL.cobalt, 'stroke-width': open ? 2 : 3, 'stroke-dasharray': open ? '7 7' : null}, gl);
      el('circle', {cx: xr, cy: y, r: 10, fill: COL.paper || '#fff', stroke: open ? COL.mist : COL.cobalt, 'stroke-width': 3}, gl);
      tx(gl, xt, y + 9, r[1], {'font-size': 25});
      if (r[2]) tx(gl, xt, y + 40, r[2], {'font-size': 19, class: 't-mono ' + (r[4] ? 't-red' : 't-quiet')});
      var grow = function (u) { ln.setAttribute('x2', xl + 12 + (xr - 12 - xl - 12) * u); };
      if (shown) { if (animate && Math.floor(i / 2) + 1 === k) tween(box, 900, function (u) { grow(ease(u)); }); else grow(1); }
    });
    var ly = y0 + 6 * dy + 6;
    el('circle', {cx: 16, cy: ly, r: 9, fill: '#fff', stroke: COL.cobalt, 'stroke-width': 3}, s);
    tx(s, 36, ly + 7, 'Partly addressed', {'font-size': 20, class: 't-quiet'});
    el('circle', {cx: 236, cy: ly, r: 9, fill: '#fff', stroke: COL.mist, 'stroke-width': 3}, s);
    tx(s, 256, ly + 7, 'Unresolved', {'font-size': 20, class: 't-quiet'});
    tx(s, 440, ly + 7, 'red text: look-ahead that remains', {'font-size': 20, class: 't-red'});
  });

  // ------------------------------------------------------------------ B12 sealed in 1995 (the former 2.2; the 'saw' variant was the former 3.2)
  function timeline(box, k, animate, focus) {
    var g = surface(box), s = g.s, F = D().facts, nb = F.nolbert, sp = F.splits, en = F.encoders, ad = F.adaptation;
    var saw = focus === 'saw';
    var defs = el('defs', {}, s);
    var p1 = el('pattern', {id: 'hg' + focus, width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'}, defs);
    el('line', {x1: 0, y1: 0, x2: 0, y2: 7, stroke: COL.mist, 'stroke-width': 2.4}, p1);
    var p2 = el('pattern', {id: 'hr' + focus, width: 8, height: 8, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'}, defs);
    el('line', {x1: 0, y1: 0, x2: 0, y2: 8, stroke: COL.red, 'stroke-width': 3}, p2);
    var L = 390, R = 1360;
    var X1 = function (d) { return L + (day(d) - day('1975-01-01')) / (day('2025-01-01') - day('1975-01-01')) * (R - L); };
    var X2 = function (d) { return L + (day(d) - day('2012-01-01')) / (day('2025-01-01') - day('2012-01-01')) * (R - L); };
    var yA = 30, yB = saw ? 40 : 330;

    if (!saw) {
      [1980, 1990, 2000, 2010, 2020].forEach(function (y) {
        el('line', {x1: X1(y + '-01-01'), x2: X1(y + '-01-01'), y1: yA + 20, y2: yA + 190, class: 'ln-rule'}, s);
        tx(s, X1(y + '-01-01'), yA + 10, String(y), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
      });
      tx(s, 0, yA + 70, 'NoLBERT pretraining', {'font-size': 25, 'font-weight': 600});
      tx(s, 0, yA + 141, 'Our news, 1.26M articles', {'font-size': 25});
      var gN = reveal(box, el('g', {}, s), k, 1, animate);
      var band = el('rect', {x: X1(nb.window[0] + '-01-01'), y: yA + 46, width: 0, height: 34, rx: 4, fill: 'rgba(44,75,239,.14)', stroke: COL.cobalt, 'stroke-width': 2.5}, gN);
      var lab = tx(gN, (X1(nb.window[0] + '-01-01') + X1('1996-01-01')) / 2, yA + 71, nb.window[0] + '–' + nb.window[1], {'text-anchor': 'middle', 'font-size': 22, 'font-weight': 600, opacity: 0});
      var gV = el('g', {opacity: 0}, gN);
      el('rect', {x: X1('1996-01-01'), y: yA + 46, width: X1('1997-01-01') - X1('1996-01-01'), height: 34, fill: 'rgba(44,75,239,.55)'}, gV);
      tx(gV, X1('1997-01-01') + 10, yA + 71, 'validation ' + nb.val, {'font-size': 20, class: 't-mono t-quiet'});
      el('line', {x1: X1('1996-01-01'), x2: X1('1996-01-01'), y1: yA + 30, y2: yA + 190, stroke: COL.cobalt, 'stroke-width': 2.5, 'stroke-dasharray': '7 6'}, gV);
      var growN = function (u) { band.setAttribute('width', (X1('1996-01-01') - X1(nb.window[0] + '-01-01')) * u); lab.setAttribute('opacity', u > .9 ? 1 : 0); gV.setAttribute('opacity', u >= 1 ? 1 : 0); };
      if (k >= 1) { if (animate && k === 1) tween(box, 1500, growN); else growN(1); }

      var gP = reveal(box, el('g', {}, s), k, 2, animate);
      el('rect', {x: X1('2012-01-01'), y: yA + 118, width: X1('2025-01-01') - X1('2012-01-01'), height: 30, rx: 4, fill: '#DADAD7'}, gP);
      tx(gP, X1('2012-01-01') - 12, yA + 140, '2012–2024', {'text-anchor': 'end', 'font-size': 20, class: 't-mono t-quiet'});
      var gapY = yA + 182, b0 = X1('1996-01-01'), b1 = X1(sp.test[0]);
      var br = el('line', {x1: b0, y1: gapY, x2: b0, y2: gapY, stroke: COL.ink, 'stroke-width': 2}, gP);
      el('line', {x1: b0, y1: gapY - 9, x2: b0, y2: gapY + 9, stroke: COL.ink, 'stroke-width': 2}, gP);
      var tick2 = el('line', {x1: b1, y1: gapY - 9, x2: b1, y2: gapY + 9, stroke: COL.ink, 'stroke-width': 2, opacity: 0}, gP);
      var yrs = (+sp.test[0].slice(0, 4)) - (nb.window[1] + 1);
      var gl = tx(gP, (b0 + b1) / 2, gapY + 40, '', {'text-anchor': 'middle', 'font-size': 24, 'font-weight': 600});
      var growG = function (u) { br.setAttribute('x2', b0 + (b1 - b0) * u); gl.textContent = Math.round(yrs * u) + ' years'; tick2.setAttribute('opacity', u >= 1 ? 1 : 0); };
      if (k >= 2) { if (animate && k === 2) tween(box, 1600, growG); else growG(1); }
    }

    // the zoom: 2012 to 2025
    var zoomStep = saw ? 0 : 3, gZ = reveal(box, el('g', {}, s), k, zoomStep, animate, 700);
    var top = yB, rowP = top + 70, rowA = top + 150, rowM = top + (saw ? 250 : 206), rowF = top + (saw ? 302 : 258);
    [2012, 2014, 2016, 2018, 2020, 2022, 2024].forEach(function (y) {
      el('line', {x1: X2(y + '-01-01'), x2: X2(y + '-01-01'), y1: top + 22, y2: rowF + 30, class: 'ln-rule'}, gZ);
      tx(gZ, X2(y + '-01-01'), top + 12, String(y), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    });
    tx(gZ, 0, rowP + 24, 'Our protocol', {'font-size': 25});
    var parts = [['train', sp.train, 0.22], ['validation', sp.valid, 0.4], ['test', sp.test, 0.62]];
    parts.forEach(function (p) {
      el('rect', {x: X2(p[1][0]), y: rowP, width: X2(p[1][1]) - X2(p[1][0]), height: 32, fill: 'rgba(22,22,22,' + p[2] * 0.45 + ')'}, gZ);
      tx(gZ, (X2(p[1][0]) + X2(p[1][1])) / 2, rowP - 10, p[0] === 'test' ? 'test: ' + sp.test_months + ' months used' : p[0], {'text-anchor': 'middle', 'font-size': 20, class: 't-quiet'});
    });
    for (var yy = sp.unused_from; yy <= 2024; yy++) {
      el('rect', {x: X2(yy + '-02-01'), y: rowP, width: X2(yy + '-07-01') - X2(yy + '-02-01'), height: 32, fill: 'url(#hg' + focus + ')'}, gZ);
    }
    tx(gZ, 0, rowP + 56, 'hatched: Feb–Jun unused from ' + sp.unused_from, {'font-size': 17, class: 't-mono t-quiet'});

    var gA = el('g', {opacity: saw ? 1 : 0}, gZ);
    tx(gA, 0, rowA + 22, 'Our NoLBERT adaptation', {'font-size': 25, 'font-weight': saw ? 600 : 400});
    el('rect', {x: X2(ad.corpus_from), y: rowA, width: X2(ad.before) - X2(ad.corpus_from), height: 28, fill: 'url(#hr' + focus + ')', stroke: COL.red, 'stroke-width': 2}, gA);
    if (saw) {
      el('rect', {x: X2(sp.valid[0]), y: rowA - 6, width: X2(ad.before) - X2(sp.valid[0]), height: 40, fill: 'none', stroke: COL.red, 'stroke-width': 3}, gA);
      tx(gA, X2(ad.corpus_from), rowA + 64, 'Jan 2015–Aug 2020 · 8-month validation overlap', {'font-size': 21, class: 't-red'});
      el('line', {x1: X2('2019-12-31'), x2: X2('2019-12-31'), y1: rowA - 16, y2: rowA + 40, stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '5 4'}, gA);
      tx(gA, X2('2019-12-31') - 10, rowA - 20, 'Refit using text through Dec 2019', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-cobalt'});
    } else tx(gA, X2(ad.before) + 12, rowA + 21, 'Jan 2015–Aug 2020', {'font-size': 20, class: 't-red'});

    var gE = el('g', {opacity: saw ? 0.35 : 0}, gZ);
    tx(gE, 0, rowM + 22, 'MiniLM (paraphrase-MiniLM-L3-v2)', {'font-size': 23});
    el('rect', {x: X2(en.minilm + '-01-01'), y: rowM, width: X2((+en.minilm + 1) + '-01-01') - X2(en.minilm + '-01-01'), height: 26, fill: 'rgba(224,68,43,.18)', stroke: COL.red, 'stroke-width': 2}, gE);
    tx(gE, X2(en.minilm + '-01-01') - 12, rowM + 20, 'published ' + en.minilm, {'text-anchor': 'end', 'font-size': 20, class: 't-red'});
    tx(gE, 0, rowF + 22, 'FinBERT topic classifier', {'font-size': 23});
    var xT = X2(en.finbert_tweets + '-15'), xP = X2(en.finbert_published + '-15');
    el('path', {d: 'M' + xT + ' ' + (rowF + 2) + 'l11 11l-11 11l-11 -11z', fill: COL.red}, gE);
    el('path', {d: 'M' + xP + ' ' + (rowF + 2) + 'l11 11l-11 11l-11 -11z', fill: COL.red}, gE);
    tx(gE, xT - 18, rowF + 20, 'tweets released ' + MON[+en.finbert_tweets.slice(5) - 1] + ' ' + en.finbert_tweets.slice(0, 4), {'text-anchor': 'end', 'font-size': 20, class: 't-red'});
    tx(gE, xP - 18, rowF + 52, 'published ' + MON[+en.finbert_published.slice(5) - 1] + ' ' + en.finbert_published.slice(0, 4), {'text-anchor': 'end', 'font-size': 20, class: 't-red'});
    if (!saw && k >= 3) {
      if (animate && k === 3) tween(box, 900, function (u) { gE.setAttribute('opacity', ease(u)); gA.setAttribute('opacity', u > .5 ? ease(u * 2 - 1) : 0); });
      else { gE.setAttribute('opacity', 1); gA.setAttribute('opacity', 1); }
    }
  }
  C.sealed = stepped(function (box, k, animate) { timeline(box, k, animate, 'sealed'); });

  // ------------------------------------------------------------------ B8 the authors' historical usage tests (Table 2), on a signed t axis
  C.table2 = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, nb = D().facts.nolbert, X = lin(-6, 6, 420, 1380);
    var rows = [['Look-ahead test', '1976–1995 vs 2020–present', nb.t_ahead],
                ['Look-back test', '19th century vs 1976–1995', nb.t_back]];
    [-6, -4, -2, 0, 2, 4, 6].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 40, y2: 400, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, s);
      tx(s, X(v), 432, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    });
    tx(s, X(6), 468, 'Paired t statistic', {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
    rows.forEach(function (r, i) {
      var y = 110 + i * 170, t = r[2], neg = t < 0;
      tx(s, 0, y + 10, r[0], {'font-size': 30, 'font-weight': 500});
      tx(s, 0, y + 44, r[1], {'font-size': 19, class: 't-mono t-quiet'});
      var bar = el('rect', {x: X(0), y: y - 22, width: 0, height: 52, fill: COL.cobalt}, s);
      var lab = tx(s, X(0), y + 10, '', {'font-size': 24, class: 't-mono', 'font-weight': 500, 'text-anchor': neg ? 'end' : 'start'});
      var grow = function (u) {
        var w = Math.abs(X(t) - X(0)) * u;
        bar.setAttribute('x', neg ? X(0) - w : X(0)); bar.setAttribute('width', w);
        lab.setAttribute('x', neg ? X(0) - w - 14 : X(0) + w + 14);
        lab.textContent = u >= 1 ? 't = ' + (neg ? '−' : '+') + Math.abs(t).toFixed(2) : '';
      };
      if (k >= 1) { if (animate && k === 1) tween(box, 1300, grow); else grow(1); }
    });
    reveal(box, tx(s, 0, 530, 'Higher mean log probability for 1976–1995 usage.', {'font-size': 24}), k, 1, animate, 900);
    tx(s, 0, 580, 'Original model · Authors’ Table 2 · 20 words per test', {'font-size': 19, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ B9 topic stability across seeds (the former 3.3)
  C.stability = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, rows = D().stability.rows, X = lin(0, 1, 300, 1000);
    tx(s, 0, 30, '60 topics · Two random seeds', {'font-size': 24, 'font-weight': 600});
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 60, y2: 420, class: v ? 'ln-rule' : 'ln-axis'}, s);
      tx(s, X(v), 452, v.toFixed(2), {'text-anchor': 'middle', 'font-size': 19, class: 't-mono t-quiet'});
    });
    tx(s, X(1), 486, 'Adjusted Rand index · 1 = identical assignments · 0 = chance agreement', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    rows.forEach(function (r, i) {
      var y = 84 + i * 112, bar = el('rect', {x: X(0), y: y, width: 0, height: 60, fill: i === 0 ? COL.cobalt : '#BDBDBA'}, s);
      tx(s, 0, y + 40, r.model, {'font-size': 30, 'font-weight': i === 0 ? 600 : 400});
      var lab = tx(s, X(0), y + 40, '', {'font-size': 24, class: 't-mono'});
      var grow = function (u) { var w = (X(r.ari) - X(0)) * u; bar.setAttribute('width', w); lab.setAttribute('x', X(0) + w + 14); lab.textContent = u >= 1 ? r.ari.toFixed(2) : ''; };
      if (k >= 1) { if (animate && k === 1) tween(box, 1100, grow); else grow(1); }
    });
  });

})();
