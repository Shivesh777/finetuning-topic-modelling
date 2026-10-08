/* Exhibits, part A: helpers, the news map hook, sections 1 to 3.
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
  function tween(box, ms, fn, done) {
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
                 ['Conditional replication with the', "authors' LDA outputs"],
                 ['Rebuilt under train, validation', 'and test']];
    var line = el('line', {x1: 60, y1: y, x2: 60, y2: y, stroke: COL.mist, 'stroke-width': 2}, s);
    var nodes = [];
    xs.forEach(function (x, i) {
      var gg = el('g', {opacity: 0}, s);
      el('circle', {cx: x, cy: y, r: 13, fill: i === 2 ? COL.cobalt : COL.ink}, gg);
      tx(gg, x, y - 38, names[i], {'text-anchor': 'middle', 'font-size': 40, class: 't-serif', 'font-weight': 500});
      tx(gg, x, y + 58, lines[i][0], {'text-anchor': 'middle', 'font-size': 24, class: 't-quiet'});
      tx(gg, x, y + 90, lines[i][1], {'text-anchor': 'middle', 'font-size': 24, class: 't-quiet'});
      nodes.push(gg);
    });
    function showStations(u) {
      var x = 60 + (1320 - 60) * u;
      line.setAttribute('x2', x);
      nodes.forEach(function (n, i) { n.setAttribute('opacity', x >= xs[i] ? 1 : 0); });
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 1400, showStations); else showStations(1); }

    // attention: one topic's monthly share of the news
    var M = D().map, ti = 0, ser = M.shares[ti], months = M.months;
    var gA = reveal(box, el('g', {}, s), k, 2, animate), top = 300, bot = 600, left = 60, right = 1320;
    var X = lin(0, ser.length - 1, left, right), mx = Math.max.apply(null, ser) * 1.1, Y = lin(0, mx, bot, top);
    el('line', {x1: left, y1: bot, x2: right, y2: bot, class: 'ln-axis', 'stroke-width': 1.5}, gA);
    for (var yy = 2013; yy <= 2024; yy += 2) {
      var i0 = months.indexOf(yy + '-01'); if (i0 < 0) continue;
      tx(gA, X(i0), bot + 30, String(yy), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    }
    [0.02, 0.04, 0.06, 0.08].forEach(function (v) {
      if (v > mx) return;
      el('line', {x1: left, y1: Y(v), x2: right, y2: Y(v), class: 'ln-rule'}, gA);
      tx(gA, left - 12, Y(v) + 7, pct(v), {'text-anchor': 'end', 'font-size': 20, class: 't-mono t-quiet'});
    });
    var path = el('path', {fill: 'none', stroke: COL.cobalt, 'stroke-width': 3}, gA);
    function drawPath(u) {
      var n = Math.max(2, Math.round(ser.length * u)), d = '';
      for (var i = 0; i < n; i++) d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(ser[i]).toFixed(1);
      path.setAttribute('d', d);
    }
    tx(gA, left, top - 34, 'One NoLBERT topic: “' + M.terms[ti] + '”', {'font-size': 26});
    tx(gA, left, top - 6, 'its share of each month’s news, 2013–2024 · 20-topic model, topics fitted on text to 2019', {'font-size': 19, class: 't-mono t-quiet'});
    if (k >= 2) { if (animate) tween(box, 2400, function (u) { drawPath(u); }); else drawPath(1); }
  });

  // ------------------------------------------------------------------ 1.2 the delta
  C.delta = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    var rows = [
      ['Short test period', 'Complete-return test: 21 months, still short', null, 'open'],
      ['Look-ahead bias', 'Topics and vocabulary fit on text to 2019', 'still not point-in-time: encoders, news source', 'partly', true],
      ['Model chosen after testing', 'Validation 2020–2021 picks the headline', 'test period revisited after documented fixes', 'partly'],
      ['Unequal comparison', 'Same articles, one top-word rule', 'text spans and encoder training still differ', 'partly'],
      ['Limited testing', 'Two topic counts for NoLBERT; seed reruns', null, 'partly'],
      ['No time-of-day check', 'News used from the next trading day', null, 'open']];
    var xl = 470, xr = 740, xt = 774, y0 = 64, dy = 86;
    tx(s, xl + 13, 22, '230ZA LIMITATION', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    tx(s, xt, 22, '230ZB ANSWER', {'font-size': 18, class: 't-mono t-quiet'});
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
    tx(s, 36, ly + 7, 'partly answered', {'font-size': 20, class: 't-quiet'});
    el('circle', {cx: 236, cy: ly, r: 9, fill: '#fff', stroke: COL.mist, 'stroke-width': 3}, s);
    tx(s, 256, ly + 7, 'open', {'font-size': 20, class: 't-quiet'});
    tx(s, 340, ly + 7, 'red text: look-ahead that remains', {'font-size': 20, class: 't-red'});
  });

  // ------------------------------------------------------------------ 2.1 two biases (the paper's examples)
  C.biases = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    var heads = ['Look-ahead', 'Look-back'];
    var quotes = [['“models trained on future information', 'contaminate inferences”'],
                  ['“representations that conflate different', 'historical contexts”']];
    [0, 720].forEach(function (x, i) {
      tx(s, x, 40, heads[i], {'font-size': 46, class: 't-serif', 'font-weight': 500});
      tx(s, x, 86, quotes[i][0], {'font-size': 21, class: 't-mono t-quiet'});
      tx(s, x, 113, quotes[i][1], {'font-size': 21, class: 't-mono t-quiet'});
    });
    var gL = reveal(box, el('g', {}, s), k, 1, animate, 700);
    tx(gL, 0, 196, 'He offered a ____ of gratitude in the form of flowers.', {'font-size': 25});
    tx(gL, 0, 232, '“token”, old era (1976–1995)', {'font-size': 20, class: 't-mono t-cobalt'});
    tx(gL, 0, 306, 'She invested in a governance ____ for the new DAO.', {'font-size': 25});
    tx(gL, 0, 342, '“token”, new era (2020–present)', {'font-size': 20, class: 't-mono t-red'});
    tx(gL, 0, 418, 'A model trained on later text knows the second sense.', {'font-size': 23, class: 't-quiet'});
    var gR = reveal(box, el('g', {}, s), k, 2, animate, 700);
    tx(gR, 720, 196, '“running a program”', {'font-size': 30, class: 't-serif', 'font-style': 'italic'});
    tx(gR, 720, 256, 'early 1900s', {'font-size': 20, class: 't-mono t-quiet'});
    tx(gR, 720, 288, 'organizing an event', {'font-size': 25});
    tx(gR, 720, 348, 'late 20th century', {'font-size': 20, class: 't-mono t-quiet'});
    tx(gR, 720, 380, 'executing computer code', {'font-size': 25});
    tx(gR, 720, 446, 'A model trained across eras blends both meanings.', {'font-size': 23, class: 't-quiet'});
  });

  // ------------------------------------------------------------------ 2.2 sealed in 1995 (also 3.2, with the adaptation in front)
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
      tx(gN, 0, yA + 262, 'NoLBERT: ' + nb.arch + ' · ' + nb.params + ' parameters · masked language modeling, ' + nb.epochs + ' epochs · its own ' + nb.vocab + '-token tokenizer', {'font-size': 17, class: 't-mono t-quiet'});
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
      var growG = function (u) { br.setAttribute('x2', b0 + (b1 - b0) * u); gl.textContent = Math.round(yrs * u) + ' years from the end of pretraining to our test'; tick2.setAttribute('opacity', u >= 1 ? 1 : 0); };
      if (k >= 2) { if (animate && k === 2) tween(box, 1600, growG); else growG(1); }
    }

    // the zoom: 2012 to 2025
    var zoomStep = saw ? 0 : 3, gZ = reveal(box, el('g', {}, s), k, zoomStep, animate, 700);
    var top = yB, rowP = top + 70, rowA = top + 150, rowM = top + (saw ? 250 : 206), rowF = top + (saw ? 302 : 258);
    [2012, 2014, 2016, 2018, 2020, 2022, 2024].forEach(function (y) {
      el('line', {x1: X2(y + '-01-01'), x2: X2(y + '-01-01'), y1: top + 22, y2: rowF + 30, class: 'ln-rule'}, gZ);
      tx(gZ, X2(y + '-01-01'), top + 12, String(y), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    });
    if (!saw) tx(gZ, 0, top + 12, '2012 to 2024, enlarged', {'font-size': 20, class: 't-mono t-quiet'});
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
      tx(gA, X2(ad.corpus_from), rowA + 64, 'news before 1 Sep 2020: 8 months inside validation, none in test', {'font-size': 21, class: 't-red'});
    } else tx(gA, X2(ad.before) + 12, rowA + 21, 'before Sep 2020 (section 3)', {'font-size': 20, class: 't-red'});

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
  C.saw = stepped(function (box, k, animate) { timeline(box, k, animate, 'saw'); });

  // ------------------------------------------------------------------ 2.3 the authors' tests (Table 2)
  C.table2 = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, nb = D().facts.nolbert, X = lin(0, 6, 400, 1100);
    var rows = [['Look-ahead test', nb.t_ahead, nb.d_ahead, '1976–1995 vs 2020–present'],
                ['Look-back test', nb.t_back, nb.d_back, '19th century vs 1976–1995']];
    [0, 2, 4, 6].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 50, y2: 440, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, s);
      tx(s, X(v), 472, String(v), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono t-quiet'});
    });
    tx(s, X(6), 506, '|t|, one-sided paired test', {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
    rows.forEach(function (r, i) {
      var y = 100 + i * 190;
      tx(s, 0, y + 16, r[0], {'font-size': 30, 'font-weight': 500});
      tx(s, 0, y + 50, r[3], {'font-size': 19, class: 't-mono t-quiet'});
      var bar = el('rect', {x: X(0), y: y - 14, width: 0, height: 44, fill: COL.cobalt}, s);
      var lab = tx(s, X(0), y + 16, '', {'font-size': 24, class: 't-mono', 'font-weight': 500});
      var grow = function (u) {
        var w = (X(Math.abs(r[1])) - X(0)) * u; bar.setAttribute('width', w);
        lab.setAttribute('x', X(0) + w + 14); lab.textContent = u >= 1 ? 't = ' + (r[1] < 0 ? '−' : '') + Math.abs(r[1]).toFixed(2) : '';
      };
      if (k >= 1) { if (animate && k === 1) tween(box, 1300, grow); else grow(1); }
      reveal(box, tx(s, X(0) + 10, y + 64, 'favors the 1976–1995 sense · log-prob gap, old − new: ' + signed(r[2]) + ' · p < 0.01', {'font-size': 20, class: 't-mono t-cobalt',
        stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'}), k, 1, animate, 900);
    });
    tx(s, 0, 566, nb.words + ' words per test · sentence pairs generated with GPT-5 · H0: equal mean log-probability in both eras', {'font-size': 19, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 3.1 the training replay
  C.replay = {
    draw: function (box, k, play) {
      halt(box);
      var g = surface(box), s = g.s, T = D().training, log = T.log, B = T.batch;
      // the task, left
      tx(s, 0, 30, 'The task, in every batch', {'font-size': 24, 'font-weight': 600});
      el('rect', {x: 0, y: 56, width: 170, height: 70, rx: 6, fill: 'rgba(44,75,239,.12)', stroke: COL.cobalt, 'stroke-width': 2}, s);
      tx(s, 85, 98, 'one chunk', {'text-anchor': 'middle', 'font-size': 21});
      tx(s, 200, 98, '→ which of these 32 is from', {'font-size': 21, class: 't-quiet'});
      tx(s, 200, 124, '    the same article?', {'font-size': 21, class: 't-quiet'});
      for (var i = 0; i < B; i++) {
        var cx = (i % 8) * 54, cy = 160 + Math.floor(i / 8) * 54;
        el('rect', {x: cx, y: cy, width: 44, height: 44, rx: 5, fill: i === 13 ? 'rgba(44,75,239,.12)' : COL.wash, stroke: i === 13 ? COL.cobalt : COL.rule, 'stroke-width': i === 13 ? 2.5 : 1.5}, s);
      }
      tx(s, 0, 408, 'Chance: 1 in ' + B + '. InfoNCE: two random', {'font-size': 19, class: 't-mono t-quiet'});
      tx(s, 0, 434, 'chunks of one article are a pair; the', {'font-size': 19, class: 't-mono t-quiet'});
      tx(s, 0, 460, 'other ' + (B - 1) + ' in the batch are negatives.', {'font-size': 19, class: 't-mono t-quiet'});
      // the strip: how many of 32 were right in the batch being replayed
      tx(s, 0, 518, 'Right in this logged batch', {'font-size': 21});
      var cells = [];
      for (var j = 0; j < B; j++) cells.push(el('rect', {x: j * 13.5, y: 534, width: 11, height: 34, fill: COL.wash, stroke: COL.rule}, s));
      var count = tx(s, 0, 606, '', {'font-size': 22, class: 't-mono'});
      // the curve, right
      var L = 620, R = 1360, top = 40, bot = 520, X = lin(0, T.steps, L, R), Y = lin(0, 1, bot, top);
      [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
        el('line', {x1: L, x2: R, y1: Y(v), y2: Y(v), class: 'ln-rule'}, s);
        tx(s, L - 12, Y(v) + 7, pct(v), {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
      });
      [0, 2000, 4000, 6000].forEach(function (v) { tx(s, X(v), bot + 30, v.toLocaleString('en-US'), {'text-anchor': 'middle', 'font-size': 19, class: 't-mono t-quiet'}); });
      tx(s, R, bot + 60, 'training step', {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
      el('line', {x1: L, x2: R, y1: Y(1 / B), y2: Y(1 / B), stroke: COL.mist, 'stroke-dasharray': '6 6', 'stroke-width': 1.5}, s);
      tx(s, R, Y(1 / B) - 10, 'chance', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
      var dots = el('g', {}, s), path = el('path', {fill: 'none', stroke: COL.cobalt, 'stroke-width': 3}, s);
      var dia = function (step, v, lab, right) {
        var gg = el('g', {}, s);
        el('path', {d: 'M' + X(step) + ' ' + (Y(v) - 12) + 'l12 12l-12 12l-12 -12z', fill: COL.cobalt}, gg);
        tx(gg, right ? X(step) - 18 : X(step) + 20, right ? Y(v) - 22 : Y(v) - 16, lab, {'text-anchor': right ? 'end' : 'start', 'font-size': 22, 'font-weight': 600,
          stroke: '#fff', 'stroke-width': 8, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
        return gg;
      };
      dia(0, T.before, 'held-out, before: ' + pct(T.before), false);
      var after = dia(T.steps, T.after, 'held-out, after: ' + pct(T.after), true); after.setAttribute('opacity', 0);
      function upto(n) {
        while (dots.childNodes.length < n) {
          var r = log[dots.childNodes.length];
          el('circle', {cx: X(r[0]), cy: Y(r[2]), r: 2.6, fill: COL.mist}, dots);
        }
        var d = '';
        for (var i = 0; i < n; i++) {
          var a = Math.max(0, i - 9), m = 0;
          for (var q = a; q <= i; q++) m += log[q][2];
          m /= (i - a + 1);
          d += (i ? 'L' : 'M') + X(log[i][0]).toFixed(1) + ' ' + Y(m).toFixed(1);
        }
        path.setAttribute('d', d);
        var last = log[Math.max(0, n - 1)], right = Math.round(last[2] * B);
        cells.forEach(function (c, i) { c.setAttribute('fill', i < right ? COL.cobalt : COL.wash); });
        count.textContent = 'step ' + last[0].toLocaleString('en-US') + ': ' + right + ' of ' + B;
        after.setAttribute('opacity', n >= log.length ? 1 : 0);
      }
      if (k >= 1 && play) tween(box, 22000, function (u) { upto(Math.max(1, Math.round(log.length * u))); });
      else if (k >= 1) upto(log.length);
      else { cells.forEach(function (c) { c.setAttribute('fill', COL.wash); }); count.textContent = log.length + ' logged batches'; }
    },
    enter: function (box, k) { this.draw(box, k, false); box._k = k; },
    step: function (box, k) { if (box._k === k) return; box._k = k; this.draw(box, k, true); },
    leave: function (box) { halt(box); },
    still: function (box) { this.draw(box, 1, false); }
  };

  // ------------------------------------------------------------------ 3.3 topic stability across seeds
  C.stability = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, rows = D().stability.rows, X = lin(0, 1, 300, 1000);
    tx(s, 0, 30, 'Adjusted Rand index of the 60-topic split, two seeds', {'font-size': 24, 'font-weight': 600});
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 60, y2: 420, class: v ? 'ln-rule' : 'ln-axis'}, s);
      tx(s, X(v), 452, v.toFixed(2), {'text-anchor': 'middle', 'font-size': 19, class: 't-mono t-quiet'});
    });
    tx(s, X(1), 486, '1 = both seeds give the same split · 0 = chance agreement', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    rows.forEach(function (r, i) {
      var y = 84 + i * 112, bar = el('rect', {x: X(0), y: y, width: 0, height: 60, fill: i === 0 ? COL.cobalt : '#BDBDBA'}, s);
      tx(s, 0, y + 40, r.model, {'font-size': 30, 'font-weight': i === 0 ? 600 : 400});
      var lab = tx(s, X(0), y + 40, '', {'font-size': 24, class: 't-mono'});
      var grow = function (u) { var w = (X(r.ari) - X(0)) * u; bar.setAttribute('width', w); lab.setAttribute('x', X(0) + w + 14); lab.textContent = u >= 1 ? r.ari.toFixed(2) + ' · ' + pct(r.kept) + ' of articles keep their topic' : ''; };
      if (k >= 1) { if (animate && k === 1) tween(box, 1100, grow); else grow(1); }
    });
  });

})();
