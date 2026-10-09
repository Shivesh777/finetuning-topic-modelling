/* Exhibits, part E: slides 1.2, 1.3, 5.1, 5.2, 5.3 and 5.4 (the 9 October briefs for sections 1 and 5).
   Every number is read from window.DATA (tools/build_data.py: facts.replication, facts.mvo, facts.za, vt, weekly,
   xasset, textmodels, facts.splits); the labels are the slide copy agreed with the team, used verbatim.
   Each chart has enter(box, step), step(box, k), leave(box) and still(box): still draws the last frame. */
(function () {
  'use strict';
  var U = window.DECKUTIL, el = U.el, tx = U.tx, surface = U.surface, lin = U.lin, ease = U.ease, tween = U.tween,
      halt = U.halt, reveal = U.reveal, COL = U.COL, signed = U.signed, MON = U.MON;
  var C = window.CHARTS = window.CHARTS || {};
  var D = function () { return window.DATA; };
  var SOFT = 'rgba(44,75,239,.12)', SOFT2 = 'rgba(44,75,239,.28)', CHAR = '#232323', GREY = '#B8B8B5', LIGHT = '#D4D4D1', NS = 'http://www.w3.org/2000/svg';

  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); box._k = -1; build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; var from = box._k; box._k = k; build(box, k, true, from); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k, false); }
    };
  }
  function part(u, a, b) { return Math.max(0, Math.min(1, (u - a) / (b - a))); }
  function mix(a, b, v) { return a + (b - a) * v; }
  function rrect(p, x, y, w, h, a) { return el('rect', Object.assign({x: x, y: y, width: w, height: h, rx: 6}, a || {}), p); }
  function label(p, x, y, s, size, cls, a) { return tx(p, x, y, s, Object.assign({'font-size': size || 20, class: cls || ''}, a || {})); }
  function arrow(p, x1, y1, x2, y2, a) {
    a = a || {};
    var g = el('g', {}, p), ang = Math.atan2(y2 - y1, x2 - x1), h = a.head || 9, col = a.stroke || COL.mist;
    if (a.width !== 0) el('line', {x1: x1, y1: y1, x2: x2 - 2 * Math.cos(ang), y2: y2 - 2 * Math.sin(ang), stroke: col, 'stroke-width': a.width || 1.8, 'stroke-dasharray': a.dash || null}, g);
    el('path', {d: 'M' + x2 + ' ' + y2 + 'L' + (x2 - h * Math.cos(ang - 0.42)) + ' ' + (y2 - h * Math.sin(ang - 0.42)) +
      'L' + (x2 - h * Math.cos(ang + 0.42)) + ' ' + (y2 - h * Math.sin(ang + 0.42)) + 'Z', fill: col}, g);
    return g;
  }
  function drawOn(path, u) { var L = path.getTotalLength ? path.getTotalLength() : 1000; path.setAttribute('stroke-dasharray', L + ' ' + L); path.setAttribute('stroke-dashoffset', L * (1 - u)); }
  function rich(p, x, y, parts, a) {   // text with sub- and superscripts: parts = [[text], [text, 'sub' | 'sup'], ...]
    var t = tx(p, x, y, '', a);
    parts.forEach(function (q) {
      var sp = document.createElementNS(NS, 'tspan');
      if (q[1]) { sp.setAttribute('font-size', '62%'); sp.setAttribute('dy', q[1] === 'sub' ? '0.3em' : '-0.6em'); }
      sp.textContent = q[0]; t.appendChild(sp);
      if (q[1]) { var back = document.createElementNS(NS, 'tspan'); back.setAttribute('font-size', '62%'); back.setAttribute('dy', q[1] === 'sub' ? '-0.3em' : '0.6em'); back.textContent = '​'; t.appendChild(back); }
    });
    return t;
  }
  function hash(str) { var h = 2166136261; for (var i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); } return (h >>> 0) / 4294967295; }
  function grid(p, x, y, cols, rows, cell, gap, fill) { var g = el('g', {}, p); for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) el('rect', {x: x + c * (cell + gap), y: y + r * (cell + gap), width: cell, height: cell, fill: fill}, g); return g; }
  function monthName(ym) { return MON[+ym.slice(5, 7) - 1] + ' ' + ym.slice(0, 4); }

  // ------------------------------------------------------------------ 1.2 what we built: the 230P comparison, the 230ZA changes, the separate replication
  C.built = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, F = D().facts, za = F.za, rp = F.replication, i, j;
    var STAGES = [['News', 100], ['Topics', 300], ['Stock exposures', 560], ['Sparse IPCA', 860], ['MVE factor portfolio', 1040]];
    STAGES.forEach(function (h) { label(s, h[1], 22, h[0], 15, 't-mono t-quiet'); });
    var TOP = {y: 140}, LOW = {y: 430};
    label(s, 0, TOP.y + 10, '230P', 28, 't-serif', {'font-weight': 500});

    // scene A, step 1: one news input, two text representations, 60 topics each
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 500);
    rrect(g1, 100, TOP.y - 30, 160, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g1, 180, TOP.y + 7, 'FNSPID news', 19, '', {'text-anchor': 'middle'});
    [['LDA', TOP.y - 44], ['MiniLM', TOP.y + 14]].forEach(function (b) {
      rrect(g1, 300, b[1], 70, 30, {fill: '#fff', stroke: CHAR, 'stroke-width': 1.5, rx: 4}); label(g1, 335, b[1] + 21, b[0], 16, '', {'text-anchor': 'middle'});
      arrow(g1, 262, TOP.y, 296, b[1] + 15, {stroke: CHAR}); grid(g1, 385, b[1] - 6, 10, 6, 6, 1, '#C9CBD8');
    });
    label(g1, 385, TOP.y + 66, za.topics + ' topics', 15, 't-mono t-quiet');
    // scene A, step 2: the shared downstream structure
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 500);
    [TOP.y - 29, TOP.y + 29].forEach(function (yy) { arrow(g2, 458, yy, 556, TOP.y, {stroke: CHAR}); });
    rrect(g2, 560, TOP.y - 30, 240, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); grid(g2, 580, TOP.y - 18, 24, 5, 6, 1.5, '#C9CBD8');
    arrow(g2, 802, TOP.y, 856, TOP.y, {stroke: CHAR});
    rrect(g2, 860, TOP.y - 30, 130, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g2, 925, TOP.y + 7, za.factors + ' factors', 19, '', {'text-anchor': 'middle'});
    arrow(g2, 992, TOP.y, 1036, TOP.y, {stroke: CHAR});
    rrect(g2, 1040, TOP.y - 30, 150, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g2, 1115, TOP.y - 3, 'MVE factor', 18, '', {'text-anchor': 'middle'}); label(g2, 1115, TOP.y + 19, 'portfolio', 18, '', {'text-anchor': 'middle'});
    label(g2, 100, 250, 'Same downstream model; different text representations.', 20);

    // scene B, step 3: the 230ZA row, unchanged stages in charcoal, the changed topic processing in cobalt
    var g3 = reveal(box, el('g', {}, s), k, 3, animate, 500);
    label(g3, 0, LOW.y + 10, '230ZA', 28, 't-serif', {'font-weight': 500});
    rrect(g3, 100, LOW.y - 30, 160, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g3, 180, LOW.y + 7, 'FNSPID news', 19, '', {'text-anchor': 'middle'});
    arrow(g3, 262, LOW.y, 296, LOW.y - 22, {stroke: CHAR});
    rrect(g3, 300, LOW.y - 40, 160, 34, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5, rx: 4}); label(g3, 380, LOW.y - 17, 'Spherical clustering', 16, 't-cobalt', {'text-anchor': 'middle', 'font-weight': 500});
    grid(g3, 300, LOW.y + 6, 10, 6, 6, 1, SOFT2);
    label(g3, 380, LOW.y + 66, za.topics + ' topics', 15, 't-mono t-quiet');
    // scene B, step 4: the same 60 topics at three half-lives, 180 exposure inputs, into the same downstream structure
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500), blocks = [];
    za.half_lives.forEach(function (h, q) {
      var bx = 560 + q * 82, bg = el('g', {opacity: 0}, g4);
      rrect(bg, bx, LOW.y - 38, 76, 54, {fill: COL.wash, stroke: COL.cobalt, 'stroke-width': 1.2, rx: 4}); grid(bg, bx + 8, LOW.y - 29, 10, 6, 5, 1, SOFT2);
      label(bg, bx + 38, LOW.y + 36, h + '-day', 12, 't-mono t-cobalt', {'text-anchor': 'middle'}); label(bg, bx + 38, LOW.y + 52, 'half-life', 12, 't-mono t-cobalt', {'text-anchor': 'middle'});
      arrow(bg, 376, LOW.y + 10, bx - 2, LOW.y - 11, {stroke: COL.cobalt, head: 7}); arrow(bg, bx + 76, LOW.y - 11, 856, LOW.y, {stroke: COL.cobalt, head: 7});
      blocks.push(bg);
    });
    var cnt = label(g4, 560, LOW.y + 80, za.topics * za.half_lives.length + ' exposure inputs', 16, 't-mono t-cobalt', {'font-weight': 500, opacity: 0});
    var rel = label(g4, 560, LOW.y + 102, za.topics + ' topics × ' + za.half_lives.length + ' half-lives', 14, 't-mono t-quiet', {opacity: 0});
    rrect(g4, 860, LOW.y - 30, 130, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g4, 925, LOW.y + 7, za.factors + ' factors', 19, '', {'text-anchor': 'middle'});
    arrow(g4, 992, LOW.y, 1036, LOW.y, {stroke: CHAR});
    rrect(g4, 1040, LOW.y - 30, 150, 60, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g4, 1115, LOW.y - 3, 'MVE factor', 18, '', {'text-anchor': 'middle'}); label(g4, 1115, LOW.y + 19, 'portfolio', 18, '', {'text-anchor': 'middle'});
    function copies(u) { blocks.forEach(function (b, q) { b.setAttribute('opacity', part(u, q * 0.22, q * 0.22 + 0.3)); }); cnt.setAttribute('opacity', u > 0.85 ? 1 : 0); rel.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 4) { if (animate && k === 4) tween(box, 1600, copies); else copies(1); }

    // scene B, step 5: what a half-life is: w(l) = 2^(-l/h) on one pair of axes, enlarged for a moment, then kept small by the exposure stage
    var mini = el('g', {opacity: 0}, s), big = el('g', {opacity: 0}, s);
    function curves(into, x0, x1, yTop, yBot, full) {
      var X = lin(0, 300, x0, x1), Y = lin(0, 1, yBot, yTop);
      el('line', {x1: x0, x2: x1, y1: Y(0), y2: Y(0), class: 'ln-axis'}, into); el('line', {x1: x0, x2: x0, y1: yTop, y2: yBot, class: 'ln-axis'}, into);
      el('line', {x1: x0, x2: x1, y1: Y(0.5), y2: Y(0.5), stroke: COL.mist, 'stroke-dasharray': '6 5', 'stroke-width': full ? 1.5 : 1}, into);
      if (full) {
        [0, 0.5, 1].forEach(function (v) { label(into, x0 - 8, Y(v) + 5, v.toFixed(1), 13, 't-mono t-quiet', {'text-anchor': 'end'}); });
        [0, 100, 200, 300].forEach(function (v) { label(into, X(v), yBot + 20, String(v), 13, 't-mono t-quiet', {'text-anchor': 'middle'}); });
        label(into, x1, yBot + 44, 'Lag, trading days', 14, 't-mono t-quiet', {'text-anchor': 'end'});
        label(into, x0 - 44, (yTop + yBot) / 2, 'Relative observation weight', 14, 't-mono t-quiet', {'text-anchor': 'middle', transform: 'rotate(-90 ' + (x0 - 44) + ' ' + ((yTop + yBot) / 2) + ')'});
        rich(into, x0 + 150, yTop - 14, [['w(ℓ) = 2'], ['−ℓ/h', 'sup'], [',  h ∈ {' + za.half_lives.join(', ') + '}']], {'font-size': 18, class: 't-serif'});
      }
      var paths = [];
      za.half_lives.forEach(function (h, q) {
        var d = '';
        for (var l = 0; l <= 300; l += 3) d += (l ? 'L' : 'M') + X(l).toFixed(1) + ' ' + Y(Math.pow(2, -l / h)).toFixed(1);
        paths.push(el('path', {d: d, fill: 'none', stroke: [COL.cobalt, '#5A6FD9', '#9AA6E8'][q], 'stroke-width': full ? 2.5 : 1.5}, into));
        el('circle', {cx: X(h), cy: Y(0.5), r: full ? 5 : 2.5, fill: [COL.cobalt, '#5A6FD9', '#9AA6E8'][q]}, into);
        if (full) label(into, X(h) + 8, Y(0.5) + (q === 1 ? 24 : -10), h + '-day half-life', 13, 't-mono', {'font-weight': 500});
      });
      return paths;
    }
    var bigPaths = curves(big, 650, 1170, 318, 540, true);
    rrect(big, 560, 282, 630, 300, {fill: 'rgba(255,255,255,.96)', stroke: COL.cobalt, 'stroke-width': 1.5, rx: 8}); big.insertBefore(big.lastChild, big.firstChild);
    curves(mini, 568, 790, 332, 382, false);
    function demo(u) {   // enlarge, draw the three curves, hold, then keep the small version by the exposure stage
      var a = part(u, 0, 0.12), b = part(u, 0.12, 0.5), c2 = part(u, 0.82, 1);
      big.setAttribute('opacity', a * (1 - c2));
      bigPaths.forEach(function (p) { drawOn(p, b); });
      mini.setAttribute('opacity', c2);
    }
    if (k === 5 && animate) tween(box, 4200, demo);
    else if (k >= 5) { big.setAttribute('opacity', 0); bigPaths.forEach(function (p) { drawOn(p, 1); }); mini.setAttribute('opacity', 1); }

    // scene C, step 6: the separate replication of the paper with the authors' own LDA outputs
    var g6 = reveal(box, el('g', {}, s), k, 6, animate, 500);
    label(g6, 1230, 40, 'Sparse IPCA replication', 20, '', {'font-weight': 600});
    label(g6, 1230, 64, 'Authors’ LDA outputs', 15, 't-mono t-quiet');
    var RX = lin(1, 6, 1290, 1560), RY = lin(0.4, 1.4, 400, 110);
    [0.4, 0.6, 0.8, 1.0, 1.2, 1.4].forEach(function (v) { el('line', {x1: 1290, x2: 1560, y1: RY(v), y2: RY(v), class: 'ln-rule'}, g6); label(g6, 1282, RY(v) + 4, v.toFixed(1), 12, 't-mono t-quiet', {'text-anchor': 'end'}); });
    rp.k.forEach(function (kk) { label(g6, RX(kk), 424, String(kk), 12, 't-mono t-quiet', {'text-anchor': 'middle'}); });
    label(g6, 1560, 450, 'Factors, K', 14, 't-mono t-quiet', {'text-anchor': 'end'});
    label(g6, 1244, 255, 'Sharpe', 14, 't-mono t-quiet', {'text-anchor': 'middle', transform: 'rotate(-90 1244 255)'});
    function series(vals, col) {
      var d = ''; vals.forEach(function (v, q) { d += (q ? 'L' : 'M') + RX(rp.k[q]).toFixed(1) + ' ' + RY(v).toFixed(1); });
      var p = el('path', {d: d, fill: 'none', stroke: col, 'stroke-width': 2}, g6), dots = el('g', {}, g6);
      vals.forEach(function (v, q) { el('circle', {cx: RX(rp.k[q]), cy: RY(v), r: 4, fill: col}, dots); });
      return {p: p, dots: dots};
    }
    var sp = series(rp.paper, CHAR), sr = series(rp.replication, COL.cobalt);
    el('line', {x1: 1290, x2: 1312, y1: 492, y2: 492, stroke: CHAR, 'stroke-width': 2}, g6); label(g6, 1320, 497, 'Paper', 15, 't-mono');
    el('line', {x1: 1290, x2: 1312, y1: 516, y2: 516, stroke: COL.cobalt, 'stroke-width': 2}, g6); label(g6, 1320, 521, 'Replication', 15, 't-mono t-cobalt');
    var maxd = Math.max.apply(null, rp.paper.map(function (v, q) { return Math.abs(v - rp.replication[q]); }));
    var ro = label(g6, 1230, 566, 'Maximum absolute difference: ' + maxd.toFixed(3), 15, 't-mono', {'font-weight': 500});
    function repl(u) { var a = part(u, 0, 0.45), b = part(u, 0.5, 0.95); drawOn(sp.p, a); sp.dots.setAttribute('opacity', a >= 1 ? 1 : 0); drawOn(sr.p, b); sr.dots.setAttribute('opacity', b >= 1 ? 1 : 0); ro.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 6) { if (animate && k === 6) tween(box, 2400, repl); else repl(1); }
  });

  // ------------------------------------------------------------------ 1.3 the earlier MVE factor portfolio: its Sharpe, its two components, its exposure, and what follows
  C.mvo = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, M = D().facts.mvo, main = el('g', {}, s);
    label(main, 0, 24, 'Earlier FNSPID portfolio · ' + monthName(M.from) + '–' + monthName(M.to), 17, 't-mono t-quiet');
    // scene A, step 1: the reported result on a Sharpe axis
    var g1 = reveal(box, el('g', {}, main), k, 1, animate, 500), SXa = lin(0, 1.4, 1000, 1540), rows = [[M.sharpe, 'Full portfolio', 130, COL.cobalt], [M.market, 'Market component', 210, CHAR], [M.tilt, 'Zero-net stock tilt', 290, CHAR]];
    [0, 0.5, 1.0].forEach(function (v) { el('line', {x1: SXa(v), x2: SXa(v), y1: 100, y2: 330, class: v ? 'ln-rule' : 'ln-axis'}, g1); label(g1, SXa(v), 352, v.toFixed(1), 14, 't-mono t-quiet', {'text-anchor': 'middle'}); });
    label(g1, 1540, 380, 'Sharpe', 16, 't-mono t-quiet', {'text-anchor': 'end'});
    label(g1, 990, 137, rows[0][1], 20, '', {'text-anchor': 'end', 'font-weight': 500});
    var dot0 = el('circle', {cx: SXa(M.sharpe), cy: 130, r: 9, fill: COL.cobalt}, g1);
    label(g1, SXa(M.sharpe) + 18, 137, 'Sharpe: ' + M.sharpe.toFixed(2), 18, 't-mono', {'font-weight': 500});
    // scene B, step 2: the weights as a market position plus a zero-net tilt (schematic)
    var g2 = reveal(box, el('g', {}, main), k, 2, animate, 500);
    label(g2, 0, 54, 'Weight decomposition · Schematic', 15, 't-mono t-quiet');
    rrect(g2, 120, 66, 300, 44, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5}); label(g2, 270, 95, 'Full portfolio', 19, '', {'text-anchor': 'middle', 'font-weight': 500});
    var boxes = [[0, 'Equal-weighted market component'], [370, 'Zero-net stock tilt']];
    boxes.forEach(function (b) { rrect(g2, b[0], 150, 330, 56, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}); label(g2, b[0] + 165, 185, b[1], 18, '', {'text-anchor': 'middle'}); });
    arrow(g2, 230, 110, 165, 146, {stroke: CHAR}); arrow(g2, 310, 110, 535, 146, {stroke: CHAR});
    rich(g2, 370, 254, [['w'], ['i,t', 'sub'], [' = m'], ['t', 'sub'], [' / N'], ['t', 'sub'], [' + w̃'], ['i,t', 'sub'], [',   Σ'], ['i', 'sub'], [' w̃'], ['i,t', 'sub'], [' = 0']], {'font-size': 21, class: 't-serif'});
    // scene C, step 3: the average net exposure on a 0x to 5x scale
    var g3 = reveal(box, el('g', {}, main), k, 3, animate, 500), EX = lin(0, 5, 80, 700);
    label(g3, 0, 320, 'Average net exposure: ' + M.exposure.toFixed(1) + '×', 24, '', {'font-weight': 600});
    el('line', {x1: EX(0), x2: EX(5), y1: 360, y2: 360, class: 'ln-axis', 'stroke-width': 1.5}, g3);
    for (var v = 0; v <= 5; v++) { el('line', {x1: EX(v), x2: EX(v), y1: 360, y2: v === 1 ? 372 : 366, class: 'ln-axis'}, g3); label(g3, EX(v), 386, v + '×', 14, 't-mono t-quiet', {'text-anchor': 'middle'}); }
    label(g3, EX(1), 406, '1× equity', 15, 't-mono', {'text-anchor': 'middle', 'font-weight': 500});
    var bar = el('rect', {x: EX(0), y: 338, width: 0, height: 20, fill: COL.cobalt}, g3);
    el('line', {x1: 40, x2: 40, y1: 206, y2: 296, stroke: COL.cobalt, 'stroke-width': 1.5, 'stroke-dasharray': '4 4'}, g3);
    function expo(u) { bar.setAttribute('width', (EX(M.exposure) - EX(0)) * u); }
    if (k >= 3) { if (animate && k === 3) tween(box, 1100, function (u) { expo(ease(u)); }); else expo(1); }
    // scene D, step 4: three independent Sharpe rows on the same axis
    var g4 = reveal(box, el('g', {}, main), k, 4, animate, 500);
    rows.slice(1).forEach(function (r) {
      label(g4, 990, r[2] + 7, r[1], 20, '', {'text-anchor': 'end'});
      el('circle', {cx: SXa(r[0]), cy: r[2], r: 8, fill: r[3]}, g4);
      label(g4, SXa(r[0]) + 18, r[2] + 7, r[0].toFixed(2), 18, 't-mono', {'font-weight': 500});
    });
    // scene E, step 5: the charts give a little room; two research directions and the closing sentence
    var g5 = reveal(box, el('g', {}, s), k, 5, animate, 500);
    [[0, 'Portfolio construction', 'Long–short · Buffering · Market tilts'], [370, 'Return prediction', 'Price-only controls · NoLBERT · EBM']].forEach(function (d) {
      arrow(g5, d[0] + 12, 500, d[0] + 12, 526, {stroke: COL.cobalt, width: 1.8});
      label(g5, d[0], 556, d[1], 21, '', {'font-weight': 600}); label(g5, d[0], 582, d[2], 16, 't-mono t-quiet');
    });
    label(g5, 0, 648, 'Separate forecast quality from market exposure and portfolio construction.', 22);
    function shrink(u) { var sc = mix(1, 0.92, u); main.setAttribute('transform', 'scale(' + sc.toFixed(4) + ')'); }
    if (k >= 5) { if (animate && k === 5) tween(box, 600, function (u) { shrink(ease(u)); }); else shrink(1); }
  });

  // ------------------------------------------------------------------ 5.6 validation against test: the ten best validation ICs followed into the test period
  C.vtest = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, V = D().vt, EXT = 0.08, PW = 440, GAPX = 110, top = 70, H = 400, Y = lin(-EXT, EXT, top + H, top);
    label(s, 8, top + H / 2, 'Rank IC', 15, 't-mono t-quiet', {'text-anchor': 'middle', transform: 'rotate(-90 8 ' + (top + H / 2) + ')'});
    var panels = [];
    V.panels.forEach(function (p, pi) {
      var x0 = 30 + pi * (PW + GAPX), xv = x0 + 150, xt = x0 + 340, P = {vals: [], tests: [], links: [], tops: []};
      label(s, x0 + 60, 34, p.name, 22, '', {'font-weight': 600});
      label(s, x0 + 60, 56, p.n + ' experiments', 14, 't-mono t-quiet');
      [-0.08, -0.04, 0, 0.04, 0.08].forEach(function (v) {
        el('line', {x1: x0 + 44, x2: x0 + PW, y1: Y(v), y2: Y(v), stroke: v ? COL.rule : COL.ink, 'stroke-width': v ? 1 : 1.5}, s);
        label(s, x0 + 36, Y(v) + 4, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(2), 13, 't-mono t-quiet', {'text-anchor': 'end'});
      });
      label(s, xv, top + H + 26, 'Validation', 15, 't-mono t-quiet', {'text-anchor': 'middle'});
      label(s, xt, top + H + 26, 'Test', 15, 't-mono t-quiet', {'text-anchor': 'middle'});
      var gl = el('g', {}, s), gv = el('g', {}, s), gt = el('g', {}, s);
      p.pts.forEach(function (q) {
        var j = (hash(q.key) - 0.5) * 84, yv = Y(Math.max(-EXT, Math.min(EXT, q.val))), yt = Y(Math.max(-EXT, Math.min(EXT, q.test)));
        var dv = el('circle', {cx: xv + j, cy: yv, r: 3.5, fill: GREY, opacity: 0}, gv);
        var dt = el('circle', {cx: xt + j, cy: yt, r: q.top ? 5.5 : 3.5, fill: q.top ? COL.cobalt : GREY, opacity: 0}, gt);
        var ln = q.top ? el('line', {x1: xv + j, y1: yv, x2: xt + j, y2: yt, stroke: COL.cobalt, 'stroke-width': 1, opacity: 0}, gl) : null;
        P.vals.push(dv); P.tests.push(dt); P.links.push(ln); P.tops.push(!!q.top);
      });
      P.tick = el('line', {x1: xt - 24, x2: xt + 24, y1: Y(p.fav), y2: Y(p.fav), stroke: COL.cobalt, 'stroke-width': 3, opacity: 0}, s);
      P.lab = label(s, x0 + 60, top + H + 60, 'Mean test IC', 15, 't-mono t-quiet', {opacity: 0});
      P.val = label(s, x0 + 60, top + H + 86, signed(p.fav, 3), 20, 't-mono t-cobalt', {'font-weight': 500, opacity: 0});
      panels.push(P);
    });
    var leg = el('g', {opacity: 0}, s);
    el('circle', {cx: 38, cy: 600, r: 5.5, fill: COL.cobalt}, leg); label(leg, 52, 605, 'Top 10 by validation IC', 16, 't-mono t-cobalt');
    var cap = label(s, 30, 646, 'High validation IC does not persist in test.', 22, '', {opacity: 0});
    function show(u) {   // step 1: the validation points
      panels.forEach(function (P) { P.vals.forEach(function (d, i) { d.setAttribute('opacity', part(u, (i % 20) / 40, (i % 20) / 40 + 0.5)); }); });
    }
    function pick(u) {   // step 2: the ten highest validation ICs
      panels.forEach(function (P) { P.vals.forEach(function (d, i) {
        if (P.tops[i]) { d.setAttribute('fill', u > 0.5 ? COL.cobalt : GREY); d.setAttribute('r', 3.5 + 2 * u); d.setAttribute('opacity', 1); }
        else d.setAttribute('opacity', 1 - 0.55 * u);
      }); });
      leg.setAttribute('opacity', u);
    }
    function follow(u) {   // step 3: their test outcomes, joined to the same experiment
      panels.forEach(function (P) { P.tests.forEach(function (d, i) { if (P.tops[i]) { d.setAttribute('opacity', u); P.links[i].setAttribute('opacity', 0.7 * u); } }); });
    }
    function rest(u) {   // step 4: every other experiment's test IC
      panels.forEach(function (P) { P.tests.forEach(function (d, i) { if (!P.tops[i]) d.setAttribute('opacity', 0.45 * part(u, (i % 20) / 40, (i % 20) / 40 + 0.5)); }); });
    }
    function mean(u) {   // step 5: the mean test IC of the ten
      panels.forEach(function (P) { P.tick.setAttribute('opacity', u); P.lab.setAttribute('opacity', u >= 1 ? 1 : 0); P.val.setAttribute('opacity', u >= 1 ? 1 : 0); });
      cap.setAttribute('opacity', u >= 1 ? 1 : 0);
    }
    var stages = [show, pick, follow, rest, mean];
    stages.forEach(function (f, i) { var st = i + 1; if (k >= st) { if (animate && k === st) tween(box, [800, 450, 650, 800, 400][i], function (u) { f(ease(u)); }); else f(1); } });
  });

  // ------------------------------------------------------------------ B19 weekly forecasts: price controls alone and with the news, against the historical mean
  C.weekly = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, W = D().weekly, sp = D().facts.splits, X = lin(-8, 2, 300, 1500), rows = [170, 260, 350, 440];
    // step 1: the timing: the news comes before the week whose return is the target; then the strip settles at the top
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 400);
    var tA = el('rect', {x: 300, y: 14, width: 0, height: 60, rx: 4, fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5}, g1);
    var tB = el('rect', {x: 912, y: 14, width: 0, height: 60, rx: 4, fill: COL.wash, stroke: CHAR, 'stroke-width': 1.5}, g1);
    var lA = label(g1, 316, 50, 'News available before the return window', 16, 't-mono', {opacity: 0});
    var lB = label(g1, 928, 50, 'Next-week return', 16, 't-mono', {opacity: 0});
    el('line', {x1: 906, x2: 906, y1: 8, y2: 80, stroke: CHAR, 'stroke-width': 1.5}, g1);
    function timing(u) {
      var a = part(u, 0, 0.4), b = part(u, 0.4, 0.75), c2 = part(u, 0.8, 1), h = mix(60, 36, c2), y = mix(14, 26, c2);
      tA.setAttribute('width', 600 * a); tB.setAttribute('width', 588 * b); lA.setAttribute('opacity', a >= 1 ? 1 : 0); lB.setAttribute('opacity', b >= 1 ? 1 : 0);
      tA.setAttribute('height', h); tB.setAttribute('height', h); tA.setAttribute('y', y); tB.setAttribute('y', y); lA.setAttribute('y', y + h / 2 + 5); lB.setAttribute('y', y + h / 2 + 5);
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 2200, timing); else timing(1); }
    // the axis: the historical mean is the zero line
    el('line', {x1: X(0), x2: X(0), y1: 128, y2: 480, stroke: COL.ink, 'stroke-width': 2}, s);
    label(s, X(0), 116, 'Historical mean', 16, 't-mono', {'text-anchor': 'middle', 'font-weight': 500});
    for (var v = -8; v <= 2; v += 2) { if (v) el('line', {x1: X(v), x2: X(v), y1: 128, y2: 480, class: 'ln-rule'}, s); label(s, X(v), 506, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v) + '%', 14, 't-mono t-quiet', {'text-anchor': 'middle'}); }
    label(s, 1500, 536, 'Out-of-sample R²', 16, 't-mono t-quiet', {'text-anchor': 'end'});
    W.rows.forEach(function (r, i) { label(s, 280, rows[i] + 7, r.label, 19, '', {'text-anchor': 'end'}); });
    // step 2: price controls; step 3: the news added; step 4: the numbers
    var ctl = [], news = [], links = [], labs = [];
    W.rows.forEach(function (r, i) {
      var y = rows[i], xc = X(r.controls * 100), xn = X(r.news * 100);
      links.push(el('line', {x1: xc, y1: y, x2: xc, y2: y, stroke: COL.cobalt, 'stroke-width': 1.5, opacity: 0}, s));
      ctl.push(el('circle', {cx: xc, cy: y, r: 9, fill: '#fff', stroke: CHAR, 'stroke-width': 2, opacity: 0}, s));
      news.push(el('circle', {cx: xn, cy: y, r: 9, fill: COL.cobalt, opacity: 0}, s));
      var nearZero = Math.abs(xc - X(0)) < 40, HALO = {stroke: '#fff', 'stroke-width': 5, 'stroke-linejoin': 'round', 'paint-order': 'stroke'};
      labs.push([label(s, nearZero ? xc + 16 : xc, nearZero ? y + 5 : y - 18, signed(r.controls * 100, 2) + '%', 15, 't-mono t-quiet', Object.assign({'text-anchor': nearZero ? 'start' : 'middle', opacity: 0}, HALO)),
                 label(s, xn, y + 32, signed(r.news * 100, 2) + '%', 15, 't-mono t-cobalt', Object.assign({'text-anchor': 'middle', 'font-weight': 500, opacity: 0}, HALO))]);
    });
    var leg = el('g', {opacity: 0}, s);
    el('circle', {cx: 310, cy: 578, r: 7, fill: '#fff', stroke: CHAR, 'stroke-width': 2}, leg); label(leg, 326, 583, 'Price controls', 16, 't-mono t-quiet');
    el('circle', {cx: 520, cy: 578, r: 7, fill: COL.cobalt}, leg); label(leg, 536, 583, 'Price controls + news', 16, 't-mono t-cobalt');
    label(leg, 1500, 583, sp.test[0].slice(0, 4) + '–' + sp.test[1].slice(0, 4) + ' · ' + W.weeks + ' test weeks', 16, 't-mono t-quiet', {'text-anchor': 'end'});
    var cap = label(s, 300, 646, 'All four news models have R² below zero.', 22, '', {opacity: 0});
    function controls(u) { ctl.forEach(function (c, i) { c.setAttribute('opacity', part(u, i * 0.15, i * 0.15 + 0.5)); }); leg.setAttribute('opacity', u); }
    function addNews(u) {
      W.rows.forEach(function (r, i) { var xc = X(r.controls * 100), xn = X(r.news * 100); links[i].setAttribute('x2', mix(xc, xn, ease(u))); links[i].setAttribute('opacity', 1); news[i].setAttribute('opacity', u >= 1 ? 1 : 0); });
    }
    function numbers(u) { labs.forEach(function (l) { l[0].setAttribute('opacity', u); l[1].setAttribute('opacity', u); }); cap.setAttribute('opacity', u >= 1 ? 1 : 0); }
    [[2, controls, 1200], [3, addNews, 1000], [4, numbers, 500]].forEach(function (st) { if (k >= st[0]) { if (animate && k === st[0]) tween(box, st[2], st[1]); else st[1](1); } });
  });

  // ------------------------------------------------------------------ 5.7 same-month explanation: when the information exists, what the crude result measures, the four assets, the sample
  C.xasset = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, XA = D().xasset, sp = D().facts.splits, crude = XA.rows[0], X = lin(-60, 60, 300, 1340), rowsY = [190, 270, 350, 430];
    // scene A, step 1: both same-month tracks fill the month; only the completed stock track feeds the factors; the coefficients are older
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 400);
    rrect(g1, 0, 20, 290, 34, {fill: COL.wash, stroke: CHAR, 'stroke-width': 1.2}); label(g1, 145, 43, 'Coefficients fitted before test', 14, 't-mono', {'text-anchor': 'middle'});
    rrect(g1, 300, 6, 460, 80, {fill: 'none', stroke: COL.rule, 'stroke-width': 1.5, rx: 6});
    label(g1, 320, 26, 'Same-month stock returns', 13, 't-mono t-quiet'); var trS = el('rect', {x: 320, y: 32, width: 0, height: 9, fill: COL.cobalt}, g1);
    label(g1, 320, 60, 'Same-month futures returns', 13, 't-mono t-quiet'); var trF = el('rect', {x: 320, y: 66, width: 0, height: 9, fill: CHAR}, g1);
    var fac = el('g', {opacity: 0}, g1);
    arrow(fac, 742, 36, 796, 46, {stroke: COL.cobalt});
    rrect(fac, 800, 28, 150, 36, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5}); label(fac, 875, 52, 'Stock factors', 16, '', {'text-anchor': 'middle', 'font-weight': 500});
    function timing(u) { var a = part(u, 0, 0.6); trS.setAttribute('width', 420 * a); trF.setAttribute('width', 420 * a); fac.setAttribute('opacity', part(u, 0.7, 1)); }
    if (k >= 1) { if (animate && k === 1) tween(box, 2000, timing); else timing(1); }

    // scene B, step 2: the crude-oil squared error with the market alone and with the factors, on one scale
    var g2 = el('g', {opacity: 0}, s), BX = lin(0, 100, 300, 1000), ratio = crude.mspe_ratio * 100, gainTxt = (crude.gain * 100).toFixed(1) + '% lower squared error';
    label(g2, 300, 150, 'Crude oil', 24, '', {'font-weight': 600});
    label(g2, 280, 206, 'Market only', 18, '', {'text-anchor': 'end'}); var b1 = el('rect', {x: 300, y: 180, width: 0, height: 40, fill: LIGHT, stroke: CHAR, 'stroke-width': 1.5}, g2);
    label(g2, 280, 276, 'Market + factors', 18, '', {'text-anchor': 'end'}); var b2 = el('rect', {x: 300, y: 250, width: 0, height: 40, fill: COL.cobalt}, g2);
    label(g2, 300, 334, 'Squared error, market only = 100', 15, 't-mono t-quiet');
    var br = el('g', {opacity: 0}, g2);
    el('path', {d: 'M' + BX(ratio) + ' 302 v8 H ' + BX(100) + ' v-8', fill: 'none', stroke: COL.cobalt, 'stroke-width': 1.5}, br);
    label(br, (BX(ratio) + BX(100)) / 2, 334, gainTxt, 18, 't-mono t-cobalt', {'text-anchor': 'middle', 'font-weight': 500});
    function bars(u) { b1.setAttribute('width', (BX(100) - BX(0)) * part(u, 0, 0.35)); b2.setAttribute('width', (BX(ratio) - BX(0)) * part(u, 0.35, 0.7)); br.setAttribute('opacity', part(u, 0.8, 1)); }
    // the same comparison, small, kept at the top right from scene C on
    var inset = el('g', {opacity: 0}, s), IX = lin(0, 100, 1180, 1480);
    label(inset, 1100, 22, 'Crude oil', 14, 't-mono', {'font-weight': 500});
    label(inset, 1174, 42, 'Market only', 11, 't-mono t-quiet', {'text-anchor': 'end'}); el('rect', {x: 1180, y: 32, width: IX(100) - IX(0), height: 12, fill: LIGHT, stroke: CHAR, 'stroke-width': 1}, inset);
    label(inset, 1174, 64, 'Market + factors', 11, 't-mono t-quiet', {'text-anchor': 'end'}); el('rect', {x: 1180, y: 54, width: IX(ratio) - IX(0), height: 12, fill: COL.cobalt}, inset);
    label(inset, 1100, 92, gainTxt, 13, 't-mono t-cobalt', {'font-weight': 500});
    label(inset, 1480, 110, 'Squared error, market only = 100', 11, 't-mono t-quiet', {'text-anchor': 'end'});

    // scene C, step 3: four assets, each estimate with its whole interval
    var g3 = reveal(box, el('g', {}, s), k, 3, animate, 400);
    label(g3, 300, 140, 'FinBERT-20 · 5-day window · 3 factors', 15, 't-mono');
    label(g3, 1340, 140, '95% bootstrap intervals · unadjusted', 15, 't-mono t-quiet', {'text-anchor': 'end'});
    [-60, -40, -20, 0, 20, 40, 60].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 156, y2: 468, stroke: v ? COL.rule : COL.ink, 'stroke-width': v ? 1 : 2}, g3);
      label(g3, X(v), 494, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v) + '%', 14, 't-mono t-quiet', {'text-anchor': 'middle'});
    });
    label(g3, 1340, 524, 'Squared-error reduction', 16, 't-mono t-quiet', {'text-anchor': 'end'});
    var rowsG = XA.rows.map(function (r, i) {
      var y = rowsY[i], col = i ? CHAR : COL.cobalt, rg = el('g', {opacity: 0}, g3);
      label(rg, 280, y + 7, r.label, 19, '', {'text-anchor': 'end', 'font-weight': i ? 400 : 500});
      el('line', {x1: X(r.lo * 100), x2: X(r.hi * 100), y1: y, y2: y, stroke: col, 'stroke-width': 3}, rg);
      [r.lo, r.hi].forEach(function (b) { el('line', {x1: X(b * 100), x2: X(b * 100), y1: y - 7, y2: y + 7, stroke: col, 'stroke-width': 2}, rg); });
      el('circle', {cx: X(r.gain * 100), cy: y, r: 7, fill: col}, rg);
      label(rg, X(r.gain * 100), y - 16, signed(r.gain * 100, 1) + '%', 15, 't-mono' + (i ? '' : ' t-cobalt'), {'text-anchor': 'middle', 'font-weight': 500});
      label(rg, X(r.lo * 100), y + 26, signed(r.lo * 100, 1) + '%', 12, 't-mono t-quiet', {'text-anchor': 'middle'});
      label(rg, X(r.hi * 100), y + 26, signed(r.hi * 100, 1) + '%', 12, 't-mono t-quiet', {'text-anchor': 'middle'});
      return rg;
    });
    function intervals(u) {
      var a = part(u, 0, 0.3);
      g2.setAttribute('opacity', 1 - a); inset.setAttribute('opacity', a);
      rowsG.forEach(function (rg, i) { rg.setAttribute('opacity', part(u, 0.3 + i * 0.16, 0.3 + i * 0.16 + 0.2)); });
    }
    if (k === 2) { g2.setAttribute('opacity', 1); if (animate) tween(box, 1800, bars); else bars(1); }
    if (k >= 3) { bars(1); if (animate && k === 3) { g2.setAttribute('opacity', 1); tween(box, 2400, intervals); } else intervals(1); }

    // scene D, step 4: the months that exist in the test sample, the adjusted p-value, the timing qualification
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500), filled = 0, total = 0, y0 = +sp.test[0].slice(0, 4), y1 = +sp.test[1].slice(0, 4);
    for (var yy = y0; yy <= y1; yy++) {
      var cy = 560 + (yy - y0) * 26;
      label(g4, 332, cy + 14, String(yy), 14, 't-mono t-quiet', {'text-anchor': 'end'});
      for (var m = 1; m <= 12; m++) {
        var on = !(yy >= sp.unused_from && m >= 2 && m <= 6); total++; if (on) filled++;
        el('rect', {x: 345 + (m - 1) * 26, y: cy, width: 22, height: 18, rx: 2, fill: on ? SOFT2 : 'none', stroke: on ? COL.cobalt : COL.rule, 'stroke-width': 1}, g4);
      }
    }
    label(g4, 345, 650, filled + ' of ' + total + ' test months', 15, 't-mono');
    label(g4, 1370, 176, 'Primary Clark–West p', 14, 't-mono t-quiet');
    label(g4, 1370, 200, 'Holm, 4 tests: ' + crude.p_holm_4.toFixed(3), 17, 't-mono', {'font-weight': 500});
    label(g4, 1364, 228, 'FinBERT not point-in-time', 14, 't-mono t-red');
    label(g4, 1340, 650, 'Crude has the largest estimated reduction.', 22, '', {'text-anchor': 'end'});
  });

  // ------------------------------------------------------------------ B20 the crude-oil result under three text models, window and factor count held fixed
  C.textmodels = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, TM = D().textmodels, ref = D().xasset.rows[0], X = lin(-60, 60, 300, 1340), rowsY = [260, 340, 420];
    [-60, -40, -20, 0, 20, 40, 60].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 70, y2: 460, stroke: v ? COL.rule : COL.ink, 'stroke-width': v ? 1 : 2}, s);
      label(s, X(v), 488, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v) + '%', 14, 't-mono t-quiet', {'text-anchor': 'middle'});
    });
    label(s, 1340, 518, 'Squared-error reduction', 16, 't-mono t-quiet', {'text-anchor': 'end'});
    function row(into, y, r, col, big) {
      el('line', {x1: X(r.lo * 100), x2: X(r.hi * 100), y1: y, y2: y, stroke: col, 'stroke-width': big ? 3 : 2.5}, into);
      [r.lo, r.hi].forEach(function (b) { el('line', {x1: X(b * 100), x2: X(b * 100), y1: y - 7, y2: y + 7, stroke: col, 'stroke-width': 2}, into); });
      el('circle', {cx: X(r.gain * 100), cy: y, r: big ? 7 : 6, fill: col}, into);
      label(into, X(r.gain * 100), y - 16, signed(r.gain * 100, 1) + '%', 15, 't-mono', {'text-anchor': 'middle', 'font-weight': 500, style: 'fill:' + col});
      label(into, X(r.lo * 100), y + 26, signed(r.lo * 100, 1) + '%', 12, 't-mono t-quiet', {'text-anchor': 'middle'});
      label(into, X(r.hi * 100), y + 26, signed(r.hi * 100, 1) + '%', 12, 't-mono t-quiet', {'text-anchor': 'middle'});
    }
    // step 1: the original primary result, muted
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 500);
    label(g1, 300, 86, 'Original primary · FinBERT-20 · 5-day window', 15, 't-mono t-quiet');
    row(g1, 110, ref, '#9C9C9A', false);
    label(g1, 1364, 116, 'FinBERT not point-in-time', 14, 't-mono t-red');
    el('line', {x1: 300, x2: 1340, y1: 160, y2: 160, class: 'ln-rule', 'stroke-width': 1.5}, g1);
    // steps 2 to 4: the post hoc comparison, one text model at a time; nothing is removed
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 400);
    label(g2, 300, 198, 'Post hoc · 63-day window · 3 factors', 18, '', {'font-weight': 600});
    TM.rows.forEach(function (r, i) {
      var gi = reveal(box, el('g', {}, s), k, i + 2, animate, 500), y = rowsY[i], col = i === 0 ? COL.cobalt : CHAR;
      label(gi, 280, y + 7, r.label, 19, '', {'text-anchor': 'end', 'font-weight': i === 0 ? 500 : 400});
      row(gi, y, r, col, true);
      if (i === 0) {
        label(gi, 1370, 236, 'FinBERT Clark–West p', 14, 't-mono t-quiet'); label(gi, 1370, 258, 'Holm, 12 tests', 14, 't-mono t-quiet');
        label(gi, 1370, 282, 'Analytic: ' + r.holm12_analytic.toFixed(3), 16, 't-mono', {'font-weight': 500});
        label(gi, 1370, 304, 'Calibrated: ' + r.holm12_calibrated.toFixed(3), 16, 't-mono', {'font-weight': 500});
        label(gi, 1364, 330, 'FinBERT not point-in-time', 14, 't-mono t-red');
      }
    });
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500);
    ['Same-month explanation', 'Same ' + TM.rows[0].n_test + ' test months', '95% bootstrap intervals · unadjusted'].forEach(function (t, i) { label(g4, 300, 566 + i * 24, t, 15, 't-mono t-quiet'); });
    label(g4, 1340, 650, 'The crude-oil result varies across text models.', 22, '', {'text-anchor': 'end'});
  });
})();
