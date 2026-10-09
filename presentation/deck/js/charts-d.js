/* Exhibits, part D: slides 2.1, 2.2, 3.1, 3.2, 3.3 and backup B11 (the NoLBERT revision of 9 October 2026).
   Every number is read from window.DATA (tools/build_data.py) or from window.DATA.similarity
   (tools/export_similarity.py, gitignored: without it 3.1 draws the calculation schematically and 3.3
   omits the matching diagram). The labels are the slide copy agreed with the team, used verbatim.
   Each chart has enter(box, step), step(box, k), leave(box) and still(box): still draws the last frame.
   The encoder of 2.2 and 3.1 is one drawing (ENC), so the two slides agree. */
(function () {
  'use strict';
  var U = window.DECKUTIL, el = U.el, tx = U.tx, surface = U.surface, lin = U.lin, ease = U.ease, tween = U.tween,
      halt = U.halt, reveal = U.reveal, COL = U.COL, day = U.day;
  var C = window.CHARTS = window.CHARTS || {};
  var D = function () { return window.DATA; };
  var SIM = function () { var s = window.DATA.similarity; return s && s.available && s.batches ? s : null; };
  var SOFT = 'rgba(44,75,239,.12)', SOFT2 = 'rgba(44,75,239,.28)', REDSOFT = 'rgba(224,68,43,.12)', SLAB = '#D6D9EA', NS = 'http://www.w3.org/2000/svg';

  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); box._k = -1; build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; var from = box._k; box._k = k; build(box, k, true, from); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k, false); }
    };
  }
  function part(u, a, b) { return Math.max(0, Math.min(1, (u - a) / (b - a))); }   // the share of [a, b] that u has covered
  function mix(a, b, v) { return a + (b - a) * v; }
  function rrect(p, x, y, w, h, a) { return el('rect', Object.assign({x: x, y: y, width: w, height: h, rx: 6}, a || {}), p); }
  function label(p, x, y, s, size, cls, a) { return tx(p, x, y, s, Object.assign({'font-size': size || 20, class: cls || ''}, a || {})); }
  function arrow(p, x1, y1, x2, y2, a) {   // a straight connector with a small head; width 0 draws the head only
    a = a || {};
    var g = el('g', {}, p), ang = Math.atan2(y2 - y1, x2 - x1), h = a.head || 9, col = a.stroke || COL.mist;
    if (a.width !== 0) el('line', {x1: x1, y1: y1, x2: x2 - 2 * Math.cos(ang), y2: y2 - 2 * Math.sin(ang), stroke: col, 'stroke-width': a.width || 1.8, 'stroke-dasharray': a.dash || null}, g);
    el('path', {d: 'M' + x2 + ' ' + y2 + 'L' + (x2 - h * Math.cos(ang - 0.42)) + ' ' + (y2 - h * Math.sin(ang - 0.42)) +
      'L' + (x2 - h * Math.cos(ang + 0.42)) + ' ' + (y2 - h * Math.sin(ang + 0.42)) + 'Z', fill: col}, g);
    return g;
  }
  function drawOn(path, u) {   // a path drawn on from its start: the dash offset follows u
    var L = path.getTotalLength ? path.getTotalLength() : 1000;
    path.setAttribute('stroke-dasharray', L + ' ' + L); path.setAttribute('stroke-dashoffset', L * (1 - u));
  }
  function cosFill(v) {   // one colour scale for every similarity cell: cobalt for positive, vermilion for negative cosines
    v = Math.max(-1, Math.min(1, v));
    return v >= 0 ? 'rgba(44,75,239,' + v.toFixed(3) + ')' : 'rgba(224,68,43,' + (-v).toFixed(3) + ')';
  }
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
  function hatch(s, id, col) {
    var defs = el('defs', {}, s), p = el('pattern', {id: id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'}, defs);
    el('line', {x1: 0, y1: 0, x2: 0, y2: 6, stroke: col, 'stroke-width': 2}, p);
    return 'url(#' + id + ')';
  }

  // ------------------------------------------------------------------ the encoder drawing shared by 2.2 and 3.1
  var ENC = {x: 300, y: 110, w: 340, h: 360, n: 12, slab: 20, gap: 7.5, rowY0: 150, pitch: 46, rowH: 34};
  function slabX(i) { return ENC.x + 12 + i * (ENC.slab + ENC.gap); }
  function rowY(i) { return ENC.rowY0 + i * ENC.pitch; }
  function encoder(s, o) {
    o = o || {};
    var g = el('g', {}, s), r = {g: g, slabs: []};
    r.block = rrect(g, ENC.x, ENC.y, ENC.w, ENC.h, {fill: COL.wash, stroke: COL.rule, 'stroke-width': 1.5});
    for (var i = 0; i < ENC.n; i++) r.slabs.push(el('rect', {x: slabX(i), y: ENC.y + 16, width: ENC.slab, height: ENC.h - 32, rx: 3, fill: SLAB}, g));
    if (o.title) label(g, ENC.x, ENC.y - 20, o.title, 22, '', {'font-weight': 600});
    (o.under || []).forEach(function (t, j) { label(g, ENC.x, ENC.y + ENC.h + 32 + j * 28, t, 20, 't-mono t-quiet'); });
    return r;
  }

  // ------------------------------------------------------------------ 2.1 the two biases, as two diagrams
  C.bias = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    // left: a forecast made at one date; text written later; its information flowing back into the forecast
    var gl = reveal(box, el('g', {}, s), k, 1, animate, 500), ax = 330, fx = 250;
    label(gl, 0, 40, 'Look-ahead bias', 40, 't-serif', {'font-weight': 500});
    el('line', {x1: 20, x2: 700, y1: ax, y2: ax, class: 'ln-axis', 'stroke-width': 2}, gl);
    el('path', {d: 'M700 ' + ax + 'l-12 -7v14z', fill: COL.mist}, gl);
    el('circle', {cx: fx, cy: ax, r: 11, fill: COL.cobalt}, gl);
    label(gl, fx, ax + 46, 'Forecast date', 21, '', {'text-anchor': 'middle', 'font-weight': 500});
    rrect(gl, 460, 232, 220, 70, {fill: '#fff', stroke: COL.red, 'stroke-width': 2, rx: 4});
    [170, 150, 120].forEach(function (w, i) { el('line', {x1: 478, x2: 478 + w, y1: 254 + i * 16, y2: 254 + i * 16, stroke: COL.mist, 'stroke-width': 3}, gl); });
    el('line', {x1: 570, x2: 570, y1: 302, y2: ax, stroke: COL.red, 'stroke-width': 2}, gl);
    el('circle', {cx: 570, cy: ax, r: 7, fill: COL.red}, gl);
    label(gl, 570, 216, 'Later training text', 21, 't-red', {'text-anchor': 'middle', 'font-weight': 500});
    var arc = el('path', {d: 'M460 252 C 400 150, 330 130, 262 ' + (ax - 16), fill: 'none', stroke: COL.red, 'stroke-width': 3}, gl);
    var head = arrow(gl, 286, ax - 60, 262, ax - 16, {stroke: COL.red, width: 0, head: 12});
    var cap = label(gl, 0, 520, 'Later information used for an earlier prediction.', 24, '', {opacity: 0});
    function back(u) { drawOn(arc, u); head.setAttribute('opacity', u >= 1 ? 1 : 0); cap.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 1) { if (animate && k === 1) tween(box, 1400, function (u) { back(ease(u)); }); else back(1); }

    // right: one phrase, two eras, two meanings; the later meaning carried back onto the earlier text
    var gr = reveal(box, el('g', {}, s), k, 2, animate, 500), ax2 = 330, x0 = 980, x1 = 1420;
    label(gr, 840, 40, 'Look-back bias', 40, 't-serif', {'font-weight': 500});
    label(gr, 1200, 150, 'Running a program', 36, 't-serif', {'text-anchor': 'middle', 'font-style': 'italic'});
    el('line', {x1: 1200, x2: x0, y1: 168, y2: ax2 - 14, stroke: COL.mist, 'stroke-width': 1.5, 'stroke-dasharray': '5 5'}, gr);
    el('line', {x1: 860, x2: 1560, y1: ax2, y2: ax2, class: 'ln-axis', 'stroke-width': 2}, gr);
    el('path', {d: 'M1560 ' + ax2 + 'l-12 -7v14z', fill: COL.mist}, gr);
    el('circle', {cx: x0, cy: ax2, r: 9, fill: COL.ink}, gr);
    el('circle', {cx: x1, cy: ax2, r: 9, fill: COL.red}, gr);
    label(gr, x0, ax2 + 40, 'Early 1900s', 20, 't-mono t-quiet', {'text-anchor': 'middle'});
    label(gr, x0, ax2 + 74, 'Organizing an event', 25, '', {'text-anchor': 'middle'});
    label(gr, x1, ax2 + 40, 'Late 20th century', 20, 't-mono t-quiet', {'text-anchor': 'middle'});
    label(gr, x1, ax2 + 74, 'Executing computer code', 25, 't-red', {'text-anchor': 'middle', 'font-weight': 500});
    var arc2 = el('path', {d: 'M' + x1 + ' ' + (ax2 + 90) + ' C ' + x1 + ' 480, ' + x0 + ' 480, ' + x0 + ' ' + (ax2 + 100), fill: 'none', stroke: COL.red, 'stroke-width': 3}, gr);
    var head2 = arrow(gr, x0, ax2 + 130, x0, ax2 + 98, {stroke: COL.red, width: 0, head: 12});
    var hl = rrect(gr, x0 - 134, ax2 + 52, 268, 32, {fill: 'none', stroke: COL.red, 'stroke-width': 2, rx: 4, opacity: 0});
    var cap2 = label(gr, 840, 520, 'Later word usage applied to earlier text.', 24, '', {opacity: 0});
    function back2(u) { drawOn(arc2, u); head2.setAttribute('opacity', u >= 1 ? 1 : 0); hl.setAttribute('opacity', u >= 1 ? 1 : 0); cap2.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 2) { if (animate && k === 2) tween(box, 1400, function (u) { back2(ease(u)); }); else back2(1); }
  });

  // ------------------------------------------------------------------ 2.2 the architecture, the original objective, the dates
  C.arch = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, nb = D().facts.nolbert, i, c;
    var words = ['The', 'company', 'reported', 'higher', '<mask>', '.'], MASK = 4, N = words.length;
    var E = encoder(s, {title: 'DeBERTa-v3 base', under: [nb.layers + ' encoder layers', '≈' + nb.params + ' parameters']});

    // scene A, step 1: tokens -> token embeddings -> the twelve layers -> token representations (tokens x 768)
    var g1 = el('g', {opacity: k >= 1 ? 1 : 0}, s);
    label(g1, 0, ENC.rowY0 - 24, 'Tokens', 20);
    label(g1, 290, ENC.rowY0 - 24, 'Token embeddings', 18, '', {'text-anchor': 'end'});
    label(g1, 700, ENC.rowY0 - 24, 'Token representations', 20);
    var toks = [], embs = [], outs = [];
    for (i = 0; i < N; i++) {
      var y = rowY(i), tg = el('g', {}, g1);
      var tb = rrect(tg, 0, y, 120, ENC.rowH, {fill: '#fff', stroke: COL.mist, 'stroke-width': 1.5, rx: 5});
      toks.push({g: tg, r: tb, t: label(tg, 60, y + 23, '', 19, '', {'text-anchor': 'middle'})});
      var eg = el('g', {}, g1);
      for (c = 0; c < 9; c++) el('rect', {x: 160 + c * 10, y: y + 11, width: 9, height: 12, fill: c % 2 ? SLAB : '#C3C8E3'}, eg);
      embs.push(eg);
      arrow(g1, 124, y + 17, 154, y + 17); arrow(g1, 254, y + 17, ENC.x - 4, y + 17); arrow(g1, ENC.x + ENC.w + 2, y + 17, 696, y + 17);
      var og = el('g', {}, g1);
      el('rect', {x: 700, y: y, width: 320, height: ENC.rowH, fill: '#fff', stroke: COL.rule}, og);
      for (c = 1; c < 40; c++) el('line', {x1: 700 + c * 8, x2: 700 + c * 8, y1: y + 1, y2: y + ENC.rowH - 1, stroke: '#E9EAF2'}, og);
      outs.push(og);
    }
    label(g1, 0, rowY(N) + 6, 'Vocabulary: ' + nb.vocab + ' tokens', 18, 't-mono t-quiet');
    var yb = rowY(N - 1) + ENC.rowH + 16;
    el('line', {x1: 700, x2: 1020, y1: yb, y2: yb, class: 'ln-axis', 'stroke-width': 1.5}, g1);
    el('line', {x1: 700, x2: 700, y1: yb - 6, y2: yb + 6, class: 'ln-axis', 'stroke-width': 1.5}, g1);
    el('line', {x1: 1020, x2: 1020, y1: yb - 6, y2: yb + 6, class: 'ln-axis', 'stroke-width': 1.5}, g1);
    label(g1, 700, yb + 26, 'Hidden size: ' + nb.hidden, 18, 't-mono t-quiet');
    function pass(u) {   // the forward pass: tokens, embeddings, a sweep through the twelve layers, the output
      toks.forEach(function (t, i) { t.g.setAttribute('opacity', part(u, 0.02 * i, 0.02 * i + 0.1)); });
      embs.forEach(function (e, i) { e.setAttribute('opacity', part(u, 0.12 + 0.02 * i, 0.22 + 0.02 * i)); });
      var sw = part(u, 0.3, 0.75);
      E.slabs.forEach(function (sl, j) { var v = part(sw, j / ENC.n, (j + 1.5) / ENC.n); sl.setAttribute('fill', v > 0 && v < 1 ? COL.cobalt : SLAB); });
      outs.forEach(function (o, i) { o.setAttribute('opacity', part(u, 0.72 + 0.02 * i, 0.82 + 0.02 * i)); });
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 2600, pass); else pass(1); }

    // scene A, step 2: one layer enlarged: content and relative position into attention, residual paths around both operations
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 600), L = 6;
    el('rect', {x: slabX(L) - 3, y: ENC.y + 13, width: ENC.slab + 6, height: ENC.h - 26, rx: 4, fill: 'none', stroke: COL.cobalt, 'stroke-width': 2}, g2);
    el('path', {d: 'M' + (slabX(L) + ENC.slab / 2) + ' ' + (ENC.y + 13) + ' V 72 H 1080', fill: 'none', stroke: COL.cobalt, 'stroke-width': 1.5, 'stroke-dasharray': '6 5'}, g2);
    var PX = 1080, cx = PX + 250, BW = 240, bx = cx - BW / 2, yIn = 118, yC = 150, yAtt = 212, yN1 = 288, yFF = 344, yN2 = 400, yOut = 440;
    rrect(g2, PX, 60, 500, 400, {fill: '#fff', stroke: COL.cobalt, 'stroke-width': 1.5, rx: 8});
    label(g2, PX + 20, 92, 'Encoder layer', 22, '', {'font-weight': 600});
    el('rect', {x: cx - 110, y: yIn, width: 220, height: 10, fill: SLAB}, g2);
    rrect(g2, PX + 60, yC, 150, 34, {fill: COL.wash, stroke: COL.rule}); label(g2, PX + 135, yC + 23, 'Content', 19, '', {'text-anchor': 'middle'});
    rrect(g2, PX + 250, yC, 190, 34, {fill: COL.wash, stroke: COL.rule}); label(g2, PX + 345, yC + 23, 'Relative position', 19, '', {'text-anchor': 'middle'});
    arrow(g2, cx - 40, yIn + 10, PX + 135, yC - 2); arrow(g2, cx + 40, yIn + 10, PX + 345, yC - 2);
    arrow(g2, PX + 135, yC + 34, cx - 60, yAtt - 2); arrow(g2, PX + 345, yC + 34, cx + 60, yAtt - 2);
    rrect(g2, bx, yAtt, BW, 56, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5}); label(g2, cx, yAtt + 24, 'Self-attention', 20, '', {'text-anchor': 'middle', 'font-weight': 500});
    label(g2, cx, yAtt + 46, nb.heads + ' attention heads', 15, 't-mono t-quiet', {'text-anchor': 'middle'});
    rrect(g2, bx, yN1, BW, 34, {fill: COL.wash, stroke: COL.rule}); label(g2, cx, yN1 + 23, 'Add + normalize', 19, '', {'text-anchor': 'middle'});
    rrect(g2, bx, yFF, BW, 40, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5}); label(g2, cx, yFF + 26, 'Feed-forward', 20, '', {'text-anchor': 'middle', 'font-weight': 500});
    rrect(g2, bx, yN2, BW, 34, {fill: COL.wash, stroke: COL.rule}); label(g2, cx, yN2 + 23, 'Add + normalize', 19, '', {'text-anchor': 'middle'});
    el('rect', {x: cx - 110, y: yOut, width: 220, height: 10, fill: SLAB}, g2);
    arrow(g2, cx, yAtt + 56, cx, yN1 - 2); arrow(g2, cx, yN1 + 34, cx, yFF - 2); arrow(g2, cx, yFF + 40, cx, yN2 - 2); arrow(g2, cx, yN2 + 34, cx, yOut - 2);
    var rx1 = PX + 34;
    el('path', {d: 'M' + (cx - 110) + ' ' + (yIn + 5) + ' H ' + rx1 + ' V ' + (yN1 + 17) + ' H ' + (bx - 12), fill: 'none', stroke: COL.cobalt, 'stroke-width': 1.8}, g2);
    arrow(g2, bx - 12, yN1 + 17, bx - 1, yN1 + 17, {stroke: COL.cobalt, width: 1.8});
    el('path', {d: 'M' + bx + ' ' + (yN1 + 28) + ' H ' + (rx1 + 12) + ' V ' + (yN2 + 17) + ' H ' + (bx - 12), fill: 'none', stroke: COL.cobalt, 'stroke-width': 1.8}, g2);
    arrow(g2, bx - 12, yN2 + 17, bx - 1, yN2 + 17, {stroke: COL.cobalt, width: 1.8});

    // scene B, step 3: the original objective, a masked token predicted by a head (schematic)
    var g3 = reveal(box, el('g', {}, s), k, 3, animate, 600), HY = 470;
    if (k >= 3) toks.forEach(function (t, i) { t.t.textContent = words[i]; if (i === MASK) { t.r.setAttribute('fill', REDSOFT); t.r.setAttribute('stroke', COL.red); } });
    rrect(g3, 700, HY, 320, 38, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5});
    label(g3, 860, HY + 25, 'Masked-token prediction', 20, '', {'text-anchor': 'middle', 'font-weight': 500});
    label(g3, 1020, HY - 10, 'Schematic', 15, 't-mono t-quiet', {'text-anchor': 'end'});
    label(g3, 700, HY + 66, 'The company reported higher <mask>.', 20);
    label(g3, 700, HY + 94, 'Target: earnings', 19, 't-mono t-cobalt');
    rich(g3, 700, HY + 128, [['L'], ['MLM', 'sub'], [' = −log P(target token)']], {'font-size': 24, class: 't-serif'});
    var link = el('path', {d: 'M1022 ' + (rowY(MASK) + 17) + ' H 1046 V ' + (HY + 19) + ' H 1032', fill: 'none', stroke: COL.red, 'stroke-width': 2}, g3);
    var lhead = arrow(g3, 1040, HY + 19, 1022, HY + 19, {stroke: COL.red, width: 0});
    if (k >= 3) { if (animate && k === 3) { lhead.setAttribute('opacity', 0); tween(box, 900, function (u) { drawOn(link, ease(u)); lhead.setAttribute('opacity', u >= 1 ? 1 : 0); }); } else drawOn(link, 1); }

    // scene C, step 4: the pretraining window and its validation year
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500), TX = lin(1976, 1997, 190, 640), ty = 622;
    label(g4, 0, ty + 21, 'Pretraining text', 20);
    var tb1 = el('rect', {x: TX(1976), y: ty, width: 0, height: 30, rx: 4, fill: SOFT, stroke: COL.cobalt, 'stroke-width': 2}, g4);
    var tl1 = label(g4, (TX(1976) + TX(1996)) / 2, ty + 21, 'Training: ' + nb.window[0] + '–' + nb.window[1], 18, 't-mono', {'text-anchor': 'middle', 'font-weight': 500, opacity: 0});
    var tb2 = el('rect', {x: TX(1996), y: ty, width: TX(1997) - TX(1996), height: 30, fill: 'rgba(44,75,239,.55)', opacity: 0}, g4);
    var tl2 = label(g4, TX(1997) + 12, ty + 21, 'Validation: ' + nb.val, 18, 't-mono t-quiet', {opacity: 0});
    label(g4, 1580, ty + 21, 'Kakhbod and Li (2025)', 18, 't-mono t-quiet', {'text-anchor': 'end'});
    function dates(u) { tb1.setAttribute('width', (TX(1996) - TX(1976)) * u); tl1.setAttribute('opacity', u > 0.9 ? 1 : 0); tb2.setAttribute('opacity', u >= 1 ? 1 : 0); tl2.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 4) { if (animate && k === 4) tween(box, 1200, function (u) { dates(ease(u)); }); else dates(1); }
  });

  // ------------------------------------------------------------------ 3.2 what changed, the batch, every similarity, the loss, the update
  C.finetune = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, sim = SIM(), ex = sim && sim.batches.train_example, T = D().training, tau = T.temperature, i, j;
    var PAD = hatch(s, 'ft-pad', COL.mist);
    var E = encoder(s, {});
    // the objective: the original one always; from step 1 the new one, the old struck through
    label(s, 0, 22, 'Original objective', 18, 't-mono t-quiet');
    label(s, 0, 54, 'Masked-token prediction', 25, k >= 1 ? 't-quiet' : '');
    var head0 = el('g', {opacity: k >= 1 ? 0 : 1}, s);   // the masked-token head of 2.2, removed by the new objective
    rrect(head0, 680, 170, 210, 36, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5});
    label(head0, 785, 194, 'Masked-token prediction', 16, '', {'text-anchor': 'middle'});
    arrow(head0, ENC.x + ENC.w + 2, 188, 676, 188);

    // scene A, step 1: two chunks of one article through one encoder
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 600);
    el('line', {x1: 0, x2: 300, y1: 46, y2: 46, stroke: COL.mist, 'stroke-width': 2, class: 'ln-strike'}, g1);
    arrow(g1, 330, 46, 396, 46, {stroke: COL.ink, width: 2});
    label(g1, 420, 22, 'Our objective', 18, 't-mono t-quiet');
    label(g1, 420, 54, 'Same-article retrieval', 25, 't-cobalt', {'font-weight': 600});
    if (animate && k === 1) tween(box, 500, function (u) { head0.setAttribute('opacity', 1 - u); });
    var CH = [{y: 150, name: 'Chunk A'}, {y: 300, name: 'Chunk B'}], chunks = [];
    CH.forEach(function (c) {
      var cg = el('g', {}, g1), cells = [], emb = [];
      rrect(cg, 0, c.y, 200, 80, {fill: '#fff', stroke: COL.ink, 'stroke-width': 1.5});
      label(cg, 16, c.y + 30, c.name, 22, '', {'font-weight': 500});
      for (j = 0; j < 14; j++) cells.push(el('rect', {x: 16 + j * 12, y: c.y + 46, width: 10, height: 18, fill: j < 11 ? SLAB : '#fff', stroke: j < 11 ? 'none' : COL.rule}, cg));
      var cut = el('line', {x1: 16 + 11 * 12 - 1, x2: 16 + 11 * 12 - 1, y1: c.y + 40, y2: c.y + 70, stroke: COL.red, 'stroke-width': 2, opacity: 0}, cg);
      for (j = 0; j < 5; j++) emb.push(el('rect', {x: 224 + j * 11, y: c.y + 34, width: 10, height: 12, fill: j % 2 ? SLAB : '#C3C8E3'}, cg));
      arrow(cg, 202, c.y + 40, 220, c.y + 40); arrow(cg, 282, c.y + 40, ENC.x - 4, c.y + 40);
      chunks.push({g: cg, cells: cells, cut: cut, emb: emb, y: c.y});
    });
    el('line', {x1: 100, x2: 100, y1: 232, y2: 298, stroke: COL.cobalt, 'stroke-width': 2}, g1);
    el('circle', {cx: 100, cy: 232, r: 3.5, fill: COL.cobalt}, g1); el('circle', {cx: 100, cy: 298, r: 3.5, fill: COL.cobalt}, g1);
    label(g1, 112, 271, 'Same article', 19, 't-mono t-cobalt');
    label(g1, ENC.x, ENC.y - 20, 'Shared encoder', 22, '', {'font-weight': 600});
    label(g1, ENC.x, ENC.y + ENC.h + 32, 'Shared weights', 20, 't-mono t-quiet');

    // scene B, step 2: the 128-token limit, the mean over the last layer (every one of the 128 positions, padding included: the run's
    // attention mask was `token id != 0` while the pad id is 1, train_nolbert.py:100 and :230), unit length
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 500), pools = [];
    label(g2, 0, 404, '128-token limit', 18, 't-mono t-quiet');
    chunks.forEach(function (c) {
      var pg = el('g', {}, g2), rows = [];
      for (j = 0; j < 8; j++) rows.push(el('rect', {x: 660, y: c.y + 4 + j * 8, width: 120, height: 7, fill: j < 6 ? SLAB : PAD, stroke: j < 6 ? 'none' : COL.rule, 'stroke-width': 0.8}, pg));
      var mean = el('rect', {x: 660, y: c.y + 23, width: 120, height: 10, rx: 3, fill: COL.cobalt, opacity: 0}, pg);
      var unit = el('g', {opacity: 0}, pg);
      el('circle', {cx: 822, cy: c.y + 28, r: 17, fill: 'none', stroke: COL.rule, 'stroke-width': 1.5}, unit);
      el('line', {x1: 822, y1: c.y + 28, x2: 835.6, y2: c.y + 17.8, stroke: COL.cobalt, 'stroke-width': 2.5}, unit);
      arrow(pg, ENC.x + ENC.w + 2, c.y + 28, 656, c.y + 28);
      pools.push({rows: rows, mean: mean, unit: unit, y: c.y});
    });
    label(g2, 660, 128, 'Last-layer mean', 19, '', {'font-weight': 500});
    var lPad = label(g2, 660, 262, '', 17, 't-mono t-quiet', {opacity: 0});   // Kevin's 3.1 (section3/): no padding label on the slide; the regime stays in the speaker notes
    var lL2 = label(g2, 660, 286, 'L2 normalization', 17, 't-mono t-quiet', {opacity: 0});
    function pool(u) {
      var a = part(u, 0, 0.3), b = part(u, 0.3, 0.7), c2 = part(u, 0.7, 1);
      if (a > 0) chunks.forEach(function (c) { c.cut.setAttribute('opacity', 1); c.cells.forEach(function (r, q) { if (q >= 11) r.setAttribute('fill', PAD); }); });
      pools.forEach(function (p) {
        p.rows.forEach(function (r, q) { r.setAttribute('y', mix(p.y + 4 + q * 8, p.y + 24, b)); r.setAttribute('opacity', 1 - 0.999 * b); });   // padding rows join the mean too
        p.mean.setAttribute('opacity', b >= 1 ? 1 : 0); p.unit.setAttribute('opacity', c2);
      });
      lPad.setAttribute('opacity', a >= 1 ? 1 : 0); lL2.setAttribute('opacity', c2 >= 1 ? 1 : 0);
    }
    if (k >= 2) { if (animate && k === 2) tween(box, 1400, pool); else pool(1); }

    // scene B, step 3: the batch, two aligned groups of 32 vectors that keep their article numbers
    var g3 = reveal(box, el('g', {}, s), k, 3, animate, 500), GA = 840, GB = 920, GY = 104, RP = 9, cellsA = [], cellsB = [];
    label(g3, GA, 80, '32 pairs · 64 chunks', 20, '', {'font-weight': 500});
    label(g3, GA, 98, 'Chunk A', 15, 't-mono t-quiet'); label(g3, GB, 98, 'Chunk B', 15, 't-mono t-quiet');
    for (i = 0; i < 32; i++) {
      cellsA.push(el('rect', {x: GA, y: GY + i * RP, width: 60, height: 7, fill: i ? SOFT2 : COL.cobalt}, g3));
      cellsB.push(el('rect', {x: GB, y: GY + i * RP, width: 60, height: 7, fill: i ? SOFT2 : COL.cobalt}, g3));
      if (i === 0 || i === 31) label(g3, GA - 6, GY + i * RP + 7, String(i + 1), 12, 't-mono t-quiet', {'text-anchor': 'end'});
    }
    function fillGroups(u) { for (var q = 0; q < 32; q++) { var v = part(u, q / 40, q / 40 + 0.2); cellsA[q].setAttribute('opacity', v); cellsB[q].setAttribute('opacity', v); } }
    if (k >= 3) { if (animate && k === 3) tween(box, 900, fillGroups); else fillGroups(1); }

    // scene C, step 4: every query against every candidate; the diagonal is the partner, the rest are the negatives
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500), MX = 1030, MY = 104, CP = 9, CS = 8, Mw = 32 * CP, cells = [];
    label(g4, MX, 80, 'Cosine similarity', 20, '', {'font-weight': 500});
    label(g4, MX, 98, 'Candidate chunks B', 15, 't-mono t-quiet');
    label(g4, MX - 12, MY + Mw / 2, 'Query chunks A', 15, 't-mono t-quiet', {'text-anchor': 'middle', transform: 'rotate(-90 ' + (MX - 12) + ' ' + (MY + Mw / 2) + ')'});
    for (i = 0; i < 32; i++) { cells.push([]); for (j = 0; j < 32; j++) cells[i].push(el('rect', {x: MX + j * CP, y: MY + i * CP, width: CS, height: CS, fill: COL.wash}, g4)); }
    function paint(v, rowsShown) {   // v: 0 = the base checkpoint, 1 = the adapted one (measured; interpolated while switching); rowsShown: rows painted so far
      for (var a = 0; a < 32; a++) for (var b = 0; b < 32; b++) {
        var on = rowsShown == null || a < rowsShown;
        cells[a][b].setAttribute('fill', !on ? COL.wash : ex ? cosFill(mix(ex.before.cos[a][b], ex.after.cos[a][b], v)) : (a === b ? COL.cobalt : COL.wash));
      }
    }
    label(g4, 1340, 126, 'Diagonal: same article', 16, 't-mono');
    label(g4, 1340, 154, 'Negatives: other articles', 16, 't-mono t-quiet'); label(g4, 1340, 176, 'in the batch', 16, 't-mono t-quiet');
    rich(g4, 1340, 232, [['S = Z'], ['A', 'sub'], ['Z'], ['B', 'sub'], ['T', 'sup'], [' / τ']], {'font-size': 26, class: 't-serif'});
    label(g4, 1340, 264, 'τ = ' + tau, 20, 't-mono');
    var defs = el('defs', {}, s), grad = el('linearGradient', {id: 'ft-scale', x1: 0, x2: 1, y1: 0, y2: 0}, defs);
    el('stop', {offset: '0%', 'stop-color': COL.red}, grad); el('stop', {offset: '50%', 'stop-color': '#fff'}, grad); el('stop', {offset: '100%', 'stop-color': COL.cobalt}, grad);
    el('rect', {x: 1340, y: 300, width: 200, height: 12, fill: 'url(#ft-scale)', stroke: COL.rule}, g4);
    label(g4, 1340, 330, '−1', 14, 't-mono t-quiet'); label(g4, 1440, 330, '0', 14, 't-mono t-quiet', {'text-anchor': 'middle'}); label(g4, 1540, 330, '+1', 14, 't-mono t-quiet', {'text-anchor': 'end'});
    label(g4, 1340, 362, ex ? 'New example batch · 32 pairs' : 'Schematic', 14, 't-mono t-quiet');
    var state = label(g4, MX, MY + Mw + 26, ex ? 'Before fine-tuning' : '', 18, 't-mono', {'font-weight': 500});

    // scene D, step 5: the first row unfolded: its cosines, divided by τ, as probabilities whose widths sum to one; the loss of the partner's share
    var g5 = reveal(box, el('g', {}, s), k, 5, animate, 500), SX = 700, SW = 640, yC = 444, yP = 486, cosRow = [], prob = [];
    el('rect', {x: MX - 2, y: MY - 2, width: Mw + 3, height: CS + 4, fill: 'none', stroke: COL.ink, 'stroke-width': 2}, g5);
    for (j = 0; j < 32; j++) cosRow.push(el('rect', {x: SX + j * 20, y: yC, width: 19, height: 16, fill: COL.wash}, g5));
    label(g5, SX, yP - 8, 'Match probability', 17, 't-mono t-quiet');
    arrow(g5, SX + SW + 16, yC + 1, SX + SW + 16, yP - 3, {stroke: COL.ink}); label(g5, SX + SW + 8, yP - 10, 'τ = ' + tau, 14, 't-mono t-quiet', {'text-anchor': 'end'});
    for (j = 0; j < 32; j++) prob.push(el('rect', {x: SX, y: yP, width: 0, height: 36, fill: j ? COL.wash : COL.cobalt, stroke: j ? COL.rule : 'none', 'stroke-width': 0.8}, g5));
    var lSame = label(g5, SX, yP + 60, 'Same article', 17, 't-mono t-cobalt', {opacity: 0});
    var lOther = label(g5, SX + SW, yP + 60, 'Other articles', 17, 't-mono t-quiet', {'text-anchor': 'end', opacity: 0});
    var LX = lin(0, 1, 1380, 1580), LY = lin(0, 4, 560, 444), dc = '';
    for (var q = 0; q <= 60; q++) { var pp = 0.018 + (1 - 0.018) * q / 60; dc += (q ? 'L' : 'M') + LX(pp).toFixed(1) + ' ' + LY(Math.min(4, -Math.log(pp))).toFixed(1); }
    el('path', {d: dc, fill: 'none', stroke: COL.mist, 'stroke-width': 2}, g5);
    el('path', {d: 'M1380 444 V 560 H 1580', fill: 'none', class: 'ln-axis'}, g5);
    var guide = el('path', {d: '', fill: 'none', stroke: COL.cobalt, 'stroke-width': 1, 'stroke-dasharray': '4 4', opacity: 0}, g5);
    var pt = el('circle', {cx: 0, cy: 0, r: 6, fill: COL.cobalt, opacity: 0}, g5);
    label(g5, 1580, 596, 'Loss = −log P(same article)', 20, '', {'text-anchor': 'end', 'font-weight': 500});
    function strip(v, u) {   // v: base -> adapted; u: how far the widths have unfolded
      var x = SX, p0 = null;
      for (var b = 0; b < 32; b++) {
        var cv = ex ? mix(ex.before.cos[0][b], ex.after.cos[0][b], v) : null;
        cosRow[b].setAttribute('fill', cv == null ? (b ? COL.wash : COL.cobalt) : cosFill(cv));
        var pb = ex ? mix(ex.before.p_row[0][b], ex.after.p_row[0][b], v) : 1 / 32, w = pb * SW * u;
        if (!b) p0 = pb;
        prob[b].setAttribute('x', x); prob[b].setAttribute('width', Math.max(b ? 0 : 1, w - (w > 2 ? 1 : 0))); x += w;   // the partner's share stays visible, at least 1 px
      }
      lSame.setAttribute('opacity', u >= 1 ? 1 : 0); lOther.setAttribute('opacity', u >= 1 ? 1 : 0);
      if (ex) {
        var L = Math.min(4, -Math.log(p0));
        pt.setAttribute('cx', LX(p0)); pt.setAttribute('cy', LY(L)); guide.setAttribute('d', 'M' + LX(p0) + ' 560 V ' + LY(L) + ' H 1380');
        pt.setAttribute('opacity', u >= 1 ? 1 : 0); guide.setAttribute('opacity', u >= 1 ? 1 : 0);
      }
    }

    // scene D, step 6: the adapted checkpoint on the same batch; a column for the other direction; the symmetric loss
    var g6 = reveal(box, el('g', {}, s), k, 6, animate, 500);
    var colHL = el('rect', {x: MX - 2, y: MY - 2, width: CS + 4, height: Mw + 3, fill: 'none', stroke: COL.red, 'stroke-width': 2}, g6);
    rich(g6, SX, 626, [['L = ½[CE(S, y) + CE(S'], ['T', 'sup'], [', y)]']], {'font-size': 26, class: 't-serif'});
    label(g6, 1090, 626, 'Symmetric InfoNCE', 19, 't-mono t-cobalt');
    label(g6, 1316, 626, 'Targets: diagonal matches', 17, 't-mono t-quiet');
    if (k === 4) { if (animate) tween(box, 800, function (u) { paint(0, Math.ceil(u * 32)); }); else paint(0); }
    if (k === 5) { paint(0); if (animate) tween(box, 800, function (u) { strip(0, ease(u)); }); else strip(0, 1); }
    if (k >= 6) {
      if (ex) state.textContent = 'After fine-tuning';
      if (animate && k === 6) {
        paint(0); strip(0, 1); colHL.setAttribute('opacity', 0);
        tween(box, 1100, function (u) { var v = ease(part(u, 0, 0.7)); paint(v); strip(v, 1); colHL.setAttribute('opacity', u > 0.8 ? 1 : 0); });
      } else { paint(1); strip(1, 1); }
    }

    // scene E, step 7: the loss traced back through every layer and the embeddings; one update (schematic); what changed
    var g7 = reveal(box, el('g', {}, s), k, 7, animate, 400);
    var sweep = el('rect', {x: 1580, y: ENC.y - 10, width: 36, height: ENC.h + 20, fill: 'rgba(44,75,239,.12)', opacity: 0}, g7);
    var upd = [label(g7, ENC.x, 548, 'Weight update: schematic', 19, 't-mono t-cobalt', {opacity: 0}),
               label(g7, ENC.x, 578, 'AdamW', 22, '', {'font-weight': 500, opacity: 0}),
               label(g7, ENC.x, 608, 'Learning rate: 2×10⁻⁵', 19, 't-mono', {opacity: 0})];
    var st = ['Encoder architecture unchanged', 'All encoder parameters trainable', 'No added head'].map(function (t, q) { return label(g7, 0, 440 + q * 34, t, 19, '', {'font-weight': 500, opacity: 0}); });
    function trace(u) {
      var a = part(u, 0, 0.5), b = part(u, 0.5, 0.75), c2 = part(u, 0.75, 1), x = 1580 - (1580 - 220) * a, pulse = Math.sin(Math.PI * b);
      sweep.setAttribute('opacity', a > 0 && a < 1 ? 1 : 0); sweep.setAttribute('x', x);
      E.slabs.forEach(function (sl, q) { var lit = x <= slabX(q) + ENC.slab; sl.setAttribute('fill', lit ? (pulse > 0.5 ? COL.cobalt : SOFT2) : SLAB); });
      chunks.forEach(function (c) { c.emb.forEach(function (r, q) { r.setAttribute('fill', a >= 1 ? (pulse > 0.5 ? COL.cobalt : SOFT2) : (q % 2 ? SLAB : '#C3C8E3')); }); });
      upd.forEach(function (t, q) { t.setAttribute('opacity', b > q * 0.3 ? 1 : 0); });
      st.forEach(function (t, q) { t.setAttribute('opacity', c2 > q / 3 + 0.01 ? 1 : 0); });
    }
    if (k >= 7) { if (animate) tween(box, 1800, trace); else trace(1); }
  });

  // ------------------------------------------------------------------ B18 the paper's protocol and ours, then the dates
  C.protocols = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, F = D().facts, nb = F.nolbert, sp = F.splits, ad = F.adaptation, T = D().training, i;
    var HR = hatch(s, 'pr-hatch', COL.red);
    // scene A, step 1: one pretrained model above two branches
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 500);
    rrect(g1, 640, 12, 300, 46, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.5});
    label(g1, 790, 42, 'NoLBERT base', 24, '', {'text-anchor': 'middle', 'font-weight': 600});
    label(g1, 790, 84, 'Training: ' + nb.window[0] + '–' + nb.window[1], 18, 't-mono t-quiet', {'text-anchor': 'middle'});
    label(g1, 790, 108, 'Validation: ' + nb.val, 18, 't-mono t-quiet', {'text-anchor': 'middle'});
    var b1 = el('path', {d: 'M660 58 C 600 100, 300 110, 140 140', fill: 'none', stroke: COL.mist, 'stroke-width': 2}, g1);
    var b2 = el('path', {d: 'M920 58 C 930 100, 910 120, 900 140', fill: 'none', stroke: COL.mist, 'stroke-width': 2}, g1);
    var hdr1 = label(g1, 40, 168, 'NoLBERT paper', 26, '', {'font-weight': 600, opacity: 0});
    var hdr2 = label(g1, 820, 168, 'Our fine-tuning', 26, 't-cobalt', {'font-weight': 600, opacity: 0});
    function branches(u) { drawOn(b1, u); drawOn(b2, u); hdr1.setAttribute('opacity', u >= 1 ? 1 : 0); hdr2.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= 1) { if (animate && k === 1) tween(box, 1000, function (u) { branches(ease(u)); }); else branches(1); }

    // scene B, step 2: the two fine-tuning paths, in parallel
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 500);
    var paper = ['Patent text: 1997–2021', '[CLS] representations', 'MLP classifier', 'Same/different patent', 'Cross-entropy'];
    var ours = ['FNSPID news: Jan 2015–Aug 2020', 'Last-layer mean', 'L2 normalization', '32-candidate retrieval', 'Symmetric InfoNCE'];
    var Y0 = 196, PITCH = 44, NW = 320;
    function chain(x, items, tint, into) {
      return items.map(function (t, q) {
        var y = Y0 + q * PITCH, ng = el('g', {opacity: 0}, into);
        rrect(ng, x, y, NW, 34, {fill: tint ? SOFT : COL.wash, stroke: tint ? COL.cobalt : COL.rule, 'stroke-width': 1.2, rx: 5});
        label(ng, x + 14, y + 23, t, 18);
        if (q) arrow(ng, x + 30, y - 10, x + 30, y - 1, {stroke: COL.mist, head: 7});
        return ng;
      });
    }
    var nodesP = chain(40, paper, false, g2), nodesO = chain(820, ours, true, g2), yS = Y0 + 3 * PITCH + 17;
    var outP = el('g', {opacity: 0}, g2);   // the paper's path ends in two classes
    arrow(outP, 362, yS, 380, yS, {head: 7});
    el('circle', {cx: 392, cy: yS, r: 8, fill: COL.ink}, outP); el('circle', {cx: 418, cy: yS, r: 8, fill: '#fff', stroke: COL.ink, 'stroke-width': 2}, outP);
    var outO = el('g', {opacity: 0}, g2);   // ours in the candidate matrix of 3.1
    arrow(outO, 1142, yS, 1156, yS, {head: 7});
    for (i = 0; i < 8; i++) for (var j2 = 0; j2 < 8; j2++) el('rect', {x: 1160 + j2 * 7, y: yS - 28 + i * 7, width: 6, height: 6, fill: i === j2 ? COL.cobalt : SOFT}, outO);
    function grow(u) {
      nodesP.forEach(function (n, q) { var v = part(u, q * 0.15, q * 0.15 + 0.2); n.setAttribute('opacity', v); nodesO[q].setAttribute('opacity', v); });
      outP.setAttribute('opacity', u >= 1 ? 1 : 0); outO.setAttribute('opacity', u >= 1 ? 1 : 0);
    }
    if (k >= 2) { if (animate && k === 2) tween(box, 1800, grow); else grow(1); }

    // scene C, step 3: the paper's yearly 70/30 splits; our pairs sent to two destinations, only one of them the update
    var g3 = reveal(box, el('g', {}, s), k, 3, animate, 500), bars = [];
    label(g3, 430, 206, 'Separate fine-tuning each year', 17, 't-mono t-quiet');
    for (i = 0; i < 25; i++) {
      var bg = el('g', {opacity: 0}, g3), bx = 430 + i * 13;
      el('rect', {x: bx, y: 222, width: 10, height: 84, fill: SOFT2}, bg); el('rect', {x: bx, y: 306, width: 10, height: 36, fill: '#CFCFCC'}, bg);
      bars.push(bg);
    }
    label(g3, 430, 368, '70% training', 17, 't-mono t-cobalt'); label(g3, 580, 368, '30% test', 17, 't-mono t-quiet');
    var oT = el('g', {opacity: 0}, g3), oH = el('g', {opacity: 0}, g3);
    rrect(oT, 1240, 204, 250, 34, {fill: SOFT, stroke: COL.cobalt, 'stroke-width': 1.2, rx: 5});
    label(oT, 1254, 227, ad.pairs.toLocaleString('en-US') + ' training pairs', 18);
    arrow(oT, 1265, 240, 1265, 254, {stroke: COL.cobalt, head: 7}); label(oT, 1240, 274, 'Weight update: schematic', 16, 't-mono t-cobalt');
    rrect(oH, 1240, 300, 250, 34, {fill: COL.wash, stroke: COL.rule, 'stroke-width': 1.2, rx: 5});
    label(oH, 1254, 323, ad.held_out.toLocaleString('en-US') + ' held-out pairs', 18);
    arrow(oH, 1265, 336, 1265, 350, {stroke: COL.mist, head: 7}); label(oH, 1240, 370, 'Top-1 retrieval: A→B', 16, 't-mono t-quiet');
    var l1 = el('line', {x1: 1140, y1: 213, x2: 1236, y2: 221, stroke: COL.mist, 'stroke-width': 1.5, opacity: 0}, g3);
    var l2 = el('line', {x1: 1140, y1: 213, x2: 1236, y2: 317, stroke: COL.mist, 'stroke-width': 1.5, opacity: 0}, g3);
    function split(u) {
      bars.forEach(function (b, q) { b.setAttribute('opacity', part(u, q / 50, q / 50 + 0.3)); });
      var v = part(u, 0.3, 0.6), w = part(u, 0.6, 0.9);
      l1.setAttribute('opacity', v); oT.setAttribute('opacity', v); l2.setAttribute('opacity', w); oH.setAttribute('opacity', w);
    }
    if (k >= 3) { if (animate && k === 3) tween(box, 1800, split); else split(1); }

    // scene D, step 4: the encoder's text and the forecasting periods on one axis; the eight overlapping months; the proposed cutoff
    var g4 = reveal(box, el('g', {}, s), k, 4, animate, 500);
    var X = function (d) { return lin(day('2012-01-01'), day('2025-01-01'), 300, 1540)(day(d)); }, yF = 482, yR = 556;
    [2012, 2014, 2016, 2018, 2020, 2022, 2024].forEach(function (y) {
      el('line', {x1: X(y + '-01-01'), x2: X(y + '-01-01'), y1: 462, y2: 596, class: 'ln-rule'}, g4);
      label(g4, X(y + '-01-01'), 616, String(y), 16, 't-mono t-quiet', {'text-anchor': 'middle'});
    });
    label(g4, 0, yF + 21, 'Encoder fine-tuning', 22);
    el('rect', {x: X(ad.corpus_from), y: yF, width: X(ad.before) - X(ad.corpus_from), height: 30, fill: HR, stroke: COL.red, 'stroke-width': 2}, g4);
    label(g4, X(ad.before) + 12, yF + 21, 'Jan 2015–Dec 2019', 18, 't-mono t-red');   // wording asked for by the team on 9 Oct; the bar still ends at the run's cutoff
    label(g4, 0, yR + 21, 'Return forecasting', 22);
    [['Training', sp.train, 0.10], ['Validation', sp.valid, 0.18], ['Test', sp.test, 0.28]].forEach(function (p) {
      var x0 = X(p[1][0]), x1 = X(p[1][1]);
      el('rect', {x: x0, y: yR, width: x1 - x0, height: 30, fill: 'rgba(22,22,22,' + p[2] + ')'}, g4);
      label(g4, (x0 + x1) / 2, yR - 10, p[0] + ': ' + p[1][0].slice(0, 4) + '–' + p[1][1].slice(0, 4), 17, 't-mono t-quiet', {'text-anchor': 'middle', stroke: '#fff', 'stroke-width': 5, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
    });
    var ov = el('rect', {x: X(sp.valid[0]), y: 468, width: X(ad.before) - X(sp.valid[0]), height: 124, fill: REDSOFT, stroke: COL.red, 'stroke-width': 1.5, opacity: 0}, g4);
    var cur = el('line', {x1: 0, x2: 0, y1: 462, y2: 596, stroke: COL.ink, 'stroke-width': 2, opacity: 0}, g4);
    var cap = label(g4, 300, 650, 'Fine-tuning includes eight months of validation text; no test-period text.', 22, '', {opacity: 0});
    var xc = X('2019-12-31');
    el('line', {x1: xc, x2: xc, y1: 458, y2: 596, stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '6 5'}, g4);
    label(g4, xc - 10, 452, 'Proposed cutoff: Dec 2019', 17, 't-mono t-cobalt', {'text-anchor': 'end'});
    function sweepDates(u) {
      var a = part(u, 0, 0.45), hold = part(u, 0.45, 0.75), b = part(u, 0.75, 1), x;
      if (a < 1) x = mix(X(sp.train[0]), X(sp.valid[0]), a);
      else if (b <= 0) x = mix(X(sp.valid[0]), X(ad.before), hold);
      else x = mix(X(ad.before), X(sp.test[1]), b);
      cur.setAttribute('x1', x); cur.setAttribute('x2', x); cur.setAttribute('opacity', u > 0 && u < 1 ? 1 : 0);
      ov.setAttribute('opacity', hold > 0 ? 1 : 0); cap.setAttribute('opacity', hold >= 1 ? 1 : 0);
    }
    if (k >= 4) { if (animate && k === 4) tween(box, 3400, sweepDates); else sweepDates(1); }
  });

  // ------------------------------------------------------------------ 3.3 the recorded loss, the matching decisions of two checkpoints, the held-out evaluation
  C.results = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, T = D().training, ad = D().facts.adaptation, sim = SIM(), ex = sim && sim.batches.heldout_example, log = T.log;
    // scene A, step 1: every logged loss observation, in optimizer-step order
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 400);
    label(g1, 0, 30, 'Logged training loss', 22, '', {'font-weight': 600});
    var X = lin(0, T.steps, 70, 540), Y = lin(0, 6, 380, 60), u32 = Math.log(T.batch);
    [0, 2, 4, 6].forEach(function (v) { el('line', {x1: 70, x2: 540, y1: Y(v), y2: Y(v), class: 'ln-rule'}, g1); label(g1, 58, Y(v) + 6, String(v), 16, 't-mono t-quiet', {'text-anchor': 'end'}); });
    [0, 2000, 4000, 6000].forEach(function (v) { label(g1, X(v), 404, v.toLocaleString('en-US'), 16, 't-mono t-quiet', {'text-anchor': 'middle'}); });
    el('line', {x1: 70, x2: 540, y1: Y(0), y2: Y(0), class: 'ln-axis', 'stroke-width': 1.5}, g1);
    label(g1, 540, 432, 'Optimizer step', 17, 't-mono t-quiet', {'text-anchor': 'end'});
    label(g1, 16, 220, 'InfoNCE loss', 17, 't-mono t-quiet', {'text-anchor': 'middle', transform: 'rotate(-90 16 220)'});
    el('line', {x1: 70, x2: 540, y1: Y(u32), y2: Y(u32), stroke: COL.mist, 'stroke-width': 1.5, 'stroke-dasharray': '6 6'}, g1);
    label(g1, 540, Y(u32) - 8, 'Uniform probabilities: log(32)', 15, 't-mono t-quiet', {'text-anchor': 'end'});
    var d = ''; log.forEach(function (r, q) { d += (q ? 'L' : 'M') + X(r[0]).toFixed(1) + ' ' + Y(r[1]).toFixed(1); });
    var path = el('path', {d: d, fill: 'none', stroke: COL.cobalt, 'stroke-width': 2}, g1), last = log[log.length - 1];
    var ann = el('g', {opacity: 0}, g1);
    el('circle', {cx: X(last[0]), cy: Y(last[1]), r: 5, fill: COL.cobalt}, ann);
    el('line', {x1: X(last[0]), x2: X(last[0]), y1: Y(last[1]) - 8, y2: Y(last[1]) - 46, stroke: COL.cobalt, 'stroke-width': 1}, ann);
    label(ann, X(last[0]), Y(last[1]) - 54, 'Step ' + last[0].toLocaleString('en-US') + ': ' + last[1].toFixed(3), 17, 't-mono', {'text-anchor': 'end', 'font-weight': 500, stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
    var sums = [T.steps.toLocaleString('en-US') + ' optimizer steps', T.budget_min + ' minutes', T.pairs_seen.toLocaleString('en-US') + ' pair presentations', 'Less than one epoch']
      .map(function (t, q) { return label(g1, 0, 486 + q * 32, t, 20, 't-mono', {opacity: 0}); });
    function draw(u) { drawOn(path, part(u, 0, 0.8)); ann.setAttribute('opacity', u >= 0.8 ? 1 : 0); sums.forEach(function (t, q) { t.setAttribute('opacity', u >= 0.8 + q * 0.05 ? 1 : 0); }); }
    if (k >= 1) { if (animate && k === 1) tween(box, 2600, draw); else draw(1); }

    // scene B, steps 2 and 3: each query joined to its highest-scoring candidate, under the base and the adapted checkpoint
    var aggStep = ex ? 4 : 2;
    function matching(into, x0, which, title) {
      var rec = ex[which], qx = x0 + 20, cx = x0 + 160, y0 = 72, p = 9, lines = [], q;
      label(into, x0, 30, title, 22, '', {'font-weight': 600});
      label(into, qx, 56, 'Query chunks A', 15, 't-mono t-quiet', {'text-anchor': 'middle'});
      label(into, cx, 376, 'Candidate chunks B', 15, 't-mono t-quiet', {'text-anchor': 'middle'});
      for (q = 0; q < 32; q++) {
        var ok = rec.choice_row[q] === q;
        lines.push(el('line', {x1: qx + 5, y1: y0 + q * p, x2: cx - 5, y2: y0 + rec.choice_row[q] * p, stroke: ok ? COL.cobalt : '#BDBDBA', 'stroke-width': ok ? 1.6 : 1.2, opacity: 0}, into));
      }
      for (q = 0; q < 32; q++) { el('circle', {cx: qx, cy: y0 + q * p, r: 3, fill: COL.ink}, into); el('circle', {cx: cx, cy: y0 + q * p, r: 3, fill: COL.ink}, into); }
      var cnt = label(into, x0, 410, 'Correct: ' + rec.correct_row + '/32', 20, 't-mono', {'font-weight': 500, opacity: 0});
      return function (u) { lines.forEach(function (l, q) { l.setAttribute('opacity', part(u, q / 40, q / 40 + 0.2)); }); cnt.setAttribute('opacity', u >= 1 ? 1 : 0); };
    }
    if (ex) {
      var g2 = reveal(box, el('g', {}, s), k, 2, animate, 400), g3 = reveal(box, el('g', {}, s), k, 3, animate, 400);
      el('line', {x1: 620, x2: 650, y1: 446, y2: 446, stroke: COL.cobalt, 'stroke-width': 2}, g2); label(g2, 660, 451, 'Correct match', 16, 't-mono t-quiet');
      el('line', {x1: 820, x2: 850, y1: 446, y2: 446, stroke: '#BDBDBA', 'stroke-width': 2}, g2); label(g2, 860, 451, 'Incorrect match', 16, 't-mono t-quiet');
      label(g2, 620, 486, 'New example batch · 32 pairs', 17, 't-mono t-quiet');
      var m1 = matching(g2, 620, 'before', 'Before fine-tuning'), m2 = matching(g3, 900, 'after', 'After fine-tuning');
      if (k >= 2) { if (animate && k === 2) tween(box, 1600, m1); else m1(1); }
      if (k >= 3) { if (animate && k === 3) tween(box, 1600, m2); else m2(1); }
    }

    // scene C: the recorded evaluation of the whole held-out set, on one percentage scale; only its two endpoints exist
    var g4 = reveal(box, el('g', {}, s), k, aggStep, animate, 500), AX = lin(0, 1, 1180, 1560), ch = 1 / T.batch;
    label(g4, 1180, 30, 'Original held-out evaluation', 22, '', {'font-weight': 600});
    label(g4, 1180, 60, ad.held_out.toLocaleString('en-US') + ' pairs · ' + T.batch + ' candidates', 17, 't-mono t-quiet');
    label(g4, 1180, 84, 'Top-1 retrieval: A→B', 17, 't-mono t-quiet');
    [0, 0.5, 1].forEach(function (v) { el('line', {x1: AX(v), x2: AX(v), y1: 118, y2: 256, class: 'ln-rule'}, g4); label(g4, AX(v), 278, U.pct(v), 15, 't-mono t-quiet', {'text-anchor': 'middle'}); });
    var bars = [['Before: ' + (T.before * 100).toFixed(2) + '%', T.before, '#BDBDBA', 130], ['After: ' + (T.after * 100).toFixed(2) + '%', T.after, COL.cobalt, 206]].map(function (r) {
      label(g4, 1180, r[3] - 8, r[0], 20, 't-mono', {'font-weight': 500});
      return {r: el('rect', {x: 1180, y: r[3], width: 0, height: 40, fill: r[2]}, g4), v: r[1]};
    });
    [[128, 172], [204, 248]].forEach(function (yy) { el('line', {x1: AX(ch), x2: AX(ch), y1: yy[0], y2: yy[1], stroke: COL.ink, 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}, g4); });
    label(g4, AX(ch) + 8, 304, 'Chance: ' + (ch * 100).toFixed(3) + '%', 16, 't-mono t-quiet');
    var fin = label(g4, 0, 640, 'Retrieval improved. Return forecasting gains are not established.', 24, '', {'font-weight': 500, opacity: 0});
    function agg(u) { bars.forEach(function (b) { b.r.setAttribute('width', (AX(b.v) - AX(0)) * u); }); fin.setAttribute('opacity', u >= 1 ? 1 : 0); }
    if (k >= aggStep) { if (animate && k === aggStep) tween(box, 1200, function (u) { agg(ease(u)); }); else agg(1); }
  });

  // ------------------------------------------------------------------ B11 the pooling probe of the adapted encoder
  C.pooling = stepped(function (box) {
    var g = surface(box), s = g.s, P = D().pooling;
    label(s, 0, 30, 'Pooling of the adapted model · ' + P.docs.toLocaleString('en-US') + ' documents · ' + P.k + ' clusters', 20, 't-mono t-quiet');
    label(s, 900, 92, 'NPMI', 19, 't-mono t-quiet', {'text-anchor': 'end'});
    label(s, 1240, 92, 'AMI vs MiniLM', 19, 't-mono t-quiet', {'text-anchor': 'end'});
    P.rows.forEach(function (r, i) {
      var y = 156 + i * 64, best = i === 0;
      el('line', {x1: 0, x2: 1240, y1: y - 40, y2: y - 40, class: 'ln-rule'}, s);
      label(s, 0, y, r.label, 26, best ? 't-cobalt' : '', {'font-weight': best ? 600 : 400});
      label(s, 900, y, r.npmi.toFixed(3), 24, 't-mono' + (best ? ' t-cobalt' : ''), {'text-anchor': 'end', 'font-weight': best ? 500 : 400});
      label(s, 1240, y, r.ami.toFixed(2), 24, 't-mono', {'text-anchor': 'end'});
    });
    el('line', {x1: 0, x2: 1240, y1: 156 + P.rows.length * 64 - 40, y2: 156 + P.rows.length * 64 - 40, class: 'ln-rule'}, s);
  });
})();
