/* Exhibits, part B: sections 4 and 5, and the backup slides. Same conventions as part A. */
(function () {
  'use strict';
  var U = window.DECKUTIL, C = window.CHARTS, el = U.el, tx = U.tx, surface = U.surface, lin = U.lin, ease = U.ease,
      tween = U.tween, halt = U.halt, reveal = U.reveal, pct = U.pct, signed = U.signed, COL = U.COL;
  var D = function () { return window.DATA; };
  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); box._k = -1; build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; var from = box._k; box._k = k; build(box, k, true, from); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k, false); }
    };
  }
  var FSHADE = {M: 0.38, '2W': 0.68, W: 1};

  // ------------------------------------------------------------------ 4.1 the twelve largest main-effect shapes, then one stock read through them
  var PRICE = {rev_21: '21-day return', rev_5: '5-day return', ivol: 'Idiosyncratic volatility', beta: 'Market beta', mom_12_1: 'Momentum, months 12 to 2',
               vol_21: '21-day volatility', size: 'Market value', max_21: 'Max daily return, 21 days'};
  var SHORT = {rev_21: '21-day ret.', rev_5: '5-day ret.', ivol: 'idio. vol.', beta: 'beta', mom_12_1: 'momentum', vol_21: '21-day vol.', size: 'size', max_21: 'max ret.'};
  function nice(t) {
    var m = /^narr_lda_t(\d+)$/.exec(t.term); if (m) return ['Topic ' + (+m[1]) + ' exposure', t.words || ''];
    m = /^firm_lda_news_t(\d+)$/.exec(t.term); if (m) return ['Firm news, topic ' + (+m[1]), t.words || ''];
    return [PRICE[t.term] || t.term, 'price feature'];
  }
  function shortName(t) { var m = /^narr_lda_t(\d+)$/.exec(t); if (m) return 'topic ' + (+m[1]); m = /^firm_lda_news_t(\d+)$/.exec(t); if (m) return 'firm t' + (+m[1]); return SHORT[t] || t; }
  C.shapes = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, S = D().glassbox.shapes_weekly_bp, ST = D().stock, GB = D().glassbox.models.filter(function (m) { return m.freq === 'W'; })[0];
    tx(s, 0, 34, 'forecast  =  intercept  +  f₁(x₁)  +  f₂(x₂)  +  …  +  f₁₂(x₁, x₂)  +  …', {'font-size': 34, class: 't-serif', 'font-style': 'italic'});
    var lo = Infinity, hi = -Infinity;
    S.forEach(function (t) { t.bins.forEach(function (b) { lo = Math.min(lo, b[2]); hi = Math.max(hi, b[2]); }); });
    tx(s, 0, 70, 'x: Feature rank, −1 to 1 · y: Weekly return contribution, bp, ' + U.signed(lo) + ' to ' + U.signed(hi) + ' · dashed: 0', {'font-size': 18, class: 't-mono t-quiet'});
    var pad = (hi - lo) * 0.15, cw = 240, ch = 64, gx = 24, byTerm = {}, have = ST && ST.available;
    if (have) ST.terms.forEach(function (t) { byTerm[t.term] = t; });
    var marks = el('g', {opacity: 0}, s);
    S.slice(0, 12).forEach(function (t, i) {
      var c = i % 6, r = Math.floor(i / 6), x0 = c * (cw + gx), y0 = 96 + r * 122, nm = nice(t);
      var X = lin(-1, 1, x0, x0 + cw), Y = lin(lo - pad, hi + pad, y0 + 40 + ch, y0 + 40);
      tx(s, x0, y0 + 14, nm[0], {'font-size': 15, class: 't-mono'});
      tx(s, x0, y0 + 31, nm[1].length > 30 ? nm[1].slice(0, 28).replace(/,?\s*$/, '') + '…' : nm[1], {'font-size': 15, class: 't-quiet'});
      el('rect', {x: x0, y: y0 + 40, width: cw, height: ch, fill: 'none', stroke: COL.rule}, s);
      el('line', {x1: x0, x2: x0 + cw, y1: Y(0), y2: Y(0), stroke: COL.mist, 'stroke-dasharray': '4 4'}, s);
      if (c === 0) { var hl = {'font-size': 11, class: 't-mono t-quiet', stroke: '#fff', 'stroke-width': 4, 'stroke-linejoin': 'round', 'paint-order': 'stroke'}; tx(s, x0 + 3, Y(hi) + 11, signed(hi, 1), hl); tx(s, x0 + 3, Y(lo) - 3, signed(lo, 1), hl); }
      if (r === 0) { tx(s, x0, y0 + 40 + ch + 13, '−1', {'font-size': 11, class: 't-mono t-quiet'}); tx(s, x0 + cw, y0 + 40 + ch + 13, '+1', {'text-anchor': 'end', 'font-size': 11, class: 't-mono t-quiet'}); }
      var d = '';
      t.bins.forEach(function (b, j) { d += (j ? 'L' : 'M') + X(b[0]).toFixed(1) + ' ' + Y(b[2]).toFixed(1) + 'L' + X(b[1]).toFixed(1) + ' ' + Y(b[2]).toFixed(1); });
      var p = el('path', {d: d, fill: 'none', stroke: COL.ink, 'stroke-width': 2, 'stroke-dasharray': '2000', 'stroke-dashoffset': k >= 1 ? 0 : 2000}, s);
      if (animate && k === 1) tween(box, 1200, function (u) { p.setAttribute('stroke-dashoffset', 2000 * (1 - ease(u))); });
      var st = byTerm[t.term];
      if (st) {
        el('line', {x1: X(st.x), x2: X(st.x), y1: y0 + 40, y2: y0 + 40 + ch, stroke: COL.cobalt, 'stroke-width': 1, 'stroke-dasharray': '3 3'}, marks);
        el('circle', {cx: X(st.x), cy: Y(st.bp), r: 5, fill: COL.cobalt}, marks);
        tx(marks, X(st.x) + (st.x > 0.3 ? -9 : 9), Y(st.bp) + (st.bp > (lo + hi) / 2 ? 19 : -8), signed(st.bp, 2), {'text-anchor': st.x > 0.3 ? 'end' : 'start', 'font-size': 14, class: 't-mono t-cobalt',
          stroke: '#fff', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round'});
      }
    });
    if (!have) return;
    if (k >= 1) { if (animate && k === 1) tween(box, 500, function (u) { marks.setAttribute('opacity', u); }); else marks.setAttribute('opacity', 1); }

    // one stock: what each of the twelve adds, stacked, then the rest, against its forecast
    var items = S.slice(0, 12).map(function (t) { var st = byTerm[t.term]; return {term: t.term, bp: st ? st.bp : 0, news: /^(narr|firm)_/.test(t.term)}; });
    items.push({term: 'rest', bp: ST.rest_bp, rest: true});
    var cum = 0, segs = items.map(function (it) { it.a = cum; cum += it.bp; it.b = cum; return it; });
    var wf = el('g', {opacity: k >= 2 ? 1 : 0}, s), top = 362, bot = 526;
    var mn = 0, mx = 0;
    segs.forEach(function (q) { mn = Math.min(mn, q.a, q.b); mx = Math.max(mx, q.a, q.b); });
    var YW = lin(mn, mx, bot, top), bw = 88, gapw = 18, xw = function (i) { return 60 + i * (bw + gapw); };
    el('line', {x1: 40, x2: xw(13) + bw + 10, y1: YW(0), y2: YW(0), class: 'ln-axis', 'stroke-width': 1.5}, wf);
    tx(wf, 0, top - 16, 'one stock of the week of ' + ST.date + ', the highest of ' + ST.n + ' forecasts: what each term adds, in basis points, stacked', {'font-size': 17, class: 't-mono'});
    var pairs = (GB.blocks.filter(function (b) { return b.block === 'mixed pair'; })[0] || {}).terms;
    var legend = tx(wf, 0, bot + 44, 'blue: narrative terms · black: price terms · rest: the intercept, the other ' + (GB.main_effects - 12) + ' main effects and the ' + pairs + ' pairwise terms', {'font-size': 15, class: 't-mono t-quiet'});
    var bars = segs.map(function (q, i) {
      var y1 = YW(Math.max(q.a, q.b)), h = Math.abs(YW(q.a) - YW(q.b));
      var r = el('rect', {x: xw(i), y: y1, width: bw, height: h, fill: q.rest ? '#C9C9C6' : q.news ? COL.cobalt : COL.ink}, wf);
      if (i < segs.length - 1) el('line', {x1: xw(i) + bw, x2: xw(i + 1), y1: YW(q.b), y2: YW(q.b), stroke: COL.mist, 'stroke-width': 1}, wf);
      tx(wf, xw(i) + bw / 2, bot + 18, q.rest ? 'rest' : shortName(q.term), {'text-anchor': 'middle', 'font-size': 13, class: 't-mono t-quiet'});
      var lab = tx(wf, xw(i) + bw / 2, q.b >= q.a ? y1 - 5 : y1 + h + 14, signed(q.bp, 2), {'text-anchor': 'middle', 'font-size': 13, class: 't-mono'});
      return {r: r, y1: y1, h: h, up: q.b >= q.a, base: YW(q.a), lab: lab, news: q.news};
    });
    var total = el('g', {}, wf);
    el('rect', {x: xw(13), y: YW(Math.max(0, ST.forecast_bp)), width: bw, height: Math.abs(YW(0) - YW(ST.forecast_bp)), fill: 'none', stroke: COL.ink, 'stroke-width': 2}, total);
    tx(total, xw(13) + bw / 2, YW(Math.max(0, ST.forecast_bp)) - 6, 'forecast ' + signed(ST.forecast_bp, 2) + ' bp', {'text-anchor': 'middle', 'font-size': 14, class: 't-mono', 'font-weight': 500});
    tx(wf, xw(13) + bw / 2, bot + 18, 'forecast', {'text-anchor': 'middle', 'font-size': 13, class: 't-mono t-quiet'});
    function build(U) {
      bars.forEach(function (b, i) {
        var u = Math.max(0, Math.min(1, U * (bars.length + 1) - i)), h = b.h * u;
        b.r.setAttribute('height', h); b.r.setAttribute('y', b.up ? b.base - h : b.base); b.lab.setAttribute('opacity', u >= 1 ? 1 : 0);
      });
      total.setAttribute('opacity', U >= 1 ? 1 : 0);
    }
    if (k >= 2) { if (animate && k === 2) tween(box, 2200, build); else build(1); }

    // without the narrative terms: their bars fade, a second total appears, with the block deletion measured on the whole test
    var newsBp = 0; segs.forEach(function (q) { if (q.news) newsBp += q.bp; });
    var without = ST.forecast_bp - newsBp, narr = GB.blocks.filter(function (b) { return b.block === 'narr'; })[0];
    var del = el('g', {opacity: k >= 3 ? 1 : 0}, wf);
    el('rect', {x: xw(13) + 4, y: YW(Math.max(0, without)), width: bw - 8, height: Math.abs(YW(0) - YW(without)), fill: 'rgba(44,75,239,.18)', stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '5 4'}, del);
    tx(del, xw(13) + bw / 2, YW(Math.max(0, without)) + (without >= ST.forecast_bp ? -24 : 20) - (without >= ST.forecast_bp ? 0 : 0), 'without ' + signed(without, 2), {'text-anchor': 'middle', 'font-size': 13, class: 't-mono t-cobalt',
      stroke: '#fff', 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round'});
    var delLine = tx(del, 0, bot + 44, 'this stock without its narrative terms: ' + signed(ST.forecast_bp, 2) + ' → ' + signed(without, 2) + ' bp · deleting the whole narrative block from the model, all test weeks: rank IC ' + GB.ic.toFixed(4) + ' → ' + (GB.ic - narr.d_ic).toFixed(4) + ', within noise', {'font-size': 15, class: 't-mono'});
    function erase(u) { bars.forEach(function (b) { if (b.news) { b.r.setAttribute('fill-opacity', 1 - 0.8 * u); b.lab.setAttribute('opacity', 1 - u); } }); legend.setAttribute('opacity', 1 - u); del.setAttribute('opacity', u); }
    if (k >= 3) { if (animate && k === 3) tween(box, 900, function (u) { erase(ease(u)); }); else erase(1); }
  });

  // ------------------------------------------------------------------ 4.2 delete one block at a time
  C.blocks = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, M = D().glassbox.models, order = ['narr', 'price', 'firm', 'mixed pair'];
    var shownAt = {narr: 1, price: 2, firm: 3, 'mixed pair': 4};
    var XI = lin(0, 0.8, 330, 600), X = lin(-0.013, 0.013, 720, 1360), y0 = 70, gh = 118;
    tx(s, 330, 22, 'Share of total importance', {'font-size': 19, class: 't-mono t-quiet'});
    tx(s, 720, 22, 'IC higher after removal', {'font-size': 19, class: 't-mono t-quiet'});
    tx(s, 1360, 22, 'IC lower after removal', {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
    tx(s, 1040, 50, 'Δ test IC = full model − model after removal', {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'});
    [-0.01, -0.005, 0, 0.005, 0.01].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: y0 - 6, y2: y0 + 4 * gh - 20, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, s);
      tx(s, X(v), y0 + 4 * gh + 8, v ? signed(v, 3) : '0', {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'});
    });
    order.forEach(function (bk, bi) {
      var y = y0 + bi * gh, first = M[0].blocks.filter(function (b) { return b.block === bk; })[0];
      tx(s, 0, y + 28, first.label, {'font-size': 26, 'font-weight': 600, fill: bk === 'price' ? COL.cobalt : COL.ink});
      if (bk === 'mixed pair') tx(s, 0, y + 58, '29 of 30 are price × firm news', {'font-size': 17, class: 't-mono t-quiet'});
      var active = k === shownAt[bk], shown = k >= shownAt[bk];
      M.forEach(function (m, mi) {
        var b = m.blocks.filter(function (q) { return q.block === bk; })[0], yy = y + mi * 32;
        var imp = el('rect', {x: XI(0), y: yy, width: XI(b.share) - XI(0), height: 24, fill: '#9D9D9A', 'fill-opacity': FSHADE[m.freq]}, s);
        tx(s, XI(b.share) + 8, yy + 19, pct(b.share, b.share < 0.1 ? 1 : 0), {'font-size': 17, class: 't-mono t-quiet'});
        if (active && animate) imp.setAttribute('fill-opacity', 0.12);
        var v = b.d_ic, w = Math.abs(X(v) - X(0)), bar = el('rect', {x: v >= 0 ? X(0) : X(0) - w, y: yy, width: shown ? w : 0, height: 24,
          fill: bk === 'price' ? COL.cobalt : COL.ink, 'fill-opacity': FSHADE[m.freq]}, s);
        tx(s, v >= 0 ? X(v) + 8 : X(v) - 8, yy + 19, shown ? signed(v, 4) : '', {'text-anchor': v >= 0 ? 'start' : 'end', 'font-size': 17, class: 't-mono t-quiet'});
        if (shown && animate && active) { bar.setAttribute('width', 0); tween(box, 700, function (u) { var ww = w * ease(u); bar.setAttribute('width', ww); if (v < 0) bar.setAttribute('x', X(0) - ww); }); }
      });
    });
    tx(s, 0, y0 + 4 * gh + 42, 'One block removed at a time; no refitting. Bars light to dark — Monthly: NoLBERT · Biweekly: FinBERT · Weekly: LDA', {'font-size': 18, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 4.3 forecast to portfolio: the same forecast ranks under the plain rule and under the buffered rule, side by side
  C.portfolio = {
    draw: function (box, k, play) {
      halt(box);
      var g = surface(box), s = g.s, P = D().portfolio, T = D().turnover;
      tx(s, 0, 20, 'Mean weekly turnover: ' + pct(T.ls_ew) + ' unbuffered; ' + pct(T.ls_buf) + ' buffered · ' + T.weeks + ' test weeks', {'font-size': 19, class: 't-mono'});
      if (!P || !P.available) {                                     // a clone without the local ranks file
        tx(s, 0, 70, 'The weekly ranks of individual stocks are not in git: they derive from CRSP data.', {'font-size': 20, class: 't-mono t-quiet'});
        tx(s, 0, 100, 'Build them with tools/build_data.py and the data folder (presentation/README.md).', {'font-size': 20, class: 't-mono t-quiet'});
        return;
      }
      var weeks = P.weeks, panels = [], pw = 740, gapx = 70;
      function jitter(id) { return ((id * 2654435761) % 1000) / 1000 - 0.5; }
      function state(wi, buffered, prev) {
        var w = weeks[wi], pos = {};
        w.id.forEach(function (id, j) {
          var q = w.pct[j], was = prev ? prev[id] || 0 : 0, side = 0;
          if (q > 0.9 || (buffered && was === 1 && q > 0.8)) side = 1;
          else if (q <= 0.1 || (buffered && was === -1 && q <= 0.2)) side = -1;
          pos[id] = side;
        });
        return pos;
      }
      function weights(pos) {                                       // equal weights within each leg: +1 long, -1 short
        var nl = 0, ns = 0, id, w = {};
        for (id in pos) { if (pos[id] === 1) nl++; else if (pos[id] === -1) ns++; }
        for (id in pos) { if (pos[id] === 1) w[id] = 1 / (nl || 1); else if (pos[id] === -1) w[id] = -1 / (ns || 1); }
        return w;
      }
      function turnover(prev, cur) {                                // half the sum of absolute weight changes
        var w1 = weights(cur), w0 = weights(prev), t = 0, id, seen = {};
        for (id in w1) { seen[id] = 1; t += Math.abs(w1[id] - (w0[id] || 0)); }
        for (id in w0) if (!seen[id]) t += Math.abs(w0[id]);
        return t / 2;
      }
      function entries(prev, cur) { var n = 0, id; for (id in cur) if (cur[id] !== 0 && (prev[id] || 0) !== cur[id]) n++; return n; }
      [false, true].forEach(function (buffered, pi) {
        var x0 = pi * (pw + gapx), rowL = 100, rowM = 272, rowS = 444;
        tx(s, x0, 60, buffered ? 'Buffered' : 'Unbuffered', {'font-size': 22, 'font-weight': 600, fill: buffered ? COL.cobalt : COL.ink});
        tx(s, x0 + pw, 60, buffered ? 'Long: enter > 90%, exit ≤ 80% · Short: enter ≤ 10%, exit > 20%' : 'Long: > 90% · Short: ≤ 10%', {'text-anchor': 'end', 'font-size': 14, class: 't-mono t-quiet'});
        tx(s, x0, rowL + 6, 'Long: top decile', {'font-size': 15, class: 't-mono', fill: COL.cobalt});
        tx(s, x0 + pw, rowS + 6, 'Short: bottom decile', {'text-anchor': 'end', 'font-size': 15, class: 't-mono'});
        tx(s, x0 + pw, rowM + 72, 'forecast rank →', {'text-anchor': 'end', 'font-size': 14, class: 't-mono t-quiet'});
        panels.push({buffered: buffered, X: lin(0, 1, x0 + 60, x0 + pw - 10), rowL: rowL, rowM: rowM, rowS: rowS, layer: el('g', {}, s), dots: {},
          count: tx(s, x0, 532, '', {'font-size': 18, class: 't-mono'}), cost: tx(s, x0, 560, '', {'font-size': 18, class: 't-mono t-quiet'}),
          turn: 0, ent: 0, lastTurn: 0, lastEnt: 0, prev: null});
      });
      function place(p, wi, pos, ms, flash) {
        var w = weeks[wi], seen = {};
        w.id.forEach(function (id, j) {
          seen[id] = 1;
          var d = p.dots[id]; if (!d) d = p.dots[id] = el('circle', {r: 4, cx: p.X(w.pct[j]), cy: p.rowM + jitter(id) * 110}, p.layer);
          var side = pos[id], ty = side === 1 ? p.rowL + jitter(id) * 56 : side === -1 ? p.rowS + jitter(id) * 56 : p.rowM + jitter(id) * 110;
          var enter = flash && side !== 0 && (p.prev ? (p.prev[id] || 0) : 0) !== side;
          d.setAttribute('fill', side === 1 ? COL.cobalt : side === -1 ? COL.ink : '#C9C9C6');
          d.setAttribute('opacity', 1);
          d.setAttribute('r', enter ? 6.5 : 4);
          d.setAttribute('stroke', enter ? (side === -1 ? COL.cobalt : COL.ink) : 'none'); d.setAttribute('stroke-width', enter ? 2 : 0);
          d._to = [p.X(w.pct[j]), ty];
        });
        Object.keys(p.dots).forEach(function (id) { if (!seen[id]) p.dots[id].setAttribute('opacity', 0); });
        var from = {};
        Object.keys(p.dots).forEach(function (id) { from[id] = [+p.dots[id].getAttribute('cx'), +p.dots[id].getAttribute('cy')]; });
        if (!ms) { Object.keys(p.dots).forEach(function (id) { var t = p.dots[id]._to; if (t) { p.dots[id].setAttribute('cx', t[0]); p.dots[id].setAttribute('cy', t[1]); } }); return; }
        tween(box, ms, function (u) {
          var e = ease(u);
          Object.keys(p.dots).forEach(function (id) { var t = p.dots[id]._to, f = from[id]; if (!t) return; p.dots[id].setAttribute('cx', f[0] + (t[0] - f[0]) * e); p.dots[id].setAttribute('cy', f[1] + (t[1] - f[1]) * e); });
        });
      }
      function tally(p, wi) {
        p.count.textContent = 'week ' + (wi + 1) + ' of ' + weeks.length + (wi ? ': ' + p.lastEnt + ' stocks entering positions · turnover ' + Math.round(p.lastTurn * 100) + '%' : ': positions formed');
        p.cost.textContent = wi ? 'cumulative: ' + p.ent + ' entries · turnover ' + Math.round(p.turn * 100) + '% · cost at 10 bp a unit: ' + (p.turn * 0.1).toFixed(2) + '%' : '';
      }
      function advance(p, wi, ms, flash) {
        var pos = state(wi, p.buffered, p.prev);
        if (p.prev) { p.lastTurn = turnover(p.prev, pos); p.lastEnt = entries(p.prev, pos); p.turn += p.lastTurn; p.ent += p.lastEnt; }
        place(p, wi, pos, ms, flash); p.prev = pos; tally(p, wi);
      }
      // k = 0: the first week's stocks by forecast, no legs. k = 1: the first week's legs. k = 2: the weeks follow, entries ringed
      panels.forEach(function (p) {
        if (k === 0) { place(p, 0, {}, 0, false); p.count.textContent = 'week of ' + weeks[0].date + ': ' + weeks[0].id.length + ' stocks by forecast'; return; }
        place(p, 0, {}, 0, false); advance(p, 0, (play && k === 1) ? 900 : 0, false);
      });
      if (k < 2) return;
      if (!play) { for (var wi = 1; wi < weeks.length; wi++) panels.forEach(function (p) { advance(p, wi, 0, false); }); return; }
      var wi2 = 1;
      function nextWeek() {
        if (wi2 >= weeks.length) return;
        panels.forEach(function (p) { advance(p, wi2, 650, true); });
        wi2++;
        var t = setTimeout(nextWeek, 820 * U.pace(box)); (box._stops = box._stops || []).push(function () { clearTimeout(t); });
      }
      var t0 = setTimeout(nextWeek, 500 * U.pace(box)); (box._stops = box._stops || []).push(function () { clearTimeout(t0); });
    },
    enter: function (box, k) { this.draw(box, k, false); box._k = k; },
    step: function (box, k) { if (box._k === k) return; box._k = k; this.draw(box, k, true); },
    leave: function (box) { halt(box); },
    still: function (box) { this.draw(box, 2, false); }
  };

  // ------------------------------------------------------------------ 5.1 three checks: the placebo, every alpha, the Sharpe intervals
  C.ladder = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, F = D().facts, PL = D().placebo_series, A = D().alpha_rain, TV = D().tvp;
    var rungs = [
      ['1', 'Price features as controls', 'Significant improvement over price controls: ' + TV.better + ' of ' + TV.pairs + ' comparisons', 'LDA attention lagged ' + PL.days + ' trading days: no significant IC difference'],
      ['2', 'Three-factor alpha', 'Significant three-factor alpha: ' + A.above + ' positive, ' + A.below + ' negative, out of ' + A.n + ' portfolios', ''],
      ['3', 'Long-short net Sharpe against the index', '', '']];
    var ys = [0, 238, 446];
    rungs.forEach(function (r, i) {
      var y = ys[i] + 12, gg = reveal(box, el('g', {}, s), k, i + 1, animate, 600);
      el('circle', {cx: 22, cy: y + 14, r: 22, fill: 'none', stroke: COL.ink, 'stroke-width': 2}, gg);
      tx(gg, 22, y + 22, r[0], {'text-anchor': 'middle', 'font-size': 22, class: 't-mono'});
      tx(gg, 70, y + 24, r[1], {'font-size': 28, 'font-weight': 500});
      if (r[2]) tx(gg, 70, y + 54, r[2], {'font-size': 20, class: 't-mono'});
      if (r[3]) tx(gg, 70, y + 80, r[3], {'font-size': 20, class: 't-mono t-quiet'});
    });

    // 1: the weekly IC of the real exposures and of the misdated ones draw together, then their means
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 600), n1 = PL.dates.length, X1 = lin(0, n1 - 1, 70, 1110);
    var lo1 = 0, hi1 = 0;
    PL.real.concat(PL.placebo).forEach(function (v) { lo1 = Math.min(lo1, v); hi1 = Math.max(hi1, v); });
    var Y1 = lin(lo1, hi1, 226, 110);
    el('line', {x1: 70, x2: 1110, y1: Y1(0), y2: Y1(0), class: 'ln-axis', 'stroke-width': 1.5}, g1);
    var seen1 = {};
    PL.dates.forEach(function (d, i) {
      var yr = d.slice(0, 4); if (seen1[yr] || yr < '2022') return; seen1[yr] = 1;
      el('line', {x1: X1(i), x2: X1(i), y1: 110, y2: 226, class: 'ln-rule'}, g1);
      tx(g1, X1(i) + 4, 237, yr, {'font-size': 14, class: 't-mono t-quiet'});
    });
    var defs = el('defs', {}, s), c1 = el('clipPath', {id: 'pl-clip'}, defs), cr1 = el('rect', {x: 60, y: 100, width: 0, height: 140}, c1);
    var gp1 = el('g', {'clip-path': 'url(#pl-clip)'}, g1);
    function pline(arr, a) { var d = ''; arr.forEach(function (v, i) { d += (i ? 'L' : 'M') + X1(i).toFixed(1) + ' ' + Y1(v).toFixed(1); }); return el('path', Object.assign({d: d, fill: 'none', 'stroke-linejoin': 'round'}, a), gp1); }
    pline(PL.placebo, {stroke: '#A8A8A6', 'stroke-width': 2});
    pline(PL.real, {stroke: COL.cobalt, 'stroke-width': 2});
    var m1 = el('g', {opacity: 0}, g1);
    el('line', {x1: 70, x2: 1110, y1: Y1(PL.mean_real), y2: Y1(PL.mean_real), stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '6 5'}, m1);
    el('line', {x1: 70, x2: 1110, y1: Y1(PL.mean_placebo), y2: Y1(PL.mean_placebo), stroke: COL.grey, 'stroke-width': 2, 'stroke-dasharray': '6 5'}, m1);
    var ya = Y1(PL.mean_real), yb = Y1(PL.mean_placebo);
    if (Math.abs(ya - yb) < 20) { var mm = (ya + yb) / 2, lower = ya > yb; ya = mm + (lower ? 10 : -10); yb = mm + (lower ? -10 : 10); }
    tx(m1, 1122, ya + 5, 'actual dates: mean IC ' + signed(PL.mean_real, 3), {'font-size': 16, class: 't-mono t-cobalt'});
    tx(m1, 1122, yb + 5, 'lagged ' + PL.days + ' days: ' + signed(PL.mean_placebo, 3), {'font-size': 16, class: 't-mono t-quiet'});
    tx(m1, 1122, Math.max(ya, yb) + 27, 'difference: t = ' + signed(PL.t, 2) + ' < ' + PL.crit.toFixed(2), {'font-size': 15, class: 't-mono t-quiet'});
    if (k >= 1) {
      if (animate && k === 1) tween(box, 1200, function (u) { cr1.setAttribute('width', 1080 * u); }, function () { tween(box, 400, function (u) { m1.setAttribute('opacity', u); }); });
      else { cr1.setAttribute('width', 1080); m1.setAttribute('opacity', 1); }
    }

    // 2: every alpha t statistic, divided by its critical value, falls onto one line
    var g2 = reveal(box, el('g', {}, s), k, 2, animate, 600), rel = A.pts.map(function (p) { return p[0] / A.crit[p[1]]; });
    var lo2 = Math.min(-1.5, Math.min.apply(null, rel)) - 0.2, hi2 = Math.max(1.5, Math.max.apply(null, rel)) + 0.2, X2 = lin(lo2, hi2, 70, 1300);
    var yTop = 342, yAxis = 430;
    el('line', {x1: 70, x2: 1300, y1: yAxis, y2: yAxis, class: 'ln-axis', 'stroke-width': 1.5}, g2);
    [-1, 0, 1].forEach(function (v) { el('line', {x1: X2(v), x2: X2(v), y1: yTop, y2: yAxis, stroke: v ? COL.ink : COL.mist, 'stroke-width': v ? 1.5 : 1, 'stroke-dasharray': v ? '5 4' : null}, g2); });
    tx(g2, X2(-1) - 8, yTop + 12, '−critical value', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});
    tx(g2, X2(1) + 8, yTop + 12, '+critical value', {'font-size': 15, class: 't-mono t-quiet'});
    tx(g2, 70, yAxis + 18, 'alpha t statistic ÷ its small-sample critical value (' + A.crit.M + ', ' + A.crit['2W'] + ', ' + A.crit.W + ')', {'font-size': 15, class: 't-mono t-quiet'});
    var cnt = el('g', {opacity: 0}, g2);
    tx(cnt, X2(-1) - 8, yTop + 34, A.below + ' below', {'text-anchor': 'end', 'font-size': 18, class: 't-mono', 'font-weight': 500});
    tx(cnt, X2(1) + 8, yTop + 34, A.above + ' above', {'font-size': 18, class: 't-mono t-cobalt', 'font-weight': 500});
    var order = rel.map(function (v, i) { return i; }).sort(function (a, b) { return (Math.abs(rel[a]) > 1 ? 1 : 0) - (Math.abs(rel[b]) > 1 ? 1 : 0); });
    var dots2 = order.map(function (i, j) {
      var v = rel[i], jit = ((i * 2654435761) % 1000) / 1000;
      var d = el('circle', {cx: X2(v), cy: yTop, r: 3.2, fill: v < -1 ? COL.ink : v > 1 ? COL.cobalt : '#C9C9C6', opacity: 0}, g2);
      d._y = 388 + jit * 36; d._i = j; return d;
    });
    function fall(U) { var n = dots2.length; dots2.forEach(function (d) { var u = Math.max(0, Math.min(1, (U - 0.5 * d._i / n) / 0.5)); d.setAttribute('cy', yTop + (d._y - yTop) * ease(u)); d.setAttribute('opacity', u > 0 ? 1 : 0); }); }
    if (k >= 2) {
      if (animate && k === 2) tween(box, 1500, fall, function () { tween(box, 300, function (u) { cnt.setAttribute('opacity', u); }); });
      else { fall(1); cnt.setAttribute('opacity', 1); }
    }

    // 3: the net Sharpe of the chosen portfolios against the index
    var gI = reveal(box, el('g', {}, s), k, 3, animate, 700), X = lin(-2.5, 2.2, 400, 1290), base = 492;
    [-2, -1, 0, 1, 2].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: base + 6, y2: base + 198, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, gI);
      tx(gI, X(v), base + 214, v ? signed(v, 0) : '0', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
    });
    tx(gI, 70, base + 232, 'grey tick: the value-weighted index, same months', {'font-size': 15, class: 't-mono t-quiet'});
    tx(gI, X(2.2), base + 232, 'net Sharpe, test, 95% interval', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});
    var idx = F.index_vw, codes = ['M', '2W', 'W'];
    [[F.headlines.rows, 'Single model selected on validation', 24, true], [F.protocol.rows, 'Fixed protocol: validation selection among 25 ensembles; buffered', 120, false]].forEach(function (grp) {
      tx(gI, 70, base + grp[2], grp[1], {'font-size': 16, class: 't-mono' + (grp[3] ? '' : ' t-quiet'), stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
      grp[0].forEach(function (h, i) {
        var y = base + grp[2] + 26 + i * 22, solid = grp[3];
        tx(gI, 70, y + 6, h.freq, {'font-size': 18, class: solid ? null : 't-quiet'});
        el('line', {x1: X(h.lo), x2: X(h.hi), y1: y, y2: y, stroke: solid ? COL.ink : COL.grey, 'stroke-width': 2.5, 'stroke-dasharray': solid ? null : '7 5'}, gI);
        el('line', {x1: X(idx[codes[i]]), x2: X(idx[codes[i]]), y1: y - 10, y2: y + 10, stroke: COL.mist, 'stroke-width': 3}, gI);
        el('circle', {cx: X(h.net), cy: y, r: 6, fill: solid ? COL.ink : '#fff', stroke: solid ? null : COL.grey, 'stroke-width': solid ? 0 : 2.5}, gI);
        tx(gI, X(h.hi) + 10, y + 6, signed(h.net), {'font-size': 16, class: 't-mono' + (solid ? '' : ' t-quiet')});
      });
    });
  });

  // ------------------------------------------------------------------ 5.2 validation's ten picks drop to their test scores, then the search as 375 paths
  C.vdrop = {
    draw: function (box, k, play) {
      halt(box);
      var g = surface(box), s = g.s, P = D().validation.panels, R = D().race, Sr = D().search;
      var ext = 0;
      P.forEach(function (p) { p.pts.forEach(function (q) { ext = Math.max(ext, Math.abs(q[0]), Math.abs(q[1])); }); });
      ext += 0.004;
      var pw = 236, gap = 46, top = 50, H = 220, GREY = '#B8B8B5';
      var dots = [], favs = [], favLabels = [];
      P.forEach(function (p, i) {
        var x0 = i * (pw + gap), X = lin(-ext, ext, x0, x0 + pw), Y = lin(-ext, ext, top + H, top);
        el('rect', {x: x0, y: top, width: pw, height: H, fill: 'none', stroke: COL.rule}, s);
        el('line', {x1: X(0), x2: X(0), y1: top, y2: top + H, class: 'ln-rule'}, s);
        el('line', {x1: x0, x2: x0 + pw, y1: Y(0), y2: Y(0), class: 'ln-rule'}, s);
        el('line', {x1: X(-ext), y1: Y(-ext), x2: X(ext), y2: Y(ext), stroke: COL.mist, 'stroke-dasharray': '6 6', 'stroke-width': 1.5}, s);
        tx(s, x0, top - 32, p.name + ' · ' + p.n + ' experiments', {'font-size': 17, class: 't-mono'});
        tx(s, x0, top - 8, 'Spearman correlation ' + signed(p.rho), {'font-size': 17, class: 't-mono t-quiet'});
        var trails = el('g', {}, s), here = [];
        p.pts.forEach(function (q) {
          var d = el('circle', {cx: X(q[0]), cy: Y(q[0]), r: 3.4, fill: GREY, stroke: q[3] ? COL.ink : 'none', 'stroke-width': q[3] ? 1.4 : 0}, s);
          d._a = Y(q[0]); d._b = Y(q[1]); d._fav = !!q[2];
          if (d._fav) d._trail = el('line', {x1: X(q[0]), y1: Y(q[0]), x2: X(q[0]), y2: Y(q[0]), stroke: COL.cobalt, 'stroke-width': 1.2, opacity: 0}, trails);
          dots.push(d); if (d._fav) here.push(d);
        });
        here.forEach(function (d) { s.appendChild(d); favs.push(d); });            // the ten stay on top
        favLabels.push(tx(s, x0, top + H + 32, 'Top 10: mean test IC ' + signed(p.fav, 3), {'font-size': 16, class: 't-mono t-cobalt', opacity: 0}));
      });
      var lx = 3 * pw + 2 * gap + 40, leg = [[['Blue: top 10 by validation IC', 't-cobalt'], ['Outlined: EBM', ''], ['Dashed: equal IC', 't-quiet']],
        [['x: validation rank IC', 't-quiet'], ['y: test rank IC', 't-quiet']]];
      leg.forEach(function (col, ci) {
        var ly = top + 14;
        col.forEach(function (l) { if (!l) { ly += 14; return; } tx(s, lx + ci * 335, ly, l[0], {'font-size': 18, class: 't-mono ' + l[1]}); ly += 24; });
      });
      // step 1: the ten are picked out, a hold, then every dot drops to its test IC; the ten leave trails; the labels come once they land
      function select(u) {
        favs.forEach(function (d) { d.setAttribute('r', 3.4 + 3.1 * u); d.setAttribute('fill', u > 0.5 ? COL.cobalt : GREY); });
        dots.forEach(function (d) { if (!d._fav) d.setAttribute('opacity', 1 - 0.55 * u); });
      }
      function drop(u) {
        dots.forEach(function (d) {
          var y = d._a + (d._b - d._a) * u; d.setAttribute('cy', y);
          if (d._trail) { d._trail.setAttribute('y2', y); d._trail.setAttribute('opacity', 0.45 * Math.min(1, u * 3)); }
        });
      }
      function landed() { favLabels.forEach(function (t) { t.setAttribute('opacity', 1); }); }
      if (k >= 1) {
        if (play && k === 1) {
          tween(box, 600, function (u) { select(ease(u)); }, function () {
            var t = setTimeout(function () { tween(box, 1500, function (u) { drop(ease(u)); }, landed); }, 500 * U.pace(box));
            (box._stops = box._stops || []).push(function () { clearTimeout(t); });
          });
        } else { select(1); drop(1); landed(); }
      }

      // step 2: the search. Every weekly decile long-short of the main pool, cumulative net return over the test weeks
      var gR = el('g', {opacity: k >= 2 ? 1 : 0}, s), y0 = 334, y1 = 512, L = 60, Rt = 1040;
      var lo = Infinity, hi = -Infinity;
      R.paths.forEach(function (p) { p.forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); }); });
      var XR = lin(0, R.dates.length - 1, L, Rt), YR = lin(lo, hi, y1, y0 + 26);
      tx(gR, 0, y0 + 6, 'Cumulative net return · ' + R.n + ' weekly decile long–short portfolios · ' + R.dates.length + ' test weeks', {'font-size': 19, class: 't-mono'});
      var stepV = hi - lo > 120 ? 40 : 20;
      for (var v = Math.ceil(lo / stepV) * stepV; v <= hi; v += stepV) {
        el('line', {x1: L, x2: Rt, y1: YR(v), y2: YR(v), class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 1.5}, gR);
        tx(gR, L - 10, YR(v) + 6, (v > 0 ? '+' : '') + v + '%', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
      }
      var seen = {};
      R.dates.forEach(function (d, i) {
        var yr = d.slice(0, 4);
        if (seen[yr] || yr < '2022') return; seen[yr] = 1;
        el('line', {x1: XR(i), x2: XR(i), y1: y0 + 26, y2: y1, class: 'ln-rule'}, gR);
        tx(gR, XR(i) + 6, y1 + 20, yr, {'font-size': 16, class: 't-mono t-quiet'});
      });
      var defs = el('defs', {}, s), clip = el('clipPath', {id: 'race-clip'}, defs);
      var cr = el('rect', {x: L - 2, y: y0 + 10, width: 0, height: y1 - y0 + 10}, clip);
      var gP = el('g', {'clip-path': 'url(#race-clip)'}, gR);
      function line(arr, a) { var d = ''; arr.forEach(function (v, i) { d += (i ? 'L' : 'M') + XR(i).toFixed(1) + ' ' + YR(v).toFixed(1); }); return el('path', Object.assign({d: d, fill: 'none', 'stroke-linejoin': 'round'}, a), gP); }
      R.paths.forEach(function (p, i) { if (i !== R.pick && i !== R.best) line(p, {stroke: '#D4D4D1', 'stroke-width': 1}); });
      line(R.median, {stroke: COL.ink, 'stroke-width': 2});
      line(R.paths[R.best], {stroke: COL.ink, 'stroke-width': 1.6, 'stroke-dasharray': '5 4'});
      line(R.paths[R.pick], {stroke: COL.cobalt, 'stroke-width': 3});
      var last = R.dates.length - 1, ends = [
        {v: R.paths[R.pick][last], t: 'selected on validation ' + signed(R.paths[R.pick][last], 1) + '%', c: 't-cobalt'},
        {v: R.median[last], t: 'median ' + signed(R.median[last], 1) + '%', c: ''},
        {v: R.paths[R.best][last], t: 'highest net Sharpe ' + signed(R.paths[R.best][last], 1) + '%', c: 't-quiet'}];
      ends.forEach(function (e) { e.y = YR(e.v); }); ends.sort(function (a, b) { return a.y - b.y; });
      for (var i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 22) ends[i].y = ends[i - 1].y + 22;
      var lab = el('g', {opacity: 0}, gR);
      ends.forEach(function (e) { tx(lab, Rt + 10, e.y + 6, e.t, {'font-size': 17, class: 't-mono ' + e.c}); });
      var w = Sr.rows.filter(function (r) { return r.freq === 'W' && r.rule === 'Long-short'; })[0];
      tx(gR, 0, y1 + 52, 'Observed maximum net Sharpe ' + w.best.toFixed(2) + ' − expected maximum under no skill ' + w.luck.toFixed(2) + ' · ' + R.n + ' experiments · 7 trading rules × 3 frequencies: ' + Sr.short + ' of ' + Sr.rows.length + ' below', {'font-size': 18, class: 't-mono'});
      function grow(u) { cr.setAttribute('width', (Rt - L + 4) * u); }
      if (k >= 2) {
        if (play && k === 2) { tween(box, 3000, grow, function () { tween(box, 400, function (u) { lab.setAttribute('opacity', u); }); }); }
        else { grow(1); lab.setAttribute('opacity', 1); }
      }
    },
    enter: function (box, k) { this.draw(box, k, false); box._k = k; },
    step: function (box, k) { if (box._k === k) return; box._k = k; this.draw(box, k, true); },
    leave: function (box) { halt(box); },
    still: function (box) { this.draw(box, 2, false); }
  };

  // ------------------------------------------------------------------ 5.3 trading less: the average experiment, plain and buffered, through the test period
  C.heldup = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, B = D().facts.buffer, P = D().buffer_paths.panels;
    tx(s, 0, 22, 'Annualized net-return gain from buffering · Test period · Mean across 200 base experiments', {'font-size': 20, 'font-weight': 600});
    tx(s, 0, 48, 'Cumulative net return: unbuffered (grey) and buffered decile long–short · Blue: exceeds the small-sample critical value', {'font-size': 15, class: 't-mono t-quiet'});
    var lo = 0, hi = 0;
    P.forEach(function (p) { p.plain.concat(p.buffered).forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); }); });
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var pw = 390, gapx = 55, top = 76, bot = 332, stepV = hi - lo > 16 ? 4 : 2;
    var defs = el('defs', {}, s), clip = el('clipPath', {id: 'buf-clip'}, defs);
    var cr = el('rect', {x: 0, y: top - 10, width: 0, height: bot - top + 20}, clip);
    var labels = el('g', {opacity: 0}, s);
    P.forEach(function (p, i) {
      var x0 = 50 + i * (pw + gapx), n = p.dates.length, X = lin(0, n - 1, x0, x0 + pw), Y = lin(lo, hi, bot, top), beyond = B.t[i] > B.crit[i];
      tx(s, x0 + 10, top + 26, ['Monthly', 'Biweekly', 'Weekly'][i], {'font-size': 22, stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
      el('rect', {x: x0, y: top, width: pw, height: bot - top, fill: 'none', stroke: COL.rule}, s);
      for (var v = Math.ceil(lo / stepV) * stepV; v <= hi; v += stepV) {
        el('line', {x1: x0, x2: x0 + pw, y1: Y(v), y2: Y(v), class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 1.5}, s);
        if (i === 0) tx(s, x0 - 10, Y(v) + 6, (v > 0 ? '+' : '') + v + '%', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
      }
      var seen = {};
      p.dates.forEach(function (d, j) {
        var yr = d.slice(0, 4); if (seen[yr] || yr < '2022') return; seen[yr] = 1;
        el('line', {x1: X(j), x2: X(j), y1: top, y2: bot, class: 'ln-rule'}, s);
        tx(s, X(j) + 5, bot + 20, yr, {'font-size': 16, class: 't-mono t-quiet'});
      });
      var gP = el('g', {'clip-path': 'url(#buf-clip)'}, s), band = '', j;
      for (j = 0; j < n; j++) band += (j ? 'L' : 'M') + X(j).toFixed(1) + ' ' + Y(p.plain[j]).toFixed(1);
      for (j = n - 1; j >= 0; j--) band += 'L' + X(j).toFixed(1) + ' ' + Y(p.buffered[j]).toFixed(1);
      el('path', {d: band + 'Z', fill: beyond ? 'rgba(44,75,239,.12)' : 'rgba(22,22,22,.07)'}, gP);
      function line(arr, a) { var d = ''; arr.forEach(function (v, q) { d += (q ? 'L' : 'M') + X(q).toFixed(1) + ' ' + Y(v).toFixed(1); }); return el('path', Object.assign({d: d, fill: 'none', 'stroke-linejoin': 'round'}, a), gP); }
      line(p.plain, {stroke: '#A8A8A6', 'stroke-width': 2});
      line(p.buffered, {stroke: beyond ? COL.cobalt : COL.grey, 'stroke-width': 2.5});
      var e1 = Y(p.plain[n - 1]), e2 = Y(p.buffered[n - 1]);
      if (Math.abs(e1 - e2) < 20) { var mid = (e1 + e2) / 2, lower = e1 > e2; e1 = mid + (lower ? 10 : -10); e2 = mid + (lower ? -10 : 10); }
      var halo = {stroke: '#fff', 'stroke-width': 5, 'stroke-linejoin': 'round', 'paint-order': 'stroke', 'text-anchor': 'end', 'font-size': 15};
      tx(labels, x0 + pw - 6, e1 + 5, 'unbuffered ' + signed(p.plain[n - 1], 1) + '%', Object.assign({class: 't-mono t-quiet'}, halo));
      tx(labels, x0 + pw - 6, e2 + 5, 'buffered ' + signed(p.buffered[n - 1], 1) + '%', Object.assign({class: 't-mono ' + (beyond ? 't-cobalt' : 't-quiet')}, halo));
      tx(labels, x0 + 10, top + 50, '+' + B.gain[i].toFixed(1) + '% a year · t = ' + B.t[i].toFixed(1) + (beyond ? ' > ' : ' < ') + B.crit[i], {'font-size': 17, class: 't-mono', 'font-weight': 500, stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
    });
    function grow(u) { cr.setAttribute('width', 1400 * u); }
    if (k >= 1) {
      if (animate && k === 1) tween(box, 2200, grow, function () { tween(box, 400, function (u) { labels.setAttribute('opacity', u); }); });
      else { grow(1); labels.setAttribute('opacity', 1); }
    }
  });

  // ------------------------------------------------------------------ 5.4 the proposed evaluation: a model with a July 2025 training cutoff, tested on what came after
  C.next = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, N = D().facts.next, sp = D().facts.splits, Sr = D().search, day = U.day;
    var L = 260, R = 1520, X = lin(day('2012-01-01'), day('2027-01-01'), L, R), d = function (iso) { return X(day(iso)); };
    var defs = el('defs', {}, s);
    var grad = el('linearGradient', {id: 'cut-grad', x1: 0, x2: 1, y1: 0, y2: 0}, defs);
    el('stop', {offset: '0', 'stop-color': '#2C4BEF', 'stop-opacity': 0}, grad);
    el('stop', {offset: '0.3', 'stop-color': '#2C4BEF', 'stop-opacity': 0.2}, grad);
    el('stop', {offset: '1', 'stop-color': '#2C4BEF', 'stop-opacity': 0.2}, grad);
    var hatch = el('pattern', {id: 'hr-next', width: 8, height: 8, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)'}, defs);
    el('line', {x1: 0, y1: 0, x2: 0, y2: 8, stroke: COL.red, 'stroke-width': 3}, hatch);
    for (var y = 2012; y <= 2026; y += 2) {
      el('line', {x1: d(y + '-01-01'), x2: d(y + '-01-01'), y1: 30, y2: 250, class: 'ln-rule'}, s);
      tx(s, d(y + '-01-01'), 20, String(y), {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'});
    }
    tx(s, 0, 20, 'Proposed evaluation', {'font-size': 19, class: 't-mono', 'font-weight': 500});

    // row 1: the protocol of this project, as on 2.2
    var g1 = reveal(box, el('g', {}, s), k, 1, animate, 500), r1 = 58;
    tx(g1, 0, r1 + 22, 'Our protocol', {'font-size': 22});
    [['train', sp.train, 0.22], ['validation', sp.valid, 0.4], ['test', sp.test, 0.62]].forEach(function (p) {
      el('rect', {x: d(p[1][0]), y: r1, width: d(p[1][1]) - d(p[1][0]), height: 30, fill: 'rgba(22,22,22,' + p[2] * 0.45 + ')'}, g1);
      tx(g1, (d(p[1][0]) + d(p[1][1])) / 2, r1 - 8, p[0], {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
    });

    // row 2: the model's training data, with its two cutoffs
    var r2 = 128, tEnd = d(N.training + '-31'), tRel = d(N.reliable + '-01');
    tx(g1, 0, r2 + 22, N.model, {'font-size': 22});
    var bar = el('rect', {x: d('2013-01-01'), y: r2, width: 0, height: 30, fill: 'url(#cut-grad)'}, g1);
    el('line', {x1: tRel, x2: tRel, y1: r2 - 6, y2: r2 + 36, stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '5 4'}, g1);
    var cutTick = el('line', {x1: tEnd, x2: tEnd, y1: r2 - 14, y2: r2 + 44, stroke: COL.cobalt, 'stroke-width': 3, opacity: 0}, g1);
    var cutLab = tx(g1, tEnd - 8, r2 - 4, 'Training cutoff: July 2025', {'text-anchor': 'end', 'font-size': 19, class: 't-mono', 'font-weight': 500, opacity: 0, stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
    tx(g1, tRel - 10, r2 + 58, 'reliable knowledge cutoff: Feb 2025', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});

    // row 3: the proposed test, after the cutoff; the 2022–2024 weeks lie before it
    var r3 = 206, tFrom = d(N.test_from), tTo = d(N.test_to);
    tx(g1, 0, r3 + 22, 'Proposed test', {'font-size': 22});
    var tb = el('rect', {x: tFrom, y: r3, width: 0, height: 30, fill: COL.cobalt}, g1);
    var tbLab = tx(g1, tTo, r3 + 74, 'Test period: August 2025 onward', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-cobalt', opacity: 0});
    el('rect', {x: d(sp.test[0]), y: r3, width: d(sp.test[1]) - d(sp.test[0]), height: 30, fill: 'url(#hr-next)', stroke: COL.red, 'stroke-width': 1.5}, g1);
    tx(g1, d(sp.test[1]), r3 + 52, '2022–2024: before the training cutoff · pre-cutoff sample for bias assessment', {'text-anchor': 'end', 'font-size': 15, class: 't-red'});
    function grow1(u) {
      var a = Math.min(1, u * 1.6), b = Math.max(0, (u - 0.6) / 0.4);
      bar.setAttribute('width', (tEnd - d('2013-01-01')) * a); cutTick.setAttribute('opacity', a >= 1 ? 1 : 0); cutLab.setAttribute('opacity', a >= 1 ? 1 : 0);
      tb.setAttribute('width', (tTo - tFrom) * b); tbLab.setAttribute('opacity', b >= 1 ? 1 : 0);
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 1800, function (u) { grow1(ease(u)); }); else grow1(1); }

    // row 4, left: the design for stock returns. Right: prediction markets, schematic
    var g4 = reveal(box, el('g', {}, s), k, 2, animate, 600), y4 = 318;
    tx(g4, 0, y4, 'Stock returns, up or down', {'font-size': 19, 'font-weight': 600});
    var cx0 = 210, cw2 = 168, ch2 = 40, rows = ['firm named', 'firm anonymized'], cols = ['before the cutoff', 'after the cutoff'];
    cols.forEach(function (c, j) { tx(g4, cx0 + j * (cw2 + 8) + cw2 / 2, y4 + 28, c, {'text-anchor': 'middle', 'font-size': 14, class: 't-mono t-quiet'}); });
    rows.forEach(function (r, i) {
      var yy = y4 + 38 + i * (ch2 + 8);
      tx(g4, cx0 - 10, yy + 26, r, {'text-anchor': 'end', 'font-size': 14, class: 't-mono t-quiet'});
      cols.forEach(function (c, j) {
        var ev = j === 1 && i === 1, xx = cx0 + j * (cw2 + 8);
        el('rect', {x: xx, y: yy, width: cw2, height: ch2, fill: ev ? COL.cobalt : j === 0 ? 'url(#hr-next)' : '#E6E6E3', stroke: ev ? COL.cobalt : COL.rule}, g4);
        tx(g4, xx + cw2 / 2, yy + 25, ev ? 'the evaluation' : j === 0 ? 'bias assessment' : 'distraction check', {'text-anchor': 'middle', 'font-size': 13, class: 't-mono', fill: ev ? '#fff' : COL.graphite || '#6F6F6F',
          stroke: j === 0 ? '#fff' : null, 'stroke-width': j === 0 ? 4 : null, 'paint-order': 'stroke'});
      });
    });
    tx(g4, 0, y4 + 152, 'same news, same protocol as section 5;', {'font-size': 14, class: 't-mono t-quiet'});
    tx(g4, 0, y4 + 172, 'the model reads the week’s headlines and gives P(up)', {'font-size': 14, class: 't-mono t-quiet'});
    var px0 = 660, pxR = 1520, XP = lin(day('2025-01-01'), day('2026-11-01'), px0, pxR), cut = XP(day('2025-07-31'));
    tx(g4, px0, y4, 'Prediction markets', {'font-size': 19, 'font-weight': 600});
    tx(g4, pxR, y4, 'schematic', {'text-anchor': 'end', 'font-size': 14, class: 't-mono t-quiet'});
    [['2025-01-01', 'Jan 2025'], ['2025-07-01', 'Jul 2025'], ['2026-01-01', 'Jan 2026'], ['2026-07-01', 'Jul 2026']].forEach(function (t) {
      el('line', {x1: XP(day(t[0])), x2: XP(day(t[0])), y1: y4 + 14, y2: y4 + 124, class: 'ln-rule'}, g4);
      tx(g4, XP(day(t[0])) + 4, y4 + 138, t[1], {'font-size': 13, class: 't-mono t-quiet'});
    });
    el('line', {x1: cut, x2: cut, y1: y4 + 12, y2: y4 + 126, stroke: COL.cobalt, 'stroke-width': 2}, g4);
    [['2025-02-01', '2025-05-15'], ['2025-04-01', '2025-09-01'], ['2025-06-10', '2025-08-20'], ['2025-08-05', '2025-11-30'], ['2025-09-15', '2026-02-01'],
     ['2025-11-01', '2026-04-10'], ['2026-01-20', '2026-06-01'], ['2026-03-01', '2026-09-15']].forEach(function (q, i) {
      var after = day(q[0]) > day('2025-07-31'), yq = y4 + 26 + i * 12;
      el('line', {x1: XP(day(q[0])), x2: XP(day(q[1])), y1: yq, y2: yq, stroke: after ? COL.cobalt : '#C9C9C6', 'stroke-width': 6, 'stroke-linecap': 'round'}, g4);
    });
    tx(g4, px0, y4 + 158, 'a question: created → resolved · grey: dated before the training cutoff, excluded · blue: dated after', {'font-size': 14, class: 't-mono t-quiet'});
    tx(g4, px0, y4 + 178, 'P(yes) from the question and the news to date · benchmark: market price at the same time · score: Brier, net P&L', {'font-size': 13, class: 't-mono t-quiet'});

    // row 5: the rules, fixed now
    var g5 = reveal(box, el('g', {}, s), k, 3, animate, 600), y5 = 548, w = Sr.rows.filter(function (r) { return r.freq === 'W' && r.rule === 'Long-short'; })[0];
    tx(g5, 0, y5, 'Fix the model, prompts and evaluation rules before testing.', {'font-size': 21, 'font-weight': 600});
    tx(g5, 0, y5 + 32, 'weekly rank IC ≥ ' + N.ic_detectable.W.toFixed(2) + ' (detectable at 80% power) · net Sharpe above the expected best of 375 with no skill, ' + w.luck.toFixed(2) + ' · Brier score below the market’s', {'font-size': 17, class: 't-mono'});
    tx(g5, 0, y5 + 58, N.model + ' (' + N.snapshot + ') · cutoffs as documented by the vendor · retirement ' + N.retirement, {'font-size': 14, class: 't-mono t-quiet'});
    tx(g5, 0, y5 + 78, 'same protocol as section 5: 10 bp a unit traded, small-sample critical values, buffered decile long-short', {'font-size': 14, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ backup: the presidents strip (Table A3)
  C.presidents = stepped(function (box) {
    var g = surface(box), s = g.s, P = D().facts.presidents, X = lin(0, 7, 120, 1260), Y = lin(0, 0.65, 520, 80);
    el('rect', {x: (X(1) + X(2)) / 2, y: 60, width: (X(5) + X(6)) / 2 - (X(1) + X(2)) / 2, height: 470, fill: 'rgba(44,75,239,.07)'}, s);
    tx(s, (X(1) + X(2)) / 2 + 10, 84, 'took office inside the 1976–1995 training window', {'font-size': 19, class: 't-mono t-cobalt'});
    [0, 0.2, 0.4, 0.6].forEach(function (v) { el('line', {x1: 60, x2: 1320, y1: Y(v), y2: Y(v), class: 'ln-rule'}, s); tx(s, 50, Y(v) + 7, v.toFixed(1), {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'}); });
    P.rows.forEach(function (r, i) {
      var inside = i >= 2 && i <= 5, bush = r[0] === 'G. W. Bush';
      el('rect', {x: X(i) - 40, y: Y(r[1]), width: 80, height: Y(0) - Y(r[1]), fill: inside ? COL.cobalt : bush ? COL.red : '#BDBDBA'}, s);
      tx(s, X(i), Y(r[1]) - 12, r[1].toFixed(2), {'text-anchor': 'middle', 'font-size': 20, class: 't-mono'});
      tx(s, X(i), 556, r[0], {'text-anchor': 'middle', 'font-size': 20});
    });
    tx(s, X(7) + 40, 600, '“Bush” also identifies George H. W. Bush', {'text-anchor': 'end', 'font-size': 17, class: 't-mono t-red'});
    tx(s, 0, 40, 'P(“president”) in “XXX is a United States <mask>.”, ordered by year taking office', {'font-size': 20, class: 't-mono'});
  });

})();
