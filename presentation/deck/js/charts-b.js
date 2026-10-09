/* Exhibits, part B: the backup slides (B1 to B5). Same conventions as part A. */
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
  // ------------------------------------------------------------------ B3 forecast to portfolio (real weekly ranks, anonymous)
  C.portfolio = {
    draw: function (box, k, play) {
      halt(box);
      var g = surface(box), s = g.s, P = D().portfolio, T = D().turnover;
      var X = lin(0, 1, 40, 1340), rowL = 76, rowM = 262, rowS = 448;
      tx(s, 0, rowL + 8, 'long: top tenth', {'font-size': 22, 'font-weight': 600, fill: COL.cobalt});
      tx(s, 1340, rowS + 8, 'short: bottom tenth', {'text-anchor': 'end', 'font-size': 22, 'font-weight': 600});
      tx(s, 1340, rowM + 92, 'forecast rank →', {'text-anchor': 'end', 'font-size': 19, class: 't-mono t-quiet'});
      var info = tx(s, 0, 18, '', {'font-size': 21, class: 't-mono'});
      tx(s, 0, 548, 'This portfolio, all ' + T.weeks + ' test weeks: turnover ' + pct(T.ls_ew) + ' a week, ' + pct(T.ls_buf) + ' with the buffer (half the sum of weight changes).', {'font-size': 19, class: 't-mono'});
      if (!P || !P.available) {                                     // a clone without the local ranks file
        tx(s, 0, rowM - 14, 'The weekly ranks of individual stocks are not in git: they derive from CRSP data.', {'font-size': 20, class: 't-mono t-quiet'});
        tx(s, 0, rowM + 16, 'Build them with tools/build_data.py and the data folder (presentation/README.md).', {'font-size': 20, class: 't-mono t-quiet'});
        return;
      }
      var weeks = P.weeks, dots = {}, layer = el('g', {}, s);
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
      function place(wi, pos, ms) {
        var w = weeks[wi], seen = {};
        w.id.forEach(function (id, j) {
          seen[id] = 1;
          var d = dots[id]; if (!d) d = dots[id] = el('circle', {r: 4.2, cx: X(w.pct[j]), cy: rowM + jitter(id) * 120}, layer);
          var side = pos[id], tx_ = X(w.pct[j]), ty = side === 1 ? rowL + jitter(id) * 60 : side === -1 ? rowS + jitter(id) * 60 : rowM + jitter(id) * 120;
          d.setAttribute('fill', side === 1 ? COL.cobalt : side === -1 ? COL.ink : '#C9C9C6');
          d.setAttribute('opacity', 1);
          d._to = [tx_, ty];
        });
        Object.keys(dots).forEach(function (id) { if (!seen[id]) dots[id].setAttribute('opacity', 0); });
        var from = {};
        Object.keys(dots).forEach(function (id) { from[id] = [+dots[id].getAttribute('cx'), +dots[id].getAttribute('cy')]; });
        if (!ms) { Object.keys(dots).forEach(function (id) { var t = dots[id]._to; if (t) { dots[id].setAttribute('cx', t[0]); dots[id].setAttribute('cy', t[1]); } }); return; }
        tween(box, ms, function (u) {
          var e = ease(u);
          Object.keys(dots).forEach(function (id) { var t = dots[id]._to, f = from[id]; if (!t) return; dots[id].setAttribute('cx', f[0] + (t[0] - f[0]) * e); dots[id].setAttribute('cy', f[1] + (t[1] - f[1]) * e); });
        });
      }
      function swaps(a, b) { var n = 0; Object.keys(b).forEach(function (id) { if ((b[id] || 0) !== (a[id] || 0) && b[id] !== 0) n++; }); return n; }
      var p0 = state(0, true, null);
      if (k === 0) { place(0, {}, 0); info.textContent = 'week of ' + weeks[0].date + ': ' + weeks[0].id.length + ' stocks by forecast'; return; }
      if (k === 1 || !play) {
        if (k === 1) { place(0, {}, 0); place(0, p0, play ? 900 : 0); info.textContent = 'week of ' + weeks[0].date; }
        if (k >= 2 && !play) {                                        // last frame: week 16, buffered
          var pb = p0, pe = state(0, false, null);
          for (var wi = 1; wi < weeks.length; wi++) { pb = state(wi, true, pb); pe = state(wi, false, pe); }
          place(weeks.length - 1, pb, 0);
          info.textContent = 'week ' + weeks.length + ' of the test, buffered';
        }
        return;
      }
      // k === 2 and playing: step through the weeks with the buffer, count names that enter a leg
      place(0, p0, 0);
      var pb2 = p0, pe2 = state(0, false, null), wi2 = 1, sumB = 0, sumE = 0;
      function nextWeek() {
        if (wi2 >= weeks.length) { info.textContent = 'avg names entering a leg each week: ' + Math.round(sumE / (weeks.length - 1)) + ' plain, ' + Math.round(sumB / (weeks.length - 1)) + ' buffered'; return; }
        var nb = state(wi2, true, pb2), ne = state(wi2, false, pe2);
        sumB += swaps(pb2, nb); sumE += swaps(pe2, ne);
        info.textContent = 'week ' + (wi2 + 1) + ': ' + swaps(pe2, ne) + ' names enter a leg plain, ' + swaps(pb2, nb) + ' buffered';
        place(wi2, nb, 650); pb2 = nb; pe2 = ne; wi2++;
        var t = setTimeout(nextWeek, 800); (box._stops = box._stops || []).push(function () { clearTimeout(t); });
      }
      var t0 = setTimeout(nextWeek, 400); (box._stops = box._stops || []).push(function () { clearTimeout(t0); });
    },
    enter: function (box, k) { this.draw(box, k, false); box._k = k; },
    step: function (box, k) { if (box._k === k) return; box._k = k; this.draw(box, k, true); },
    leave: function (box) { halt(box); },
    still: function (box) { this.draw(box, 2, false); }
  };

  // ------------------------------------------------------------------ B4 three checks after the class's ladder
  C.ladder = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, F = D().facts;
    var rungs = [
      ['1', 'Price features as controls', 'Text significantly better in ' + F.text_vs_price.better + ' of ' + F.text_vs_price.pairs + ' paired comparisons;', 'LDA news misdated by ' + F.placebo.days + ' trading days does as well.'],
      ['2', 'Three-factor alpha', F.alpha.above + ' of ' + F.alpha.portfolios + ' decile long-shorts significantly positive; ' + F.alpha.below + ' significantly negative.', 'Five-factor alpha: a next step.'],
      ['3', 'Long-short net Sharpe against the index', '', '']];
    rungs.forEach(function (r, i) {
      var y = 14 + i * 104, gg = reveal(box, el('g', {}, s), k, i + 1, animate, 600);
      el('circle', {cx: 22, cy: y + 14, r: 22, fill: 'none', stroke: COL.ink, 'stroke-width': 2}, gg);
      tx(gg, 22, y + 22, r[0], {'text-anchor': 'middle', 'font-size': 22, class: 't-mono'});
      tx(gg, 70, y + 24, r[1], {'font-size': 30, 'font-weight': 500});
      if (r[2]) tx(gg, 70, y + 58, r[2], {'font-size': 21, class: 't-mono'});
      if (r[3]) tx(gg, 70, y + 86, r[3], {'font-size': 21, class: 't-mono t-quiet'});
    });
    var gI = reveal(box, el('g', {}, s), k, 3, animate, 700), X = lin(-2.5, 2.2, 400, 1290), base = 262;
    [-2, -1, 0, 1, 2].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: base + 36, y2: base + 280, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, gI);
      tx(gI, X(v), base + 306, v ? signed(v, 0) : '0', {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'});
    });
    tx(gI, 70, base + 332, 'grey tick: the value-weighted index, same months', {'font-size': 18, class: 't-mono t-quiet'});
    tx(gI, X(2.2), base + 332, 'net Sharpe, test, 95% interval', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    var idx = F.index_vw, codes = ['M', '2W', 'W'];
    [[F.headlines.rows, 'chosen on validation (the headline)', 26, true], [F.protocol.rows, 'protocol fixed in advance: 25 ensembles, chosen on validation, buffered', 166, false]].forEach(function (grp) {
      tx(gI, 70, base + grp[2], grp[1], {'font-size': 17, class: 't-mono' + (grp[3] ? '' : ' t-quiet'), stroke: '#fff', 'stroke-width': 6, 'stroke-linejoin': 'round', 'paint-order': 'stroke'});
      grp[0].forEach(function (h, i) {
        var y = base + grp[2] + 28 + i * 36, solid = grp[3];
        tx(gI, 70, y + 7, h.freq, {'font-size': 20, class: solid ? null : 't-quiet'});
        el('line', {x1: X(h.lo), x2: X(h.hi), y1: y, y2: y, stroke: solid ? COL.ink : COL.grey, 'stroke-width': 2.5, 'stroke-dasharray': solid ? null : '7 5'}, gI);
        el('line', {x1: X(idx[codes[i]]), x2: X(idx[codes[i]]), y1: y - 12, y2: y + 12, stroke: COL.mist, 'stroke-width': 3}, gI);
        el('circle', {cx: X(h.net), cy: y, r: 7, fill: solid ? COL.ink : '#fff', stroke: solid ? null : COL.grey, 'stroke-width': solid ? 0 : 2.5}, gI);
        tx(gI, X(h.hi) + 12, y + 7, signed(h.net), {'font-size': 18, class: 't-mono' + (solid ? '' : ' t-quiet')});
      });
    });
  });

  // ------------------------------------------------------------------ B5 validation's favorites, then the search
  C.vdrop = {
    draw: function (box, k, play) {
      halt(box);
      var g = surface(box), s = g.s, P = D().validation.panels, Sr = D().search;
      var ext = 0;
      P.forEach(function (p) { p.pts.forEach(function (q) { ext = Math.max(ext, Math.abs(q[0]), Math.abs(q[1])); }); });
      ext += 0.004;
      var pw = 260, gap = 50, top = 62, H = 260;
      var dotsAll = [];
      P.forEach(function (p, i) {
        var x0 = i * (pw + gap), X = lin(-ext, ext, x0, x0 + pw), Y = lin(-ext, ext, top + H, top);
        el('rect', {x: x0, y: top, width: pw, height: H, fill: 'none', stroke: COL.rule}, s);
        el('line', {x1: X(0), x2: X(0), y1: top, y2: top + H, class: 'ln-rule'}, s);
        el('line', {x1: x0, x2: x0 + pw, y1: Y(0), y2: Y(0), class: 'ln-rule'}, s);
        el('line', {x1: X(-ext), y1: Y(-ext), x2: X(ext), y2: Y(ext), stroke: COL.mist, 'stroke-dasharray': '6 6', 'stroke-width': 1.5}, s);
        tx(s, x0, top - 34, p.name + ' · ' + p.n + ' experiments', {'font-size': 19, class: 't-mono'});
        tx(s, x0, top - 10, 'rank corr ' + signed(p.rho), {'font-size': 19, class: 't-mono t-quiet'});
        var favs = [];
        p.pts.forEach(function (q) {
          var d = el('circle', {cx: X(q[0]), cy: Y(q[0]), r: q[2] ? 6 : 3.4, fill: q[2] ? COL.cobalt : '#B8B8B5',
            stroke: q[3] ? COL.ink : 'none', 'stroke-width': q[3] ? 1.4 : 0}, s);
          d._a = Y(q[0]); d._b = Y(q[1]); (q[2] ? favs : dotsAll).push(d);
        });
        favs.forEach(function (d) { s.appendChild(d); dotsAll.push(d); });
        reveal(box, tx(s, x0, top + H + 30, 'top ten: test IC ' + signed(p.fav, 3), {'font-size': 19, class: 't-mono t-cobalt'}), k, 1, play, 1600);
      });
      var lx = 3 * pw + 2 * gap + 40, leg = [['blue: validation’s', 't-cobalt'], ['ten favorites', 't-cobalt'], null, ['ringed: EBM, the model', ''], ['validation flatters most', ''], null,
        ['dashed: test = validation', 't-quiet'], ['across: validation IC', 't-quiet'], ['up: test IC', 't-quiet'], ['rank corr: validation', 't-quiet'], ['IC vs test IC', 't-quiet']];
      var ly = top + 14;
      leg.forEach(function (l) { if (!l) { ly += 14; return; } tx(s, lx, ly, l[0], {'font-size': 18, class: 't-mono ' + l[1]}); ly += 24; });
      function drop(u) { dotsAll.forEach(function (d) { d.setAttribute('cy', d._a + (d._b - d._a) * u); }); }
      if (k >= 1) { if (play && k === 1) tween(box, 1600, function (u) { drop(ease(u)); }); else drop(1); }

      // the search: the best of 375 tries against the best of 375 with no skill, per trading rule
      var gS = reveal(box, el('g', {}, s), k, 2, play, 700), y0 = 398, Xs = lin(-1.4, 0.8, 300, 1180);
      tx(gS, 0, y0, 'Search: best net Sharpe of 375 tries minus the best of 375 with no skill · 7 trading rules', {'font-size': 19, class: 't-mono'});
      [-1, -0.5, 0, 0.5].forEach(function (v) {
        el('line', {x1: Xs(v), x2: Xs(v), y1: y0 + 18, y2: y0 + 122, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, gS);
        tx(gS, Xs(v), y0 + 146, v ? signed(v, 1) : '0', {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'});
      });
      ['M', '2W', 'W'].forEach(function (f, fi) {
        var y = y0 + 40 + fi * 32, rs = Sr.rows.filter(function (r) { return r.freq === f; });
        var below = rs.filter(function (r) { return r.best < r.luck; }).length;
        tx(gS, 0, y + 7, ['monthly', 'biweekly', 'weekly'][fi], {'font-size': 20});
        rs.forEach(function (r) { el('circle', {cx: Xs(r.best - r.luck), cy: y, r: 7, fill: r.best < r.luck ? '#B8B8B5' : COL.ink, stroke: '#fff', 'stroke-width': 1.5}, gS); });
        tx(gS, Xs(0.8) + 24, y + 7, below + ' of ' + rs.length + ' below', {'font-size': 19, class: 't-mono'});
      });
      tx(gS, Xs(0.8) + 24, y0 + 146, Sr.short + ' of ' + Sr.rows.length + ' below', {'font-size': 19, class: 't-mono', 'font-weight': 500});
    },
    enter: function (box, k) { this.draw(box, k, false); box._k = k; },
    step: function (box, k) { if (box._k === k) return; box._k = k; this.draw(box, k, true); },
    leave: function (box) { halt(box); },
    still: function (box) { this.draw(box, 2, false); }
  };

  // ------------------------------------------------------------------ B1 the presidents strip (Table A3)
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
    tx(s, X(7) + 40, 600, 'the paper’s one exception: “Bush” implies president (it saw George H. W. Bush)', {'text-anchor': 'end', 'font-size': 17, class: 't-mono t-red'});
    tx(s, 0, 40, 'P(“president”) in “XXX is a United States <mask>.”, ordered by year taking office', {'font-size': 20, class: 't-mono'});
  });

})();
