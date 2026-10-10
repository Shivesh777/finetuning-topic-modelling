/* Exhibits, part C: section 4 (how it works) and section 5 (results). Same conventions as parts A and B:
   every number is read from window.DATA (tools/build_data.py); drawings marked "schematic" carry no data. */
(function () {
  'use strict';
  var U = window.DECKUTIL, C = window.CHARTS, el = U.el, surface = U.surface, lin = U.lin, ease = U.ease,
      tween = U.tween, halt = U.halt, reveal = U.reveal, pct = U.pct, signed = U.signed, COL = U.COL, MON = U.MON;
  var D = function () { return window.DATA; };
  var SOFT = 'rgba(44,75,239,.10)', SHADE = {M: '#BDBDBA', '2W': '#7E7E7B', W: COL.ink}, FN = {M: 'Monthly', '2W': 'Biweekly', W: 'Weekly'};
  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); box._k = -1; build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; var from = box._k; box._k = k; build(box, k, true, from); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k, false); }
    };
  }
  // deck.css colours svg text with a rule, which beats a fill attribute: a text colour goes in its style
  function styled(a) {
    if (!a || !a.fill) return a;
    a = Object.assign({}, a); a.style = 'fill:' + a.fill + (a.style ? ';' + a.style : ''); delete a.fill;
    return a;
  }
  function tx(parent, x, y, s, a) { return U.tx(parent, x, y, s, styled(a)); }
  // text with sub- and superscripts: 'π_{a,k}', 'x^{2}'
  function mtx(parent, x, y, s, a) {
    var t = el('text', Object.assign({x: x, y: y}, styled(a) || {}), parent), fs = +((a && a['font-size']) || 20), shift = 0;
    var re = /([_^])\{([^}]*)\}/g, last = 0, m;
    function add(str, kind) {
      if (!str) return;
      var to = kind === '_' ? fs * 0.3 : kind === '^' ? -fs * 0.4 : 0;
      var sp = el('tspan', {dy: to - shift ? (to - shift).toFixed(1) : null, 'font-size': kind ? Math.round(fs * 0.68) : null}, t);
      sp.textContent = str; shift = to;
    }
    while ((m = re.exec(s))) { add(s.slice(last, m.index), ''); add(m[2], m[1]); last = re.lastIndex; }
    add(s.slice(last), '');
    return t;
  }
  function arrow(g, x1, y1, x2, y2, a) {
    a = a || {};
    var col = a.stroke || COL.ink, hs = a.head || 9, ang = Math.atan2(y2 - y1, x2 - x1);
    el('line', {x1: x1, y1: y1, x2: x2 - Math.cos(ang) * hs * 0.6, y2: y2 - Math.sin(ang) * hs * 0.6, stroke: col, 'stroke-width': a.w || 2, 'stroke-dasharray': a.dash || null}, g);
    var p = [[x2, y2], [x2 - hs * Math.cos(ang - 0.45), y2 - hs * Math.sin(ang - 0.45)], [x2 - hs * Math.cos(ang + 0.45), y2 - hs * Math.sin(ang + 0.45)]];
    return el('path', {d: 'M' + p.map(function (q) { return q[0].toFixed(1) + ' ' + q[1].toFixed(1); }).join('L') + 'Z', fill: col}, g);
  }
  function rand(i) { var x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); }   // schematic only
  function fit(words, n) {                       // the first top words that fit in n characters (at least one)
    var out = '', full = false;
    words.split(', ').forEach(function (w) { var t = out ? out + ', ' + w : w; if (!full && (!out || t.length <= n)) out = t; else full = true; });
    return out;
  }
  function big(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function ic3(v) { return Math.abs(v) < 0.0005 ? '0.000' : signed(v, 3); }          // no '−0.000'
  var SERIF = {class: 't-serif', 'font-style': 'italic'};
  function serif(a) { return Object.assign({}, SERIF, a); }

  // ------------------------------------------------------------------ 4.1 end to end
  C.flow = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, P = D().pipeline;
    var nodes = [
      ['News', 'article a', big(P.articles) + ' articles', '2012 to 2024'],
      ['Topics', 'π_{a,k}', 'weight of article a', 'on topic k'],
      ['Attention', 'θ_{k,d}', 'topic k’s share of', 'day d’s news'],
      ['Shock', 's_{k,d}', 'attention minus its', 'recent mean'],
      ['Features', 'z_{i,t}', 'exposures, own news,', 'price; ranked'],
      ['Forecast', 'ŷ_{i,t+1}', 'eight models and', 'their ensemble'],
      ['Portfolio', 'w_{i,t}', 'long-short, buffer,', 'tilt; net of costs']];
    var y = 30, xs = nodes.map(function (_, i) { return 98 + i * 197; });
    var line = el('line', {x1: xs[0], y1: y, x2: xs[0], y2: y, stroke: COL.mist, 'stroke-width': 2}, s);
    var gs = nodes.map(function (n, i) {
      var gg = el('g', {opacity: 0}, s), x = xs[i];
      el('circle', {cx: x, cy: y, r: 13, fill: i === 0 || i === 6 ? COL.cobalt : COL.ink}, gg);
      tx(gg, x, y + 52, n[0], {'text-anchor': 'middle', 'font-size': 26, 'font-weight': 600});
      mtx(gg, x, y + 88, n[1], serif({'text-anchor': 'middle', 'font-size': 24, fill: COL.cobalt}));
      tx(gg, x, y + 120, n[2], {'text-anchor': 'middle', 'font-size': 18, class: 't-quiet'});
      tx(gg, x, y + 144, n[3], {'text-anchor': 'middle', 'font-size': 18, class: 't-quiet'});
      if (i) arrow(gg, xs[i] - 66, y, xs[i] - 22, y, {stroke: COL.mist, w: 2});
      return gg;
    });
    function grow(u) {
      var x = xs[0] + (xs[6] - xs[0]) * u;
      line.setAttribute('x2', x);
      gs.forEach(function (n, i) { n.setAttribute('opacity', x >= xs[i] - 1 ? 1 : 0); });
    }
    if (k >= 1) { if (animate && k === 1) tween(box, 2200, function (u) { grow(u); }); else grow(1); }

    // the protocol
    var gT = reveal(box, el('g', {}, s), k, 2, animate, 800), X = lin(2012, 2025, 230, 1370), r0 = 250;
    for (var yr = 2012; yr <= 2024; yr++) {
      tx(gT, (X(yr) + X(yr + 1)) / 2, r0, String(yr), {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
      el('line', {x1: X(yr), x2: X(yr), y1: r0 + 10, y2: r0 + 222, class: 'ln-rule'}, gT);
    }
    var rows = [['Sample', r0 + 24], ['Topic models', r0 + 84], ['Forecasters', r0 + 134], ['Reported', r0 + 184]];
    rows.forEach(function (r) { tx(gT, 0, r[1] + 26, r[0], {'font-size': 21}); });
    var y1 = rows[0][1];
    el('rect', {x: X(2013), y: y1, width: X(2020) - X(2013), height: 40, fill: SOFT}, gT);
    tx(gT, (X(2013) + X(2020)) / 2, y1 + 27, 'train 2013–2019', {'text-anchor': 'middle', 'font-size': 20, class: 't-cobalt'});
    el('rect', {x: X(2020), y: y1, width: X(2022) - X(2020), height: 40, fill: '#E6E6E3'}, gT);
    tx(gT, (X(2020) + X(2022)) / 2, y1 + 27, 'validation', {'text-anchor': 'middle', 'font-size': 20});
    el('rect', {x: X(2022) + 1, y: y1 + 1, width: X(2025) - X(2022) - 2, height: 38, fill: '#fff', stroke: COL.ink, 'stroke-width': 2}, gT);
    tx(gT, (X(2022) + X(2025)) / 2, y1 + 27, 'test 2022–2024', {'text-anchor': 'middle', 'font-size': 20, 'font-weight': 600});
    var y2 = rows[1][1] + 6;
    el('rect', {x: X(2012), y: y2, width: X(2020) - X(2012), height: 30, fill: '#D9D9D5'}, gT);
    tx(gT, (X(2012) + X(2020)) / 2, y2 + 22, 'fitted on text to 2019', {'text-anchor': 'middle', 'font-size': 18});
    el('line', {x1: X(2020) + 4, x2: X(2025), y1: y2 + 15, y2: y2 + 15, stroke: COL.mist, 'stroke-width': 3, 'stroke-dasharray': '8 6'}, gT);
    tx(gT, (X(2020) + X(2025)) / 2, y2 + 8, 'applied, unchanged', {'text-anchor': 'middle', 'font-size': 17, class: 't-quiet'});
    var y3 = rows[2][1] + 6;
    el('rect', {x: X(2013), y: y3, width: X(2020) - X(2013), height: 30, fill: '#D9D9D5'}, gT);
    tx(gT, (X(2013) + X(2020)) / 2, y3 + 22, 'fitted', {'text-anchor': 'middle', 'font-size': 18});
    el('rect', {x: X(2020), y: y3, width: X(2022) - X(2020), height: 30, fill: COL.cobalt}, gT);
    tx(gT, (X(2020) + X(2022)) / 2, y3 + 22, 'tuned, chosen', {'text-anchor': 'middle', 'font-size': 18, fill: '#fff'});
    el('line', {x1: X(2022) + 4, x2: X(2025), y1: y3 + 15, y2: y3 + 15, stroke: COL.mist, 'stroke-width': 3, 'stroke-dasharray': '8 6'}, gT);
    tx(gT, (X(2022) + X(2025)) / 2, y3 + 8, 'refit to 2021, forecast', {'text-anchor': 'middle', 'font-size': 17, class: 't-quiet'});
    var y4 = rows[3][1] + 6, per = P.periods;
    el('rect', {x: X(2022), y: y4, width: X(2025) - X(2022), height: 30, fill: COL.ink}, gT);
    tx(gT, (X(2022) + X(2025)) / 2, y4 + 22, 'net returns, rank IC', {'text-anchor': 'middle', 'font-size': 18, fill: '#fff'});
    tx(gT, X(2022) - 14, y4 + 22, 'test periods: ' + per.M[2] + ' months · ' + per['2W'][2] + ' fortnights · ' + per.W[2] + ' weeks',
       {'text-anchor': 'end', 'font-size': 17, class: 't-mono t-quiet'});
    tx(gT, 0, r0 + 268, 'News of day d is used from day d + 1 · universe: the 1,000 largest stocks each June · features use data up to formation',
       {'font-size': 17, class: 't-mono t-quiet'});
    tx(gT, 0, r0 + 296, 'Months with incomplete return data are left out · a stock with a missing feature gets another stock’s row at random',
       {'font-size': 17, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 4.2 four topic readers
  C.readers = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, R = D().readers.rows;
    var subs = ['counts its words', 'embeds the sentences', 'embeds them, in 1976–95 English', 'classifies the headline'];
    var math = [['π_{a} ~ Dirichlet(α)', 'each word: z ~ π_{a}, w ~ φ_{z}'],
                ['e_{a} = MiniLM(headline + lead)', 'π_{a,k} ∝ exp(cos(e_{a}, c_{k}) / τ)'],
                ['e_{a} = NoLBERT(headline + lead)', 'π_{a,k} ∝ exp(cos(e_{a}, c_{k}) / τ)'],
                ['π_{a,k} = P(class k | headline)', 'twenty classes, set by its authors']];
    var facts = [['fitted on our text to 2019', ''], ['encoder published 2021;', 'k-means on our text to 2019'],
                 ['encoder trained on 1976–95 text,', 'adapted on news by our team'], ['fine-tuned on finance tweets,', 'published Dec 2022']];
    R.forEach(function (r, i) {
      var x0 = i * 350, gg = reveal(box, el('g', {}, s), k, 1, animate && k === 1, 500 + i * 250), cx = x0 + 165;
      tx(gg, x0, 26, r.name, {'font-size': 30, 'font-weight': 600});
      tx(gg, x0 + 330, 26, (i === 2 ? r.topics + ' and ' + D().readers.nolbert20.topics : r.topics + (i === 3 ? ' classes' : ' topics')), {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
      tx(gg, x0, 56, subs[i], {'font-size': 19, class: 't-quiet'});
      // schematic of how it reads
      var d = el('g', {}, gg);
      if (i === 0) {
        var words = [['oil', 46], ['crude', 64], ['prices', 70], ['barrel', 66], ['output', 68], ['shares', 66]];
        words.forEach(function (w, j) {
          var wx = x0 + (j % 3) * 104, wy = 80 + Math.floor(j / 3) * 40;
          el('rect', {x: wx, y: wy, width: w[1] + 30, height: 30, rx: 6, fill: j < 5 ? SOFT : COL.wash, stroke: 'none'}, d);
          tx(d, wx + 10, wy + 21, w[0], {'font-size': 17, fill: j < 5 ? COL.cobalt : COL.grey});
          tx(d, wx + w[1] + 22, wy + 21, String([3, 2, 2, 1, 1, 1][j]), {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});
        });
        arrow(d, cx, 166, cx, 190, {stroke: COL.mist});
        var segs = [[0.62, COL.cobalt], [0.18, '#9D9D9A'], [0.12, '#C9C9C6'], [0.08, '#E2E2DF']], sx = x0;
        segs.forEach(function (q) { el('rect', {x: sx, y: 198, width: 330 * q[0] - 2, height: 24, fill: q[1]}, d); sx += 330 * q[0]; });
      } else if (i === 1 || i === 2) {
        var ox = x0 + 110, oy = 150, rr = 64, ang = [-2.6, -1.9, -1.2, -0.45, 0.35, 1.2, 2.2, 2.9];
        el('circle', {cx: ox, cy: oy, r: rr, fill: 'none', stroke: COL.rule, 'stroke-width': 2}, d);
        ang.forEach(function (a, j) { el('circle', {cx: ox + rr * Math.cos(a), cy: oy + rr * Math.sin(a), r: j === 3 ? 7 : 5, fill: j === 3 ? COL.cobalt : '#B8B8B5'}, d); });
        arrow(d, ox, oy, ox + (rr - 6) * Math.cos(-0.62), oy + (rr - 6) * Math.sin(-0.62), {stroke: COL.ink, w: 2.5});
        tx(d, ox + 6, oy + 26, 'e', {'font-size': 18, class: 't-serif', 'font-style': 'italic'});
        mtx(d, ox + (rr + 12) * Math.cos(-0.45), oy + (rr + 12) * Math.sin(-0.45) - 8, 'c_{k}', serif({'font-size': 18, fill: COL.cobalt}));
        var wts = [0.03, 0.05, 0.12, 0.62, 0.1, 0.04, 0.02, 0.02];
        wts.forEach(function (w, j) { el('rect', {x: x0 + 210 + j * 15, y: 214 - w * 150, width: 11, height: w * 150, fill: j === 3 ? COL.cobalt : '#C9C9C6'}, d); });
        el('line', {x1: x0 + 206, x2: x0 + 330, y1: 214, y2: 214, class: 'ln-axis'}, d);
      } else {
        for (var j = 0; j < 20; j++) {
          var p = j === 6 ? 0.58 : j === 1 ? 0.11 : 0.02 + 0.03 * rand(j + 3);
          el('rect', {x: x0 + j * 16.5, y: 214 - p * 220, width: 12, height: p * 220, fill: j === 6 ? COL.cobalt : '#C9C9C6'}, d);
        }
        el('line', {x1: x0, x2: x0 + 330, y1: 214, y2: 214, class: 'ln-axis'}, d);
      }
      mtx(gg, x0, 262, math[i][0], serif({'font-size': 21}));
      mtx(gg, x0, 297, math[i][1], serif({'font-size': 21}));
      tx(gg, x0, 334, facts[i][0], {'font-size': 17, class: 't-mono t-quiet'});
      if (facts[i][1]) tx(gg, x0, 358, facts[i][1], {'font-size': 17, class: 't-mono t-quiet'});
      // three of its topics: an oil topic, a distinctive one, a templated notice
      var gE = reveal(box, el('g', {}, gg), k, 2, animate, 600);
      tx(gE, x0, 398, 'three of its topics', {'font-size': 16, class: 't-mono t-quiet'});
      r.examples.forEach(function (e, j) {
        var ey = 412 + j * 50, fill = j === 0 ? SOFT : j === 1 ? COL.wash : '#fff';
        el('rect', {x: x0, y: ey, width: 330, height: 40, rx: 6, fill: fill, stroke: j === 2 ? COL.rule : 'none'}, gE);
        var lab = e.label ? e.label + ': ' : '';
        var t = tx(gE, x0 + 12, ey + 27, '', {'font-size': 18, fill: j === 0 ? COL.cobalt : j === 2 ? COL.grey : COL.ink});
        if (lab) { var b = el('tspan', {'font-weight': 600}, t); b.textContent = lab; }
        var w = el('tspan', {}, t); w.textContent = fit(e.words, 30 - lab.length);
      });
    });
  });

  // ------------------------------------------------------------------ B21 attention and its shock (one real topic)
  C.shock = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, A = D().attention, n = A.days.length;
    mtx(s, 0, 28, 'θ_{k,d} = Σ_{a∈d} π_{a,k} / N_{d}', serif({'font-size': 28}));
    tx(s, 0, 58, 'attention: topic k’s share of the day’s articles', {'font-size': 17, class: 't-mono t-quiet'});
    mtx(s, 640, 28, 's_{k,d} = θ_{k,d} − (1/w) Σ_{j=1..w} θ_{k,d−j}', serif({'font-size': 28}));
    tx(s, 640, 58, 'shock: attention minus its mean over the previous w trading days', {'font-size': 17, class: 't-mono t-quiet'});
    var X = lin(0, n - 1, 214, 1372), top = 112, bot = 292, mx = 0.09, Y = lin(0, mx, bot, top);
    tx(s, 214, 96, A.model + ' topic “' + A.words + '”: share of each trading day’s news', {'font-size': 19});
    [0, 0.03, 0.06, 0.09].forEach(function (v) {
      el('line', {x1: 214, x2: 1372, y1: Y(v), y2: Y(v), class: v ? 'ln-rule' : 'ln-axis'}, s);
      tx(s, 200, Y(v) + 6, pct(v), {'text-anchor': 'end', 'font-size': 17, class: 't-mono t-quiet'});
    });
    var mpos = [];
    A.days.forEach(function (d, i) { if (i === 0 || d.slice(5, 7) !== A.days[i - 1].slice(5, 7)) mpos.push(i); });
    mpos.forEach(function (i) {
      var m = +A.days[i].slice(5, 7);
      tx(s, X(i) + 4, bot + 24, MON[m - 1] + (m === 1 ? ' 2022' : m === 12 ? ' 2021' : ''), {'font-size': 17, class: 't-mono t-quiet'});
      el('line', {x1: X(i), x2: X(i), y1: bot, y2: bot + 8, class: 'ln-axis'}, s);
    });
    function path(vals) {
      var d = '', pen = false;
      vals.forEach(function (v, i) { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(Math.min(v, mx)).toFixed(1); pen = true; });
      return d;
    }
    var th = el('path', {d: path(A.theta), fill: 'none', stroke: COL.ink, 'stroke-width': 2.4}, s);
    if (k >= 1 && animate && k === 1) { th.setAttribute('stroke-dasharray', '4000'); th.setAttribute('stroke-dashoffset', 4000); tween(box, 1600, function (u) { th.setAttribute('stroke-dashoffset', 4000 * (1 - ease(u))); }); }
    th.setAttribute('opacity', k >= 1 ? 1 : 0);
    var pk = A.days.indexOf(A.peak.day), gP = reveal(box, el('g', {}, s), k, 1, animate, 900);
    tx(gP, X(pk) - 12, Y(A.peak.theta) + 6, '8 Mar 2022: the US bans Russian oil imports', {'text-anchor': 'end', 'font-size': 17, class: 't-mono'});
    var tr = reveal(box, el('path', {d: path(A.trail5), fill: 'none', stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '6 5'}, s), k, 2, animate, 700);
    tr.setAttribute('d', path(A.trail5));
    var gL = reveal(box, el('g', {}, s), k, 2, animate, 700);
    el('line', {x1: 1064, x2: 1094, y1: 90, y2: 90, stroke: COL.cobalt, 'stroke-width': 2, 'stroke-dasharray': '6 5'}, gL);
    tx(gL, 1372, 96, 'mean of the previous 5 days', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-cobalt'});
    // shock rows: 5 days (the main run), then the windows tuned on validation
    var rows = [['5', 'w = 5 days', 'main run, biweekly', 2], ['1', 'w = 1 day', 'tuned: weekly', 3], ['63', 'w = 63 days', 'tuned: monthly', 3]];
    rows.forEach(function (r, j) {
      var y0 = 340 + j * 78, mid = y0 + 34, H = lin(-0.05, 0.05, mid + 32, mid - 32), gR = reveal(box, el('g', {}, s), k, r[3], animate, 700);
      tx(gR, 0, mid - 4, r[1], {'font-size': 20, fill: r[3] === 2 ? COL.ink : COL.cobalt});
      tx(gR, 0, mid + 20, r[2], {'font-size': 16, class: 't-mono t-quiet'});
      el('line', {x1: 214, x2: 1372, y1: mid, y2: mid, class: 'ln-axis'}, gR);
      A.shock[r[0]].forEach(function (v, i) {
        if (v == null) return;
        var c = Math.max(-0.05, Math.min(0.05, v));
        el('rect', {x: X(i) - 4, y: Math.min(H(c), mid), width: 8, height: Math.abs(H(c) - mid), fill: v > 0 ? COL.cobalt : '#B8B8B5'}, gR);
      });
    });
  });

  // ------------------------------------------------------------------ 4.3 the features of one stock
  C.features = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, P = D().pipeline;
    var cards = [
      ['Narrative exposure', 'K per stock: one per topic', [['β_{i,k} = Σ_{d} ω_{d} (r_{i,d} − r̄_{i}) s̃_{k,d} / Σ_{d} ω_{d}', 1]],
       ['last 252 trading days, half-life 60', 's̃: the shock with the market return', 'projected out (Bybee, Kelly and Su)']],
      ['Firm news', '1 + K per stock: its own articles', [['n_{i} = log(1 + articles, last 21 days)', 1], ['q_{i,k} = share of them on topic k', 1]],
       [big(P.firm) + ' articles filed under at', 'most 3 tickers; no coverage for', '126 days: the feature is missing']],
      ['Price', '8 per stock', [],
       ['market beta · idiosyncratic volatility', '21-day volatility · largest daily return', 'momentum (12 to 1 months)', 'return of the last 21 and 5 days · size']]];
    cards.forEach(function (c, i) {
      var x0 = i * 470, gg = reveal(box, el('g', {}, s), k, 1, animate && k === 1, 500 + i * 300);
      el('rect', {x: x0, y: 0, width: 440, height: 236, fill: i === 0 ? SOFT : COL.wash}, gg);
      tx(gg, x0 + 20, 36, c[0], {'font-size': 25, 'font-weight': 600, fill: i === 0 ? COL.cobalt : COL.ink});
      tx(gg, x0 + 20, 64, c[1], {'font-size': 17, class: 't-mono t-quiet'});
      var y = 104;
      c[2].forEach(function (f) { mtx(gg, x0 + 20, y, f[0], serif({'font-size': 21})); y += 32; });
      c[3].forEach(function (l, j) { tx(gg, x0 + 20, y + 4 + j * 26, l, {'font-size': 17, class: 't-mono t-quiet'}); });
    });
    // the panel: stocks by features, ranked (schematic values)
    var gM = reveal(box, el('g', {}, s), k, 2, animate, 800), blocks = [['price', 8, COL.ink], ['exposures', 60, COL.cobalt], ['firm news', 61, '#6F6F6F']];
    var cw = 6.1, ch = 10.6, rows = 20, x0 = 0, y0 = 322, cx = x0;
    blocks.forEach(function (b, bi) {
      if (bi) tx(gM, cx, y0 - 12, b[0] + ' · ' + b[1], {'font-size': 17, class: 't-mono', fill: b[2]});
      else { tx(gM, cx, y0 - 36, b[0] + ' · ' + b[1], {'font-size': 17, class: 't-mono', fill: b[2]}); el('line', {x1: cx + 4, x2: cx + 4, y1: y0 - 30, y2: y0 - 4, stroke: COL.ink}, gM); }
      for (var c = 0; c < b[1]; c++) {
        for (var r = 0; r < rows; r++) {
          var v = rand(bi * 1000 + c * 37 + r * 101) * 2 - 1;
          el('rect', {x: cx + c * cw, y: y0 + r * ch, width: cw - 0.8, height: ch - 0.8, fill: b[2], 'fill-opacity': (0.08 + 0.8 * Math.abs(v)).toFixed(2)}, gM);
        }
      }
      cx += b[1] * cw + 14;
    });
    tx(gM, 0, y0 + rows * ch + 26, 'one row per stock (about 1,000), one column per feature, one panel per rebalancing date · schematic values',
       {'font-size': 16, class: 't-mono t-quiet'});
    arrow(gM, cx + 6, y0 + rows * ch / 2, cx + 56, y0 + rows * ch / 2, {stroke: COL.mist});
    var tx0 = cx + 72;
    tx(gM, tx0, y0 + 6, 'Every period, every feature is ranked', {'font-size': 20});
    tx(gM, tx0, y0 + 32, 'across the stocks of that date:', {'font-size': 20});
    mtx(gM, tx0, y0 + 74, 'z = 2 (rank − 1) / (N − 1) − 1  ∈ [−1, 1]', serif({'font-size': 23}));
    tx(gM, tx0, y0 + 116, 'Target: next period’s return minus the', {'font-size': 20});
    tx(gM, tx0, y0 + 142, 'risk-free rate and the stocks’ average', {'font-size': 20});
    tx(gM, tx0, y0 + 178, 'Sets: price · exposures · firm news · text', {'font-size': 17, class: 't-mono t-quiet'});
    tx(gM, tx0, y0 + 202, '(both) · all (price and text): up to ' + P.features_all_lda, {'font-size': 17, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 4.4 the forecasters
  C.models = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    var cards = [
      ['Linear', 'OLS · ridge · lasso', ['ŷ = z′b', 'ridge: min Σ (y − z′b)^{2} + λ‖b‖^{2}', 'lasso: min Σ (y − z′b)^{2} + λ‖b‖_{1}'],
       ['λ: a path of 25 (ridge) or 40 (lasso)', 'values, from heavy to light penalty']],
      ['Boosted trees', 'XGBoost', ['ŷ = Σ_{m} η f_{m}(z)', 'tree m fits what trees 1 … m−1 missed'],
       ['depth 1, 2, 3 or 5 · η = 0.05 · half the', 'rows and columns per tree; stop at', 'the lowest validation error (≤ 600)']],
      ['Random forest', 'many deep trees, averaged', ['ŷ = (1/B) Σ_{b} f_{b}(z)', 'each tree: a random sample of rows,', 'a random √p features at each split'],
       ['100 trees · depth 3, 6 or 10', 'at least 50 stocks per leaf']]];
    cards.forEach(function (c, i) {
      var x0 = i * 470, gg = reveal(box, el('g', {}, s), k, 1, animate && k === 1, 500 + i * 300), vis = el('g', {}, gg);
      tx(gg, x0, 28, c[0], {'font-size': 27, 'font-weight': 600});
      tx(gg, x0, 56, c[1], {'font-size': 18, class: 't-mono t-quiet'});
      // schematic of the model
      if (i === 0) {
        el('rect', {x: x0, y: 74, width: 230, height: 120, fill: COL.wash}, vis);
        for (var j = 0; j < 26; j++) {
          var px = x0 + 12 + rand(j) * 206, py = 74 + 100 - (px - x0) * 0.3 - 20 + (rand(j + 50) - 0.5) * 60;
          el('circle', {cx: px, cy: Math.max(80, Math.min(188, py)), r: 3.6, fill: '#9D9D9A'}, vis);
        }
        el('line', {x1: x0 + 6, y1: 168, x2: x0 + 224, y2: 102, stroke: COL.cobalt, 'stroke-width': 3}, vis);
      } else {
        var n = 4, w = 98;
        for (var t = 0; t < n; t++) {
          var tx0 = x0 + t * (w + (i === 1 ? 14 : 8)), deep = i === 2;
          var nodes = deep ? [[0.5, 0], [0.25, 1], [0.75, 1], [0.12, 2], [0.38, 2], [0.62, 2], [0.88, 2]] : [[0.5, 0], [0.3, 1], [0.7, 1]];
          var ny = function (l) { return 86 + l * 42; };
          nodes.forEach(function (nd, q) {
            if (q) { var par = nodes[Math.floor((q - 1) / 2)]; el('line', {x1: tx0 + par[0] * w, y1: ny(par[1]), x2: tx0 + nd[0] * w, y2: ny(nd[1]), stroke: '#9D9D9A', 'stroke-width': 1.6}, vis); }
          });
          nodes.forEach(function (nd) { el('circle', {cx: tx0 + nd[0] * w, cy: ny(nd[1]), r: 6, fill: nd[1] === (deep ? 2 : 1) ? COL.cobalt : COL.ink}, vis); });
          if (i === 1 && t < n - 1) tx(vis, tx0 + w + 7, 126, '+', {'text-anchor': 'middle', 'font-size': 24, class: 't-quiet'});
        }
        if (i === 2) { el('path', {d: 'M' + x0 + ' 190 Q' + (x0 + 212) + ' 214 ' + (x0 + 424) + ' 190', fill: 'none', stroke: COL.mist, 'stroke-width': 2}, vis); tx(vis, x0 + 212, 222, 'average', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'}); }
      }
      var y = 254;
      c[2].forEach(function (f, j) { mtx(gg, x0, y, f, serif({'font-size': j ? 20 : 24})); y += j ? 30 : 36; });
      c[3].forEach(function (l, j) { tx(gg, x0, y + 6 + j * 25, l, {'font-size': 17, class: 't-mono t-quiet'}); });
    });
    var gB = reveal(box, el('g', {}, s), k, 2, animate, 800), yb = 470;
    el('line', {x1: 0, x2: 1380, y1: yb - 30, y2: yb - 30, class: 'ln-rule'}, gB);
    mtx(gB, 0, yb + 4, 'Ensemble:  ŷ^{ens}_{i} = (1/8) Σ_{m} rank_{m}(ŷ_{i})', serif({'font-size': 23}));
    tx(gB, 0, yb + 36, 'the average of the eight models’ forecast ranks', {'font-size': 17, class: 't-mono t-quiet'});
    tx(gB, 700, yb + 4, 'Ranked target: each model also fitted on', {'font-size': 20});
    tx(gB, 700, yb + 32, 'the cross-sectional rank of the return', {'font-size': 20});
    tx(gB, 700, yb + 62, 'IPCA and sparse IPCA: last time · EBM: next', {'font-size': 17, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 4.5 the EBM
  C.ebm = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, E = D().ebm;
    mtx(s, 0, 34, 'ŷ_{i} = β_{0} + f_{1}(z_{i1}) + f_{2}(z_{i2}) + … + f_{p}(z_{ip}) + Σ f_{jk}(z_{ij}, z_{ik})', serif({'font-size': 32}));
    tx(s, 0, 68, 'one curve per feature, a few pairwise surfaces; the forecast is their sum, so each term can be read', {'font-size': 18, class: 't-mono t-quiet'});
    var steps = [['Score every feature, in a shuffled order', 'first rounds round-robin, then boost the most useful'],
                 ['Fit a two-leaf tree on that feature', 'to what the model still gets wrong'],
                 ['Add it, shrunk by the learning rate', '0.02: many small steps, up to 2,000 rounds'],
                 ['Stop at the lowest validation error', 'average four bagged copies'],
                 ['Optionally add ten pairwise terms', 'validation decides; read every f off']];
    var gS = reveal(box, el('g', {}, s), k, 1, animate, 700);
    steps.forEach(function (st, i) {
      var y = 118 + i * 74;
      el('circle', {cx: 16, cy: y - 7, r: 15, fill: 'none', stroke: COL.cobalt, 'stroke-width': 2}, gS);
      tx(gS, 16, y - 1, String(i + 1), {'text-anchor': 'middle', 'font-size': 17, class: 't-mono t-cobalt'});
      tx(gS, 46, y, st[0], {'font-size': 21});
      tx(gS, 46, y + 26, st[1], {'font-size': 17, class: 't-mono t-quiet'});
    });
    // four real shape functions, grown as boosting would
    var S = E.shapes_bp, lo = Infinity, hi = -Infinity;
    S.forEach(function (t) { t.bins.forEach(function (b) { lo = Math.min(lo, b[2]); hi = Math.max(hi, b[2]); }); });
    var pad = (hi - lo) * 0.12, cw = 330, chh = 120, gx = 40, base = 640;
    var gH = reveal(box, el('g', {}, s), k, 2, animate, 500);
    tx(gH, base, 104, 'Biweekly EBM on price and FinBERT text: four of its curves', {'font-size': 19});
    tx(gH, base, 128, 'x: the feature’s rank, −1 to 1 · y: basis points of return, dashed = 0', {'font-size': 16, class: 't-mono t-quiet'});
    var paths = [];
    S.forEach(function (t, i) {
      var c = i % 2, r = Math.floor(i / 2), x0 = base + c * (cw + gx), y0 = 152 + r * 196;
      var X = lin(-1, 1, x0, x0 + cw), Y = lin(lo - pad, hi + pad, y0 + 34 + chh, y0 + 34);
      tx(gH, x0, y0 + 20, t.name, {'font-size': 18, fill: /^exposure/.test(t.name) ? COL.cobalt : COL.ink});
      el('rect', {x: x0, y: y0 + 34, width: cw, height: chh, fill: 'none', stroke: COL.rule}, gH);
      el('line', {x1: x0, x2: x0 + cw, y1: Y(0), y2: Y(0), stroke: COL.mist, 'stroke-dasharray': '4 4'}, gH);
      tx(gH, x0 - 6, Y(0) + 5, '0', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});
      var p = el('path', {fill: 'none', stroke: /^exposure/.test(t.name) ? COL.cobalt : COL.ink, 'stroke-width': 2.4}, gH);
      paths.push([p, t, X, Y]);
      [-10, 10].forEach(function (v) { if (Y(v) > y0 + 34 && Y(v) < y0 + 34 + chh) { el('line', {x1: x0 - 4, x2: x0, y1: Y(v), y2: Y(v), class: 'ln-axis'}, gH); tx(gH, x0 - 6, Y(v) + 5, signed(v, 0), {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'}); } });
    });
    function draw(u) {
      paths.forEach(function (q) {
        var d = '';
        q[1].bins.forEach(function (b, j) { var yy = q[3](b[2] * u); d += (j ? 'L' : 'M') + q[2](b[0]).toFixed(1) + ' ' + yy.toFixed(1) + 'L' + q[2](b[1]).toFixed(1) + ' ' + yy.toFixed(1); });
        q[0].setAttribute('d', d);
      });
    }
    if (k >= 2 && animate && k === 2) tween(box, 2200, function (u) { draw(ease(u)); }); else draw(k >= 2 ? 1 : 0);
    var gW = reveal(box, el('g', {}, s), k, 3, animate, 800);
    el('rect', {x: 0, y: 468, width: 600, height: 102, fill: SOFT}, gW);
    tx(gW, 18, 500, 'We tried it on every feature set and frequency', {'font-size': 21, fill: COL.cobalt});
    tx(gW, 18, 530, E.fits + ' EBMs fitted. Best validation IC of the eight', {'font-size': 17, class: 't-mono'});
    tx(gW, 18, 556, 'models at every frequency: ' + E.val_ic.map(function (v) { return v.toFixed(3); }).join(', '), {'font-size': 17, class: 't-mono'});
  });

  // ------------------------------------------------------------------ 4.6 from forecast to portfolio (schematic stocks)
  C.rules = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, T = D().turnover, Fr = D().frequency.rows, n = 50, X = lin(0, n - 1, 14, 846);
    var rows = [['Decile long-short', ['w_{i} = +1/N_{L}  in the top tenth', 'w_{i} = −1/N_{S}  in the bottom tenth'], ['the classic sort; rebalanced every period']],
                ['Buffered long-short', ['enter in the top (bottom) tenth,', 'hold until it leaves the top (bottom) fifth'],
                 ['weekly, one model: turnover ' + pct(T.ls_ew) + ' → ' + pct(T.ls_buf) + ' a week', 'Novy-Marx and Velikov (2016)']],
                ['Cap-weighted tilt', ['w_{i} ∝ cap_{i} (1 + s_{i})', 's_{i}: forecast rank scaled to [−1, 1]'], ['long only; judged against the index']]];
    rows.forEach(function (r, ri) {
      var y0 = ri * 178, gR = reveal(box, el('g', {}, s), k, ri + 1, animate, 700), mid = y0 + 70;
      tx(gR, 900, y0 + 30, r[0], {'font-size': 26, 'font-weight': 600, fill: ri === 1 ? COL.cobalt : COL.ink});
      r[1].forEach(function (f, j) { mtx(gR, 900, y0 + 64 + j * 35, f, serif({'font-size': 21})); });
      r[2].forEach(function (l, j) { tx(gR, 900, y0 + 134 + j * 23, l, {'font-size': 16, class: 't-mono t-quiet'}); });
      if (ri < 2) {
        if (ri === 1) {
          el('rect', {x: X(0) - 8, y: mid - 30, width: X(9) - X(0) + 16, height: 60, fill: '#ECECEA'}, gR);
          el('rect', {x: X(40) - 8, y: mid - 30, width: X(49) - X(40) + 16, height: 60, fill: SOFT}, gR);
          tx(gR, X(0) - 8, mid + 50, 'hold zone: bottom fifth', {'font-size': 15, class: 't-mono t-quiet'});
          tx(gR, X(49) + 8, mid + 50, 'hold zone: top fifth', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-cobalt'});
        }
        for (var i = 0; i < n; i++) {
          var side = i >= 45 ? 1 : i < 5 ? -1 : 0, held = ri === 1 && ((i >= 40 && i < 45 && (i === 41 || i === 43)) || (i >= 5 && i < 10 && i === 7));
          if (held) side = i >= 40 ? 1 : -1;
          el('circle', {cx: X(i), cy: mid, r: 7, fill: side === 1 ? COL.cobalt : side === -1 ? COL.ink : '#D2D2CF'}, gR);
          if (held) el('circle', {cx: X(i), cy: mid, r: 11.5, fill: 'none', stroke: side === 1 ? COL.cobalt : COL.ink, 'stroke-width': 1.6}, gR);
        }
        if (ri === 0) {
          tx(gR, X(2), mid - 22, 'short', {'text-anchor': 'middle', 'font-size': 17});
          tx(gR, X(47), mid - 22, 'long', {'text-anchor': 'middle', 'font-size': 17, class: 't-cobalt'});
          tx(gR, X(24.5), mid + 40, 'forecast rank →', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
        } else tx(gR, X(24.5), mid + 50, 'ringed: held from last period', {'text-anchor': 'middle', 'font-size': 15, class: 't-mono t-quiet'});
      } else {
        var caps = [], tot = 0, tilt = [], tt = 0;
        for (var j = 0; j < n; j++) { var c = Math.exp(1.3 * (rand(j + 200) * 2 - 1) + (rand(j + 900) > 0.93 ? 1.6 : 0)); caps.push(c); tot += c; }
        caps.forEach(function (c, j) { var sv = 2 * j / (n - 1) - 1, w = c * (1 + sv); tilt.push(w); tt += w; });
        var Hh = lin(0, 0.12, y0 + 150, y0 + 40);
        caps.forEach(function (c, j) {
          var a = c / tot, b = tilt[j] / tt;
          el('rect', {x: X(j) - 6, y: Hh(Math.min(a, 0.12)), width: 6, height: Hh(0) - Hh(Math.min(a, 0.12)), fill: '#C9C9C6'}, gR);
          el('rect', {x: X(j), y: Hh(Math.min(b, 0.12)), width: 6, height: Hh(0) - Hh(Math.min(b, 0.12)), fill: COL.cobalt}, gR);
        });
        el('line', {x1: 0, x2: 860, y1: Hh(0), y2: Hh(0), class: 'ln-axis'}, gR);
        tx(gR, 0, y0 + 30, 'grey: weight in the index · blue: tilted weight', {'font-size': 16, class: 't-mono t-quiet'});
        tx(gR, X(49), Hh(0) + 22, 'forecast rank →', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
      }
    });
  });

  // ------------------------------------------------------------------ 5.3 monthly, biweekly or weekly
  C.freq = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, R = D().frequency.rows, Xi = lin(-0.01, 0.03, 250, 600), Xs = lin(-0.6, 0.4, 680, 1150);
    var unit = {M: ' months', '2W': ' fortnights', W: ' weeks'};
    tx(s, 250, 24, 'Price features alone: test rank IC', {'font-size': 18, class: 't-mono t-quiet'});
    tx(s, 250, 46, 'bar: average of 8 models · dots: each', {'font-size': 16, class: 't-mono t-mist'});
    tx(s, 680, 24, 'Median decile long-short: Sharpe, gross → net', {'font-size': 18, class: 't-mono t-quiet'});
    tx(s, 1380, 24, 'cost a year', {'text-anchor': 'end', 'font-size': 18, class: 't-mono t-quiet'});
    var top = 74, rh = 156;
    [-0.6, -0.4, -0.2, 0, 0.2, 0.4].forEach(function (v) {
      el('line', {x1: Xs(v), x2: Xs(v), y1: top - 8, y2: top + 3 * rh - 20, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, s);
      tx(s, Xs(v), top + 3 * rh + 4, v ? signed(v, 1) : '0', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
    });
    [-0.01, 0, 0.01, 0.02, 0.03].forEach(function (v) { tx(s, Xi(v), top + 3 * rh + 4, v ? signed(v, 2) : '0', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'}); });
    el('line', {x1: Xi(0), x2: Xi(0), y1: top - 8, y2: top + 3 * rh - 20, class: 'ln-axis', 'stroke-width': 2}, s);
    R.forEach(function (r, i) {
      var c = top + i * rh + 52, wk = r.freq === 'W';
      tx(s, 0, c - 4, FN[r.freq], {'font-size': 30});
      tx(s, 0, c + 24, 'test: ' + r.periods[2] + unit[r.freq], {'font-size': 17, class: 't-mono t-quiet'});
      tx(s, 0, c + 48, 'train ' + r.periods[0] + ' · validation ' + r.periods[1], {'font-size': 16, class: 't-mono t-mist'});
      var gI = reveal(box, el('g', {}, s), k, 1, animate, 600), v = r.price_ic.test;
      var bar = el('rect', {x: Xi(0), y: c - 26, width: 0, height: 32, fill: wk ? COL.cobalt : '#BDBDBA'}, gI);
      var lab = tx(gI, Xi(v) + 10, c - 2, v.toFixed(3), {'font-size': 20, class: 't-mono'});
      r.price_ic_models.forEach(function (m) { el('circle', {cx: Xi(Math.max(-0.01, Math.min(0.03, m))), cy: c + 22, r: 4.5, fill: wk ? COL.cobalt : '#9D9D9A', 'fill-opacity': 0.75}, gI); });
      var grow = function (u) { bar.setAttribute('width', Math.max(0, Xi(v * u) - Xi(0))); };
      if (k >= 1 && animate && k === 1) { lab.setAttribute('opacity', 0); tween(box, 900, function (u) { grow(ease(u)); if (u >= 1) lab.setAttribute('opacity', 1); }); } else grow(1);
      var gS = reveal(box, el('g', {}, s), k, 2, animate, 700), a = r.ls_ew;
      arrow(gS, Xs(a.gross) - 10, c - 10, Xs(a.net) + 12, c - 10, {stroke: COL.mist, w: 2.5, head: 11});
      el('circle', {cx: Xs(a.gross), cy: c - 10, r: 9, fill: '#fff', stroke: COL.ink, 'stroke-width': 2.5}, gS);
      el('circle', {cx: Xs(a.net), cy: c - 10, r: 9, fill: COL.ink}, gS);
      tx(gS, Xs(a.gross), c - 30, 'gross ' + signed(a.gross), {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
      tx(gS, Xs(a.net), c - 30, 'net ' + signed(a.net), {'text-anchor': 'middle', 'font-size': 16, class: 't-mono'});
      tx(gS, 1380, c - 4, pct(a.cost, 1), {'text-anchor': 'end', 'font-size': 26});
      var gB = reveal(box, el('g', {}, s), k, 3, animate, 700), b = r.ls_buf;
      el('circle', {cx: Xs(b.net), cy: c + 24, r: 9, fill: COL.cobalt}, gB);
      tx(gB, Xs(b.net) + 16, c + 30, 'buffered ' + signed(b.net), {'font-size': 16, class: 't-mono t-cobalt', stroke: '#fff', 'stroke-width': 5, 'paint-order': 'stroke'});
      tx(gB, 1380, c + 26, pct(b.cost, 1) + ' buffered', {'text-anchor': 'end', 'font-size': 17, class: 't-mono t-cobalt'});
    });
  });

  // ------------------------------------------------------------------ 5.1 topic models, then the tuned shock window
  var TM_ORDER = ['lda', 'lda_all', 'minilm', 'nolbert', 'nolbert20', 'finbert'];
  C.tm = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, T = D().topics, rows = TM_ORDER.map(function (m) { return T.rows.filter(function (r) { return r.model === m; })[0]; });
    var cols = [300, 420, 540], ry = function (i) { return 150 + i * 58; };
    tx(s, 0, 24, 'Text features only: average test IC', {'font-size': 21, 'font-weight': 600});
    tx(s, 0, 50, 'exposures, firm news, both · 8 models each', {'font-size': 16, class: 't-mono t-quiet'});
    ['Monthly', 'Biweekly', 'Weekly'].forEach(function (f, j) { tx(s, cols[j], 98, f, {'text-anchor': 'middle', 'font-size': 18, class: 't-mono t-quiet'}); });
    var best = [0, 1, 2].map(function (j) { return Math.max.apply(null, rows.map(function (r) { return r.test[j]; })); });
    rows.forEach(function (r, i) {
      var y = ry(i);
      tx(s, 0, y + 7, r.name, {'font-size': 21, fill: /NoLBERT/.test(r.name) ? COL.cobalt : COL.ink});
      el('line', {x1: 0, x2: 1380, y1: y + 29, y2: y + 29, class: 'ln-rule'}, s);
    });
    var gL = reveal(box, el('g', {}, s), k, 1, animate, 700);
    rows.forEach(function (r, i) {
      var y = ry(i);
      r.test.forEach(function (v, j) {
        var top = v === best[j], w = Math.min(46, Math.abs(v) / 0.01 * 46);
        el('rect', {x: v >= 0 ? cols[j] : cols[j] - w, y: y + 12, width: w, height: 8, fill: top ? COL.cobalt : v < 0 ? '#D2D2CF' : '#9D9D9A'}, gL);
        tx(gL, cols[j], y + 4, ic3(v), {'text-anchor': 'middle', 'font-size': 19, class: 't-mono', fill: top ? COL.cobalt : v < 0 ? COL.grey : COL.ink, 'font-weight': top ? 500 : null});
      });
    });
    rows.forEach(function (r, i) { [0, 1, 2].forEach(function (j) { el('line', {x1: cols[j], x2: cols[j], y1: ry(i) + 9, y2: ry(i) + 23, stroke: COL.mist, 'stroke-width': 1}, gL); }); });
    // the tuned window: from the 5-day IC to the tuned one, same rows
    var gR = reveal(box, el('g', {}, s), k, 2, animate, 800), axes = [[660, 'Monthly: 5 → 63 days'], [1040, 'Weekly: 5 → 1 day']];
    tx(gR, 660, 24, 'Tuned shock window: average test IC', {'font-size': 21, 'font-weight': 600});
    tx(gR, 660, 50, 'exposures, text, price and text · 8 models each', {'font-size': 16, class: 't-mono t-quiet'});
    T.shock.forEach(function (sh, a) {
      var x0 = axes[a][0], X = lin(-0.015, 0.035, x0, x0 + 320);
      tx(gR, x0, 98, axes[a][1], {'font-size': 18, class: 't-mono t-quiet'});
      [-0.01, 0, 0.01, 0.02, 0.03].forEach(function (v) {
        el('line', {x1: X(v), x2: X(v), y1: ry(0) - 20, y2: ry(5) + 22, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 1.6}, gR);
        if (v === 0 || v === 0.02) tx(gR, X(v), ry(5) + 46, v ? '+0.02' : '0', {'text-anchor': 'middle', 'font-size': 15, class: 't-mono t-quiet'});
      });
      TM_ORDER.forEach(function (m, i) {
        var p = sh.topic_models.filter(function (q) { return q.model === m; })[0], y = ry(i), up = p.new > p.base, col = up ? COL.cobalt : '#9D9D9A';
        el('circle', {cx: X(p.base), cy: y, r: 5, fill: '#9D9D9A'}, gR);
        var ln = arrow(gR, X(p.base), y, X(p.new), y, {stroke: col, w: 2.5, head: 10});
        tx(gR, up ? X(p.new) + 8 : X(p.new) - 8, y + 6, ic3(p.new), {'text-anchor': up ? 'start' : 'end', 'font-size': 15, class: 't-mono', fill: col, stroke: '#fff', 'stroke-width': 5, 'paint-order': 'stroke'});
      });
      tx(gR, x0, ry(5) + 76, 'all six: ' + sh.ic[0].toFixed(3) + ' → ' + sh.ic[1].toFixed(3), {'font-size': 17, class: 't-mono'});
      tx(gR, x0, ry(5) + 100, 'median net Sharpe ' + signed(sh.sharpe_median[0]) + ' → ' + signed(sh.sharpe_median[1]), {'font-size': 17, class: 't-mono t-cobalt'});
    });
  });

  // ------------------------------------------------------------------ 5.2 the forecasters, test against validation
  var GROUP_END = {lasso: 1, sipca: 1, rf: 1, ebm: 1};
  C.forecasters = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, F = D().forecasters, R = F.rows, W = 286, xs = [262, 640, 1018], ry = function (j) { return 86 + j * 43; };
    R.forEach(function (r, j) {
      tx(s, 0, ry(j) + 7, r.name, {'font-size': 20, fill: r.model === 'ebm' ? COL.cobalt : COL.ink});
      if (GROUP_END[r.model]) el('line', {x1: 0, x2: 1364, y1: ry(j) + 21.5, y2: ry(j) + 21.5, class: 'ln-rule'}, s);
    });
    ['M', '2W', 'W'].forEach(function (f, i) {
      var x0 = xs[i], X = lin(-0.015, 0.025, x0, x0 + W), test = R.map(function (r) { return r.test[i]; });
      var top2 = test.slice().sort(function (a, b) { return b - a; }).slice(0, 2);
      tx(s, x0, 28, FN[f], {'font-size': 23, 'font-weight': 600});
      tx(s, x0 + W + 76, 28, 'test', {'text-anchor': 'end', 'font-size': 15, class: 't-mono t-quiet'});
      [-0.01, 0, 0.01, 0.02].forEach(function (v) {
        el('line', {x1: X(v), x2: X(v), y1: ry(0) - 22, y2: ry(R.length - 1) + 22, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 1.6}, s);
        tx(s, X(v), ry(R.length - 1) + 46, v ? signed(v, 2) : '0', {'text-anchor': 'middle', 'font-size': 15, class: 't-mono t-quiet'});
      });
      var gT = reveal(box, el('g', {}, s), k, 1, animate, 700), gV = reveal(box, el('g', {}, s), k, 2, animate, 700);
      R.forEach(function (r, j) {
        var v = r.test[i], y = ry(j), hot = top2.indexOf(v) >= 0, w = Math.abs(X(v) - X(0));
        el('rect', {x: v >= 0 ? X(0) : X(v), y: y - 11, width: w, height: 22, fill: hot ? COL.cobalt : '#C4C4C1'}, gT);
        tx(gT, x0 + W + 76, y + 6, ic3(v), {'text-anchor': 'end', 'font-size': 16, class: 't-mono', fill: hot ? COL.cobalt : COL.grey});
        el('circle', {cx: X(r.validation[i]), cy: y, r: 6.5, fill: '#fff', 'fill-opacity': 0.6, stroke: r.model === 'ebm' ? COL.red : COL.ink, 'stroke-width': 2.2}, gV);
      });
    });
    tx(s, 0, ry(R.length - 1) + 78, 'bar: test IC (blue: the two highest) · ring: validation IC', {'font-size': 16, class: 't-mono t-quiet'});
    var o = F.other_ranked;
    tx(s, 1364, ry(R.length - 1) + 78, 'the other ' + o.models + ' models on ranked returns: test IC ' + signed(o.test_lo, 3) + ' to ' + signed(o.test_hi, 3),
       {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 5.4 the portfolio rules, side by side
  C.strategies = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, Rr = D().rules, X = lin(-1.8, 0.8, 350, 1050), fr = ['M', '2W', 'W'];
    var ys = [80, 124, 168, 212, 256, 344, 388, 432], yIdx = 494;
    tx(s, 350, 24, 'Median test net Sharpe ratio', {'font-size': 18, class: 't-mono t-quiet'});
    fr.forEach(function (f, i) {
      el('circle', {cx: 690 + i * 120, cy: 18, r: 7, fill: SHADE[f]}, s);
      tx(s, 702 + i * 120, 24, FN[f].toLowerCase(), {'font-size': 16, class: 't-mono t-quiet'});
    });
    tx(s, 1380, 24, 'weekly: turnover · cost', {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
    tx(s, 0, 50, 'Long-short', {'font-size': 16, class: 't-mono t-quiet'});
    var gO = reveal(box, el('g', {}, s), k, 2, animate, 700);
    tx(gO, 0, 314, 'Long only', {'font-size': 16, class: 't-mono t-quiet'});
    [-1.5, -1, -0.5, 0, 0.5].forEach(function (v) {
      el('line', {x1: X(v), x2: X(v), y1: 56, y2: yIdx + 20, class: v ? 'ln-rule' : 'ln-axis', 'stroke-width': v ? 1 : 2}, s);
      tx(s, X(v), yIdx + 50, v ? signed(v, 1) : '0', {'text-anchor': 'middle', 'font-size': 16, class: 't-mono t-quiet'});
    });
    var lo = Math.min.apply(null, Rr.index_vw), hi = Math.max.apply(null, Rr.index_vw);
    el('rect', {x: X(lo), y: 56, width: X(hi) - X(lo), height: yIdx + 20 - 56, fill: 'rgba(0,0,0,.06)'}, gO);
    tx(gO, (X(lo) + X(hi)) / 2, 50, 'index', {'text-anchor': 'middle', 'font-size': 14, class: 't-mono t-quiet'});
    Rr.rows.forEach(function (r, j) {
      var y = ys[j], gg = j < 5 ? reveal(box, el('g', {}, s), k, 1, animate, 600) : el('g', {}, gO), hot = r.rule === 'ls_buf' || r.rule === 'tilt_vw';
      if (hot) el('rect', {x: 0, y: y - 19, width: 1380, height: 38, fill: r.rule === 'ls_buf' ? SOFT : 'rgba(0,0,0,.035)'}, gg);
      tx(gg, 0, y + 7, r.name, {'font-size': 20, fill: r.rule === 'ls_buf' ? COL.cobalt : COL.ink, 'font-weight': hot ? 600 : null});
      fr.forEach(function (f, i) { el('circle', {cx: X(r.net[i]), cy: y, r: 8, fill: SHADE[f], stroke: '#fff', 'stroke-width': 1.5}, gg); });
      tx(gg, 1380, y + 6, pct(r.turnover[2]) + ' · ' + pct(r.cost[2], 1), {'text-anchor': 'end', 'font-size': 17, class: 't-mono'});
    });
    var gI = el('g', {}, gO);
    tx(gI, 0, yIdx + 7, 'Index, value-weighted', {'font-size': 20, 'font-weight': 600});
    fr.forEach(function (f, i) { el('rect', {x: X(Rr.index_vw[i]) - 7, y: yIdx - 7, width: 14, height: 14, fill: SHADE[f]}, gI); });
    tx(gI, X(hi) + 22, yIdx + 6, Rr.index_vw.map(function (v) { return v.toFixed(2); }).join(', '), {'font-size': 16, class: 't-mono t-quiet'});
  });

  // ------------------------------------------------------------------ 5.5 best performers: the validation pick of each pool
  C.best = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s, B = D().best, xs = [262, 636, 1010], CW = 360, unit = {M: ' test months', '2W': ' test fortnights', W: ' test weeks'};
    var per = D().pipeline.periods, top = 64, RH = 100;
    B.columns.forEach(function (c, ci) {
      tx(s, xs[ci], 24, FN[c.freq], {'font-size': 23, 'font-weight': 600});
      tx(s, xs[ci] + CW, 24, per[c.freq][2] + unit[c.freq], {'text-anchor': 'end', 'font-size': 16, class: 't-mono t-quiet'});
    });
    B.pools.forEach(function (p, r) {
      var y = top + r * RH;
      tx(s, 0, y + 30, p.label, {'font-size': 21, 'font-weight': 600});
      p.how.split('|').forEach(function (l, j) { tx(s, 0, y + 56 + j * 22, l, {'font-size': 16, class: 't-mono t-quiet'}); });
    });
    var gG = reveal(box, el('g', {}, s), k, 1, animate, 700), gH = reveal(box, el('g', {}, s), k, 2, animate, 700);
    B.columns.forEach(function (c, ci) {
      c.cells.forEach(function (x, r) {
        var y = top + r * RH, x0 = xs[ci], best = r === c.best;
        el('rect', {x: x0, y: y, width: CW, height: RH - 10, fill: COL.wash}, gG);
        if (!x) { tx(gG, x0 + 16, y + 52, 'no tuned window: stays at 5 days', {'font-size': 17, class: 't-mono t-quiet'}); return; }
        var dag = x.pool === 'shock' && c.freq === 'M' ? ' †' : '';
        tx(gG, x0 + 14, y + 27, x.set + dag, {'font-size': 18, 'font-weight': 500, fill: best && k >= 2 ? COL.cobalt : COL.ink});
        tx(gG, x0 + 14, y + 51, x.model_name + ' · test IC ' + ic3(x.ic), {'font-size': 16, class: 't-mono t-quiet'});
        tx(gG, x0 + 14, y + 77, 'net Sharpe ' + signed(x.net) + ' · buffered ' + signed(x.buf), {'font-size': 16, class: 't-mono'});
        if (best) {
          el('rect', {x: x0 + 1, y: y + 1, width: CW - 2, height: RH - 12, fill: 'none', stroke: COL.cobalt, 'stroke-width': 2.5}, gH);
        }
      });
    });
    var yI = top + 3 * RH;
    tx(s, 0, yI + 24, 'Index, value-weighted', {'font-size': 20});
    B.index_vw.forEach(function (ix, ci) { tx(s, xs[ci] + 14, yI + 24, 'net Sharpe ' + signed(ix.sharpe), {'font-size': 16, class: 't-mono t-quiet'}); });
    // cumulative net excess return of the best cell (buffered) and of the index, test months only
    var y0 = yI + 50, H = 112;
    tx(gH, 0, y0 + 30, 'Framed: best of the column.', {'font-size': 16, class: 't-mono t-cobalt'});
    tx(gH, 0, y0 + 52, 'Its buffered return (blue)', {'font-size': 16, class: 't-mono t-quiet'});
    tx(gH, 0, y0 + 74, 'and the index (grey), net', {'font-size': 16, class: 't-mono t-quiet'});
    tx(gH, 0, y0 + 96, 'of the risk-free rate', {'font-size': 16, class: 't-mono t-quiet'});
    B.curves.forEach(function (cv, ci) {
      var x0 = xs[ci], n = cv.dates.length, lo = Math.min(0, Math.min.apply(null, cv.pick), Math.min.apply(null, cv.index));
      var hi = Math.max(Math.max.apply(null, cv.pick), Math.max.apply(null, cv.index));
      var X = lin(0, n - 1, x0, x0 + CW - 64), Y = lin(lo, hi, y0 + H, y0 + 14);
      el('line', {x1: x0, x2: x0 + CW - 64, y1: Y(0), y2: Y(0), class: 'ln-axis'}, gH);
      var segs = [[0]];
      for (var i = 1; i < n; i++) { if ((Date.parse(cv.dates[i]) - Date.parse(cv.dates[i - 1])) / 864e5 > 45) segs.push([i]); else segs[segs.length - 1].push(i); }
      segs.forEach(function (sg, si) {
        if (si) el('line', {x1: X(sg[0] - 0.5), x2: X(sg[0] - 0.5), y1: y0 + 8, y2: y0 + H, stroke: COL.rule, 'stroke-dasharray': '3 4'}, gH);
        [['index', '#A8A8A6'], ['pick', COL.cobalt]].forEach(function (q) {
          var d = '';
          sg.forEach(function (i, j) { d += (j ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(cv[q[0]][i]).toFixed(1); });
          el('path', {d: d, fill: 'none', stroke: q[1], 'stroke-width': q[0] === 'pick' ? 2.6 : 2}, gH);
        });
      });
      var ep = cv.pick[n - 1], ei = cv.index[n - 1], yp = Y(ep), yi = Y(ei);
      if (Math.abs(yp - yi) < 18) { if (yp < yi) yp = yi - 18; else yi = yp - 18; }
      tx(gH, x0 + CW - 58, yp + 5, signed(ep * 100, 0) + '%', {'font-size': 16, class: 't-mono t-cobalt'});
      tx(gH, x0 + CW - 58, yi + 5, signed(ei * 100, 0) + '%', {'font-size': 16, class: 't-mono t-quiet'});
    });
  });
})();
