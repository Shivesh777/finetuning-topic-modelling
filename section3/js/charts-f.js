/* Intro exhibit for section 3: one article, two chunks, the space reshaped.
   Beat 1: the two chunks travel through the encoder and land far apart.
   Beat 2: the fine-tune pulls the pair together and pushes the rest away.
   Beat 3: the settled pair and the held-out result. */
(function () {
  'use strict';
  var U = window.DECKUTIL, C = window.CHARTS, el = U.el, tx = U.tx, surface = U.surface,
      ease = U.ease, tween = U.tween, halt = U.halt, reveal = U.reveal, COL = U.COL;

  function stepped(build) {
    return {
      enter: function (box, k) { halt(box); build(box, k, false); box._k = k; },
      step: function (box, k) { if (box._k === k) return; box._k = k; build(box, k, true); },
      leave: function (box) { halt(box); },
      still: function (box, k) { halt(box); build(box, k == null ? 3 : k, false); }
    };
  }

  C.pairintro = stepped(function (box, k, animate) {
    var g = surface(box), s = g.s;
    var seed = 23;
    function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647; }

    // the article, synthetic text
    tx(s, 0, 34, 'one article · text synthetic', {'font-size': 18, class: 't-mono t-quiet'});
    el('rect', {x: 0, y: 48, width: 340, height: 252, rx: 8, fill: '#fff', stroke: COL.rule, 'stroke-width': 1.5}, s);
    tx(s, 18, 80, 'Chipmaker lifts forecast', {'font-size': 17, 'font-weight': 600});
    var lines = ['The company raised its outlook',
                 'after revenue climbed 38%.',
                 'New accelerator chips are sold',
                 'out through next spring.',
                 'Analysts flagged tight supply.',
                 'Shares rose 6% after hours.'];
    lines.forEach(function (t, i) { tx(s, 18, 112 + i * 26, t, {'font-size': 15}); });
    [[92, 'Chunk A'], [196, 'Chunk B']].forEach(function (r) {
      el('rect', {x: 10, y: r[0], width: 320, height: 54, rx: 6, fill: 'rgba(44,75,239,.10)', stroke: COL.cobalt, 'stroke-width': 2.5}, s);
      tx(s, 322, r[0] + 19, r[1], {'text-anchor': 'end', 'font-size': 14, class: 't-mono t-cobalt'});
    });

    // the encoder
    var gE = reveal(box, el('g', {}, s), k, 1, animate, 500);
    el('line', {x1: 344, y1: 185, x2: 424, y2: 185, stroke: COL.mist, 'stroke-width': 2}, gE);
    el('rect', {x: 430, y: 140, width: 130, height: 90, rx: 10, fill: 'rgba(44,75,239,.06)', stroke: COL.ink, 'stroke-width': 2}, gE);
    tx(gE, 495, 192, 'NoLBERT', {'text-anchor': 'middle', 'font-size': 20, 'font-weight': 600});

    // the embedding space
    var F = {x0: 640, x1: 1360, y0: 44, y1: 600};
    var greys = [];
    for (var i = 0; i < 34; i++) {
      greys.push({b: [F.x0 + rnd() * (F.x1 - F.x0), F.y0 + rnd() * (F.y1 - F.y0)]});
    }
    var Pc = [985, 310];
    var A = {b: [760, 110], a: [Pc[0] - 13, Pc[1] - 6]};
    var B = {b: [1250, 500], a: [Pc[0] + 13, Pc[1] + 7]};
    greys.forEach(function (d) {
      var vx = d.b[0] - Pc[0], vy = d.b[1] - Pc[1], n = Math.sqrt(vx * vx + vy * vy) || 1;
      var push = 95 * Math.exp(-n / 240);
      d.a = [d.b[0] + vx / n * push, d.b[1] + vy / n * push];
    });

    var gF = reveal(box, el('g', {}, s), k, 1, animate, 700);
    var gdots = greys.map(function (d) { return el('circle', {r: 5.5, fill: '#C4C4C1', cx: d.b[0], cy: d.b[1]}, gF); });
    var link = el('line', {stroke: COL.cobalt, 'stroke-width': 1.6, opacity: 0}, gF);
    var da = el('circle', {r: 8, fill: COL.cobalt, opacity: 0}, gF);
    var db = el('circle', {r: 8, fill: COL.cobalt, opacity: 0}, gF);

    function setPair(pa, pb, showLink) {
      da.setAttribute('cx', pa[0].toFixed(1)); da.setAttribute('cy', pa[1].toFixed(1));
      db.setAttribute('cx', pb[0].toFixed(1)); db.setAttribute('cy', pb[1].toFixed(1));
      link.setAttribute('x1', pa[0].toFixed(1)); link.setAttribute('y1', pa[1].toFixed(1));
      link.setAttribute('x2', pb[0].toFixed(1)); link.setAttribute('y2', pb[1].toFixed(1));
      link.setAttribute('opacity', showLink ? 0.5 : 0);
    }
    function flight(u) {
      function path(from, to, u2) {
        var mid = [495, 185];
        if (u2 < 0.45) { var v = u2 / 0.45; return [from[0] + (mid[0] - from[0]) * v, from[1] + (mid[1] - from[1]) * v]; }
        var w = (u2 - 0.45) / 0.55; return [mid[0] + (to[0] - mid[0]) * w, mid[1] + (to[1] - mid[1]) * w];
      }
      setPair(path([180, 122], A.b, u), path([180, 234], B.b, u), u >= 1);
    }
    function reshape(u) {
      setPair([A.b[0] + (A.a[0] - A.b[0]) * u, A.b[1] + (A.a[1] - A.b[1]) * u],
              [B.b[0] + (B.a[0] - B.b[0]) * u, B.b[1] + (B.a[1] - B.b[1]) * u], true);
      greys.forEach(function (d, j) {
        gdots[j].setAttribute('cx', (d.b[0] + (d.a[0] - d.b[0]) * u).toFixed(1));
        gdots[j].setAttribute('cy', (d.b[1] + (d.a[1] - d.b[1]) * u).toFixed(1));
      });
    }

    var lab1 = reveal(box, el('g', {}, s), k, 1, animate, 900);
    tx(lab1, 640, 646, 'Before', {'font-size': 18, class: 't-mono t-quiet'});
    tx(lab1, 726, 646, 'far apart', {'font-size': 18, class: 't-quiet'});

    var lab2 = reveal(box, el('g', {}, s), k, 2, animate, 700);
    tx(lab2, 430, 300, 'Pull together', {'font-size': 22, 'font-weight': 600, class: 't-cobalt'});
    tx(lab2, 430, 330, 'Push apart', {'font-size': 22, class: 't-quiet'});

    var lab3 = reveal(box, el('g', {}, s), k, 3, animate, 700);
    el('circle', {cx: Pc[0], cy: Pc[1], r: 46, fill: 'none', stroke: COL.mist, 'stroke-width': 2, 'stroke-dasharray': '8 7'}, lab3);
    tx(lab3, 1360, 646, 'After   8% → 89%', {'text-anchor': 'end', 'font-size': 24, 'font-weight': 600, class: 't-cobalt'});

    if (k >= 1) { da.setAttribute('opacity', 1); db.setAttribute('opacity', 1); }
    if (k === 1) {
      if (animate) { flight(0); tween(box, 1600, function (u) { flight(ease(u)); }); }
      else flight(1);
    } else if (k >= 2) {
      if (animate && k === 2) { reshape(0); tween(box, 2200, function (u) { reshape(ease(u)); }); }
      else reshape(1);
    }
  });
})();
