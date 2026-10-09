/* The title and closing panel: what the news was about, month by month.
   Columns: the 20 topics of the 20-topic NoLBERT model, named after their top words (DATA.map.terms).
   Rows: months, Sep 2020 at the top to Dec 2024 at the bottom. A dot: the topic took a larger share of
   that month's news than its median share of the previous 12 months (DATA.map.beads, computed by
   tools/build_data.py), so each row uses only its own month and earlier ones.
   Four callouts mark months in which one topic stood out and the headlines say why. Each was checked
   on the article file: the headline keyword in the callout peaks in that month (vaccine: Nov 2020;
   Russia/Ukraine and oil: Mar 2022; iPhone: Sep 2023; election: Nov-Dec 2024). The ratio in a callout is
   the topic's share that month over its median share of the previous 12 months (DATA.map.shares).
   The rows fill in once, top to bottom, and the grid then holds. */
(function () {
  'use strict';
  var MS_PER_MONTH = 90, FADE = 3;                              // a new row fades in over FADE rows
  var BEAD = '44,75,239', HOT = '#93A5FF', INK = '#EDEDEB', QUIET = '#A3A3A0', DIM = '#7A7A77';
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var CALM = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // our names for the 20 topics, from their top words (the first three are in DATA.map.terms)
  var NAMES = ['Biotech & drugs', 'Dividends', 'Deals', 'Trading tape', 'Earnings vs est.', 'Cannabis', 'Investing',
               'Markets', 'Dividend dates', 'Big Tech', 'Energy & oil', 'Charts', 'Sector updates', 'Stock screens',
               'Funds & ETFs', 'Earnings calls', 'Trading tips', 'Earnings beats', 'ETF flows', 'Finance'];
  var EVENTS = [{month: '2020-11', topic: 0, what: 'Vaccine results'},
                {month: '2022-03', topic: 10, what: 'War in Ukraine'},
                {month: '2023-09', topic: 9, what: 'iPhone launch'},
                {month: '2024-11', topic: 7, what: 'US election'}];
  var FONT_SANS = '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif', FONT_MONO = '"IBM Plex Mono", Menlo, monospace',
      FONT_SERIF = '"Source Serif 4", Georgia, serif';

  // the topic's share in a month over its median share of the previous 12 months
  function ratio(M, month, k) {
    var i = M.months.indexOf(month), s = M.shares[k];
    if (i < 12) return null;
    var prev = s.slice(i - 12, i).slice().sort(function (a, b) { return a - b; });
    return s[i] / ((prev[5] + prev[6]) / 2);
  }

  function Grid(box) {
    var M = window.DATA.map;
    this.box = box; this.M = M; this.B = M.beads;
    this.canvas = box.querySelector('canvas'); this.ctx = this.canvas.getContext('2d');
    this.when = box.querySelector('.when'); this.raf = null; this.t0 = 0;
    this.events = EVENTS.map(function (e) {
      var row = M.beads.months.indexOf(e.month), r = ratio(M, e.month, e.topic);
      return {row: row, topic: e.topic, what: e.what, ratio: r,
              label: MON[+e.month.slice(5, 7) - 1] + ' ' + e.month.slice(0, 4)};
    }).filter(function (e) { return e.row >= 0 && M.beads.rows[e.row][e.topic]; });   // only where the topic has its dot
  }
  Grid.prototype.size = function () {
    var w = this.box.offsetWidth || 960, h = this.box.offsetHeight || 1080, r = 2, n = this.B.rows[0].length, m = this.B.rows.length;
    if (this.canvas.width !== w * r) { this.canvas.width = w * r; this.canvas.height = h * r; }
    this.W = w; this.H = h; this.ctx.setTransform(r, 0, 0, r, 0, 0);
    this.x0 = 118; this.sx = Math.min(27, (w * 0.62 - this.x0) / (n - 1));    // the right third holds the callouts
    this.right = this.x0 + this.sx * (n - 1);
    this.top = 318; this.bot = h - 196;
    this.sy = (this.bot - this.top) / (m - 1);
    this.rad = Math.min(this.sx * 0.3, this.sy * 0.43);
  };
  Grid.prototype.y = function (j) { return this.top + j * this.sy; };
  Grid.prototype.x = function (i) { return this.x0 + i * this.sx; };
  // p: rows filled so far (0 to the number of months)
  Grid.prototype.draw = function (p) {
    var c = this.ctx, B = this.B, M = this.M, n = B.rows[0].length, m = B.rows.length, me = this, i, j;
    var hot = {}; this.events.forEach(function (e) { hot[e.topic] = 1; });
    c.clearRect(0, 0, this.W, this.H);
    // heading
    c.textBaseline = 'alphabetic'; c.textAlign = 'left';
    c.fillStyle = INK; c.font = '600 31px ' + FONT_SERIF;
    c.fillText('What the news was about,', 44, 74);
    c.fillText('month by month', 44, 112);
    // topic names above the wires, slanted
    for (i = 0; i < n; i++) {
      c.save(); c.translate(this.x(i) + 4, this.top - 18); c.rotate(-Math.PI / 3.2);
      c.fillStyle = hot[i] ? HOT : QUIET; c.font = (hot[i] ? '600 ' : '400 ') + '15px ' + FONT_SANS;
      c.fillText(NAMES[i] || M.terms[i].split(', ')[0], 0, 0); c.restore();
    }
    // wires, year rules and year labels
    c.strokeStyle = 'rgba(64,92,240,0.32)'; c.lineWidth = 1.2;
    for (i = 0; i < n; i++) { c.beginPath(); c.moveTo(this.x(i), this.top - 8); c.lineTo(this.x(i), this.bot + 8); c.stroke(); }
    c.font = '400 15px ' + FONT_MONO; c.textAlign = 'right';
    B.months.forEach(function (mo, j2) {
      var first = j2 === 0, jan = mo.slice(5, 7) === '01';
      if (!first && !jan) return;
      var y = me.y(j2);
      if (jan) { c.strokeStyle = 'rgba(255,255,255,0.10)'; c.beginPath(); c.moveTo(me.x0 - 12, y - me.sy / 2); c.lineTo(me.right + 12, y - me.sy / 2); c.stroke(); }
      c.fillStyle = j2 < p ? QUIET : DIM;
      c.fillText(first ? MON[+mo.slice(5, 7) - 1] + ' ' + mo.slice(0, 4) : mo.slice(0, 4), me.x0 - 20, y + 5);
    });
    c.textAlign = 'left';
    // callout leaders behind the dots
    var shown = function (row) { return Math.max(0, Math.min(1, (p - row) / FADE)); };
    this.events.forEach(function (e) {
      var a = shown(e.row); if (!a) return;
      var y = me.y(e.row);
      c.globalAlpha = a * 0.75; c.strokeStyle = HOT; c.lineWidth = 1.4;
      c.beginPath(); c.moveTo(me.x(e.topic) + me.rad + 3, y); c.lineTo(me.right + 34, y); c.stroke();
      c.globalAlpha = 1;
    });
    // the dots
    for (j = 0; j < m && j < p; j++) {
      var a = shown(j), y = this.y(j), row = B.rows[j];
      c.fillStyle = 'rgba(' + BEAD + ',' + a.toFixed(3) + ')';
      for (i = 0; i < n; i++) {
        if (!row[i]) continue;
        c.beginPath(); c.arc(this.x(i), y, this.rad, 0, Math.PI * 2); c.fill();
      }
    }
    // the callouts
    this.events.forEach(function (e) {
      var a = shown(e.row); if (!a) return;
      var x = me.x(e.topic), y = me.y(e.row), tx = me.right + 44;
      c.globalAlpha = a;
      c.strokeStyle = '#FFFFFF'; c.lineWidth = 2; c.beginPath(); c.arc(x, y, me.rad + 3.5, 0, Math.PI * 2); c.stroke();
      c.fillStyle = HOT; c.beginPath(); c.arc(me.right + 34, y, 2.5, 0, Math.PI * 2); c.fill();
      c.fillStyle = INK; c.font = '600 18px ' + FONT_SANS; c.fillText(e.label + ' · ' + e.what, tx, y + 6);
      c.fillStyle = QUIET; c.font = '400 15px ' + FONT_MONO;
      c.fillText((NAMES[e.topic] || '') + ' news: ' + (e.ratio ? e.ratio.toFixed(1) + '× usual' : 'above usual'), tx, y + 30);
      c.globalAlpha = 1;
    });
    if (this.when) { var k = Math.max(0, Math.min(m - 1, Math.ceil(p) - 1)), mo = B.months[k]; this.when.textContent = MON[+mo.slice(5, 7) - 1] + ' ' + mo.slice(0, 4); }
  };
  Grid.prototype.tick = function (now) {
    var me = this, m = this.B.rows.length, end = m + FADE;
    if (!this.t0) this.t0 = now;
    var p = Math.min(end, (now - this.t0) / MS_PER_MONTH);
    this.draw(p);
    if (p < end) this.raf = requestAnimationFrame(function (t) { me.tick(t); });   // fill once, then hold
    else this.raf = null;
  };
  Grid.prototype.start = function () {
    this.stop(); this.size();
    if (CALM) { this.still(); return; }
    var me = this; this.t0 = 0; this.raf = requestAnimationFrame(function (t) { me.tick(t); });
  };
  Grid.prototype.stop = function () { if (this.raf) cancelAnimationFrame(this.raf); this.raf = null; };
  Grid.prototype.still = function () { this.size(); this.draw(this.B.rows.length + FADE); };   // every month, settled

  var grids = new WeakMap();
  function get(box) { if (!grids.has(box)) grids.set(box, new Grid(box)); return grids.get(box); }
  window.BEADS = {
    enter: function (box) { get(box).start(); },
    leave: function (box) { get(box).stop(); },
    still: function (box) { get(box).still(); }
  };
  // canvas text needs the web fonts: draw again once they are in
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () {
    document.querySelectorAll('.panel-dark[data-chart="beads"]').forEach(function (box) {
      if (grids.has(box) && !get(box).raf) get(box).draw(get(box).B.rows.length + FADE);
    });
  });
})();
