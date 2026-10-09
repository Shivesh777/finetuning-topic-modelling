/* The title and closing panel: the news as an abacus, after the bead grid of the D. E. Shaw site.
   One wire per NoLBERT topic (the 20-topic model); one row per month, Sep 2020 to Dec 2024, the
   newest at the bottom. A bead: the topic took a larger share of that month's news than its median
   share of the previous 12 months (DATA.map.beads, computed by tools/build_data.py), so each row
   uses only its own month and earlier ones. The grid climbs one row a month, holds on Dec 2024,
   then starts again. Sep 2020 is the first month after every fitted part was frozen. */
(function () {
  'use strict';
  var MS_PER_MONTH = 1000, MOVE = 0.55, HOLD_END = 3000;
  var BEAD = '44,75,239', WIRE = 'rgba(64,92,240,0.5)';
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var CALM = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var UNUSED = (window.DATA.facts && window.DATA.facts.splits && window.DATA.facts.splits.unused_from) || 2017;

  function Grid(box) {
    this.box = box; this.B = window.DATA.map.beads;
    this.canvas = box.querySelector('canvas'); this.ctx = this.canvas.getContext('2d');
    this.when = box.querySelector('.when'); this.raf = null; this.t0 = 0;
  }
  Grid.prototype.size = function () {
    var w = this.box.offsetWidth || 960, h = this.box.offsetHeight || 1080, r = 2, n = this.B.rows[0].length;
    if (this.canvas.width !== w * r) { this.canvas.width = w * r; this.canvas.height = h * r; }
    this.W = w; this.H = h; this.ctx.setTransform(r, 0, 0, r, 0, 0);
    this.sx = Math.min(46, (w - 120) / (n - 1));                // wire spacing; rows as far apart
    this.x0 = (w - this.sx * (n - 1)) / 2;
    this.top = 96; this.bot = h - 172;                         // room for the month and the caption
    this.rad = this.sx * 0.44;
    this.first = Math.min(this.B.rows.length - 1, Math.floor((this.bot - this.top) / this.sx));   // open on a full grid
  };
  // p: the newest month shown (an index into B.rows); its fraction is the climb into place
  Grid.prototype.draw = function (p, fade) {
    var c = this.ctx, B = this.B, k = Math.floor(p), u = Math.min(1, (p - k) / MOVE), e = u * u * (3 - 2 * u);
    var sx = this.sx, top = this.top, bot = this.bot, n = B.rows[0].length, i, j;
    c.clearRect(0, 0, this.W, this.H);
    c.globalAlpha = fade == null ? 1 : fade;
    c.strokeStyle = WIRE; c.lineWidth = 1.5;
    for (i = 0; i < n; i++) {
      var x = this.x0 + i * sx;
      c.beginPath(); c.moveTo(x, top - sx * 0.8); c.lineTo(x, bot + sx * 0.6); c.stroke();
    }
    for (j = k; j >= 0; j--) {
      var y = bot - (k - j - 1 + e) * sx;                       // every row climbs one step as the new month enters
      if (y < top - sx) break;
      var a = Math.max(0, Math.min(1, (y - (top - sx)) / (2.5 * sx)));   // the oldest rows fade out at the top
      if (j === k) a *= e;                                      // the new month fades in at the bottom
      if (a <= 0) continue;
      var mo = +B.months[j].slice(5, 7), dim = +B.months[j].slice(0, 4) >= UNUSED && mo >= 2 && mo <= 6;   // Feb–Jun: no complete return file
      c.fillStyle = 'rgba(' + BEAD + ',' + (a * (dim ? 0.28 : 1)).toFixed(3) + ')';
      var row = B.rows[j];
      for (i = 0; i < n; i++) {
        if (!row[i]) continue;
        c.beginPath(); c.arc(this.x0 + i * sx, y, this.rad, 0, Math.PI * 2); c.fill();
      }
    }
    if (this.when) { var m = B.months[k]; this.when.textContent = MON[+m.slice(5, 7) - 1] + ' ' + m.slice(0, 4); }
    c.globalAlpha = fade == null ? 1 : fade;
    c.fillStyle = '#9C9C9A'; c.font = '400 15px "IBM Plex Mono", Menlo, monospace'; c.textBaseline = 'alphabetic';
    c.fillText('dimmed rows: Feb to Jun, months without a complete return file', this.x0, 40);
  };
  Grid.prototype.tick = function (now) {
    var me = this, last = this.B.rows.length - 1 + MOVE, run = (last - this.first) * MS_PER_MONTH;
    if (!this.t0) this.t0 = now;
    var el = now - this.t0, p = Math.min(last, this.first + el / MS_PER_MONTH);
    var fade = Math.min(1, el / 500, (run + HOLD_END - el) / 500);   // fade in at the start, out before the loop
    if (el > run + HOLD_END) { this.t0 = now; p = this.first; fade = 0; }   // hold on Dec 2024, then start again
    this.draw(p, Math.max(0, fade));
    this.raf = requestAnimationFrame(function (t) { me.tick(t); });
  };
  Grid.prototype.start = function () {
    this.stop(); this.size();
    if (CALM) { this.still(); return; }
    var me = this; this.t0 = 0; this.raf = requestAnimationFrame(function (t) { me.tick(t); });
  };
  Grid.prototype.stop = function () { if (this.raf) cancelAnimationFrame(this.raf); this.raf = null; };
  Grid.prototype.still = function () { this.size(); this.draw(this.B.rows.length - 1 + MOVE); };   // Dec 2024, settled

  var grids = new WeakMap();
  function get(box) { if (!grids.has(box)) grids.set(box, new Grid(box)); return grids.get(box); }
  window.BEADS = {
    enter: function (box) { get(box).start(); },
    leave: function (box) { get(box).stop(); },
    still: function (box) { get(box).still(); }
  };
})();
