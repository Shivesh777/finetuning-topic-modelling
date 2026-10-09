/* Deck engine: a fixed 1920 x 1080 stage, slides with steps, keyboard and clicker control.
   One click per slide: going forward to a slide plays all of its steps by themselves, in order.
   A slide's data-auto lists when each step starts, in ms after the slide appears.

   Keys   next slide: Right, Down, PageDown, Space, Enter (or a click)
          back: Left, Up, PageUp, Backspace (the previous slide, shown complete)
          R: replay this slide's animations    G then a number then Enter: jump to that slide
          I: index of slides (click one)    Home / End    F: full screen    T: timer    B or . : black screen
          N: this slide's speaker notes (an <aside class="notes"> in the slide); ?still=1&notes=1 prints them under each slide
   A click anywhere goes forward, except on anything marked data-noadvance.
   #5 in the address opens slide 5 and plays it; #5.2 opens it at step 2 and holds there.
   ?still=1 shows every slide at its last step, one under another: the pre-read, printable to PDF.
   ?still=1&only=12 shows slide 12 alone; ?still=1&qa=1 also runs js/qa.js, a layout check
   (overlapping text, text past the margins); ?selftest=1 clicks through every step (js/selftest.js). */
(function () {
  'use strict';
  var W = 1920, H = 1080;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  var STILL = /[?&]still=1/.test(location.search);
  if (STILL) document.documentElement.classList.add('still');
  var stage = $('#stage'), slides = $$('.slide');
  slides.forEach(function (sl, i) { sl.id = 's' + (i + 1); });   // ?still=1#s12 opens the pre-read at slide 12
  var SECTIONS = {1: 'Recap', 2: 'NoLBERT', 3: 'Fine-tuning', 4: 'How it works', 5: 'Results', B: 'Backup'};
  var cur = -1, step = 0, started = null, typed = null;
  var CHARTS = window.CHARTS || {};

  // ------------------------------------------------------------------ progress marks: the five sections in the header row, the current one in cobalt
  slides.forEach(function (sl) {
    var sec = sl.dataset.sec, tags = sl.querySelector('.tags');
    if (!sec || sec === '0' || !tags) return;
    var marks = document.createElement('div'); marks.className = 'marks';
    marks.innerHTML = Object.keys(SECTIONS).filter(function (k) { return k !== 'B' || sec === 'B'; }).map(function (k) {
      return '<span class="' + (k === sec ? 'cur' : '') + '">' + (k === 'B' ? '' : k + ' ') + SECTIONS[k] + '</span>';
    }).join('');
    var who = tags.querySelector('.who');
    if (who) tags.insertBefore(marks, who); else tags.appendChild(marks);
  });

  // ------------------------------------------------------------------ chrome
  var chrome = document.createElement('div'); chrome.id = 'chrome';
  chrome.innerHTML = '<div id="count"></div><div id="timer"></div><div id="goto"></div><div id="notes"></div>';
  stage.appendChild(chrome);
  var black = document.createElement('div'); black.id = 'black'; stage.appendChild(black);
  var index = document.createElement('div'); index.id = 'index'; stage.appendChild(index);
  index.innerHTML = slides.map(function (sl, i) {
    return '<div data-noadvance data-go="' + i + '"><b>' + (i + 1) + '</b>' + (sl.dataset.num ? sl.dataset.num + ' ' : '') + (sl.dataset.name || 'Title') + '</div>';
  }).join('');
  index.addEventListener('click', function (e) {
    var d = e.target.closest('[data-go]'); if (!d) return;
    index.classList.remove('on'); enter(+d.dataset.go); e.stopPropagation();
  });

  function fit() { if (STILL) return; var s = Math.min(window.innerWidth / W, window.innerHeight / H); stage.style.transform = 'translate(-50%,-50%) scale(' + s + ')'; }
  window.addEventListener('resize', fit); fit();

  function maxStep(sl) {
    if (sl.dataset.steps) return +sl.dataset.steps;
    var m = 0; $$('[data-step]', sl).forEach(function (e) { m = Math.max(m, +e.dataset.step); });
    return m;
  }
  function charts(sl, fn, arg) {
    $$('[data-chart]', sl).forEach(function (c) {
      var ch = CHARTS[c.dataset.chart];
      if (ch && ch[fn]) { try { ch[fn](c, arg, sl); } catch (e) { console.error(c.dataset.chart, e); } }
    });
  }
  function applyStep(sl) {
    sl.dataset.at = step;
    $$('[data-step]', sl).forEach(function (e) { e.classList.toggle('shown', step >= +e.dataset.step); });
    charts(sl, 'step', step);
  }
  function paint() {
    $('#count').textContent = (cur + 1) + ' / ' + slides.length;
    try { history.replaceState(null, '', location.search + '#' + (cur + 1) + (step ? '.' + step : '')); } catch (e) {}
  }
  function go(i, st, keep) {
    if (!keep) stopAuto();
    i = Math.max(0, Math.min(slides.length - 1, i));
    var sl = slides[i];
    if (i !== cur) {
      if (cur >= 0) { slides[cur].classList.remove('on'); charts(slides[cur], 'leave'); }
      if (cur === 0 && started == null) started = Date.now();
      cur = i; step = Math.max(0, Math.min(maxStep(sl), st || 0));
      sl.classList.add('on');
      var nt = sl.querySelector('.notes'); $('#notes').textContent = nt ? nt.textContent : '';
      charts(sl, 'enter', step);
    } else step = Math.max(0, Math.min(maxStep(sl), st || 0));
    applyStep(sl); paint();
  }
  // autoplay: the steps of the slide just entered, each at its time from data-auto
  // (default: the first after 0.6 s, then one every 1.6 s, at the deck's pace: 2x, 4x in section 4).
  // Leaving the slide cancels the rest.
  var timers = [];
  function stopAuto() { timers.forEach(clearTimeout); timers = []; }
  function autoplay(i) {
    var sl = slides[i], at = (sl.dataset.auto || '').split(/\s+/).filter(Boolean).map(Number);
    for (var k = 1; k <= maxStep(sl); k++) (function (k) {
      timers.push(setTimeout(function () { if (cur === i && step === k - 1) go(i, k, true); },
        at[k - 1] != null ? at[k - 1] : (600 + (k - 1) * 1600) * window.DECKUTIL.pace(sl)));
    })(k);
  }
  function enter(i) { go(i, 0); autoplay(cur); }
  function next() { if (cur < slides.length - 1) enter(cur + 1); }
  function prev() { if (cur > 0) go(cur - 1, maxStep(slides[cur - 1])); }

  // ------------------------------------------------------------------ keys
  function key(k) {
    if (black.classList.contains('on') && k !== 'f' && k !== 'F') { black.classList.remove('on'); return true; }
    if (typed !== null) {                                       // G was pressed: collect digits
      if (/^[0-9]$/.test(k)) { typed += k; $('#goto').textContent = 'go to ' + typed; return true; }
      if (k === 'Enter' && typed) { var n = +typed; typed = null; $('#goto').textContent = ''; enter(n - 1); return true; }
      if (k === 'Escape' || k === 'Backspace') { typed = null; $('#goto').textContent = ''; return true; }
    }
    if (k === 'g' || k === 'G') { typed = ''; $('#goto').textContent = 'go to …'; return true; }
    if (k === 'i' || k === 'I') { index.classList.toggle('on'); return true; }
    if (k === 'Escape') { index.classList.remove('on'); return true; }
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || k === ' ' || k === 'Spacebar' || k === 'Enter') next();
    else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp' || k === 'Backspace') prev();
    else if (k === 'Home') enter(0);
    else if (k === 'End') enter(slides.length - 1);
    else if (k === 'r' || k === 'R') enter(cur);
    else if (k === 'f' || k === 'F') { if (document.fullscreenElement) document.exitFullscreen(); else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen(); }
    else if (k === 't' || k === 'T') $('#timer').classList.toggle('on');
    else if (k === 'n' || k === 'N') $('#notes').classList.toggle('on');
    else if (k === 'b' || k === 'B' || k === '.') black.classList.add('on');
    else return false;
    return true;
  }

  // ------------------------------------------------------------------ start
  if (STILL) {
    // every slide at its last step; charts draw their final frame. ?still=1&only=12 shows slide 12 alone
    var only = /[?&]only=(\d+)/.exec(location.search), notes = /[?&]notes=1/.test(location.search);
    slides.forEach(function (sl, i) {
      if (only && i + 1 !== +only[1]) { sl.style.display = 'none'; return; }
      var nt = sl.querySelector('.notes');
      if (notes && nt) { nt.classList.add('notes-out'); stage.insertBefore(nt, sl.nextSibling); }   // speaker notes under the slide
      sl.classList.add('on');
      $$('[data-step]', sl).forEach(function (e) { e.classList.add('shown'); });
      charts(sl, 'still', maxStep(sl));
    });
    window.DECK = {still: true};
    if (/[?&]qa=1/.test(location.search)) { var q = document.createElement('script'); q.src = 'js/qa.js'; document.body.appendChild(q); }
    return;
  }
  window.addEventListener('keydown', function (e) { if (e.metaKey || e.ctrlKey || e.altKey) return; if (key(e.key)) e.preventDefault(); });
  window.addEventListener('click', function (e) {
    if (e.target.closest && e.target.closest('[data-noadvance]')) return;
    if (black.classList.contains('on')) black.classList.remove('on'); else next();
  });
  setInterval(function () {
    var s = started == null ? 0 : Math.floor((Date.now() - started) / 1000);
    $('#timer').textContent = Math.floor(s / 60) + ':' + ('0' + s % 60).slice(-2);
  }, 500);
  function fromHash() {
    var h = (location.hash || '').replace('#', '').split('.'), i = Math.max(0, (parseInt(h[0], 10) || 1) - 1);
    if (h.length > 1) go(i, parseInt(h[1], 10) || 0); else enter(i);
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();
  window.DECK = {go: go, next: next, prev: prev, replay: function () { enter(cur); },
    steps: function (i) { return maxStep(slides[i]); }, state: function () { return {slide: cur + 1, step: step}; }};
  if (/[?&]selftest=1/.test(location.search)) { var t = document.createElement('script'); t.src = 'js/selftest.js'; document.body.appendChild(t); }
})();
