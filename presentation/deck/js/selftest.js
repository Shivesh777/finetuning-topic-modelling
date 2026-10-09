/* Click-through test of the live deck: open /?selftest=1.
   1. Applies every step of every slide the way autoplay does (each step's animation code runs).
   2. Goes back through the deck (every slide shown complete).
   3. Checks that one forward click plays every step of slide 3 by itself, and that a click
      during a slide's animations moves straight on to the next slide.
   The result is printed into <pre id="selftest-out">. */
(function () {
  'use strict';
  var D = window.DECK, errors = [], applied = 0;
  var log = console.error;
  console.error = function () { errors.push(Array.prototype.map.call(arguments, String).join(' ')); log.apply(console, arguments); };
  window.addEventListener('error', function (e) { errors.push('uncaught: ' + e.message); });
  function run(label, f) { var b = errors.length; try { f(); } catch (e) { errors.push('thrown: ' + e.message); } if (errors.length > b) errors[errors.length - 1] = label + ': ' + errors[errors.length - 1]; }

  var n = document.querySelectorAll('.slide').length;
  for (var i = 0; i < n; i++) {
    run('slide ' + (i + 1) + ' enter', function () { D.go(i, 0); });
    for (var k = 1; k <= D.steps(i); k++) { run('slide ' + (i + 1) + ' step ' + k, function () { D.go(i, k, true); }); applied++; }
  }
  for (i = n - 1; i > 0; i--) run('back from slide ' + (i + 1), D.prev);

  run('autoplay', function () { D.go(1, 0); D.next(); });
  var auto3 = (document.querySelectorAll('.slide')[2].dataset.auto || '').split(/\s+/).filter(Boolean).map(Number);
  var wait3 = (auto3.length ? Math.max.apply(null, auto3) : 600 + (D.steps(2) - 1) * 1600) + 1500;   // slide 3's last step, plus a margin
  setTimeout(function () {
    var s = D.state(), want = D.steps(2);
    if (s.slide !== 3 || s.step !== want) errors.push('autoplay: slide ' + s.slide + ' stopped at step ' + s.step + ' of ' + want);
    run('click during autoplay', function () { D.next(); D.next(); });
    var t = D.state();
    if (t.slide !== 5 || t.step !== 0) errors.push('click during autoplay: at slide ' + t.slide + ' step ' + t.step + ', expected slide 5 step 0');
    setTimeout(function () {                                // let slide 5's animations run too
      var pre = document.createElement('pre'); pre.id = 'selftest-out';
      pre.textContent = 'SELFTEST ' + n + ' slides, ' + applied + ' steps applied, autoplay checked: ' +
        (errors.length ? errors.length + ' error(s)\n' + errors.join('\n') : 'no errors');
      document.body.appendChild(pre); document.title = 'selftest done';
    }, 6000);
  }, wait3);
})();
