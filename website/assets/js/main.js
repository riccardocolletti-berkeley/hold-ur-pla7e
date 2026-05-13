// =============================================================
// Layout partials: header + footer defined here, injected per page.
// Single source of truth — edit only this block to change nav/footer
// across every page. Works on file://, local servers, and GitHub Pages.
// =============================================================
(function injectLayout() {
  // Absolute paths under the site root. Works because each subpage lives at
  // /<slug>/index.html (e.g. /control/), so prefixes are no longer needed.

  // Small line icons for the mobile drawer. Hidden on desktop via CSS so
  // the inline nav stays text-only as before.
  function ic(d) {
    return '<svg class="nav-ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + d + '</svg>';
  }
  var ICONS = {
    home:     '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/>',
    setup:    '<path d="M14.7 6.3a4 4 0 015.7 5l-3.4 3.4L6 25 3 22l10.3-10.3 3.4-3.4z"/><circle cx="17" cy="7" r="0.6" fill="currentColor"/>',
    physics:  '<circle cx="12" cy="12" r="2.4"/><ellipse cx="12" cy="12" rx="9" ry="3.6"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="9" ry="3.6" transform="rotate(-60 12 12)"/>',
    vision:   '<path d="M3 7.5h4l1.5-2h7l1.5 2H21v11.5H3z"/><circle cx="12" cy="13" r="3.6"/>',
    control:  '<path d="M4 6h10"/><circle cx="17" cy="6" r="1.8"/><path d="M4 12h6"/><circle cx="13" cy="12" r="1.8"/><path d="M4 18h12"/><circle cx="19" cy="18" r="1.8"/>',
    pid:      '<path d="M4 6h10"/><circle cx="17" cy="6" r="1.8"/><path d="M4 12h6"/><circle cx="13" cy="12" r="1.8"/><path d="M4 18h12"/><circle cx="19" cy="18" r="1.8"/>',
    mpc:      '<path d="M3 18h18"/><path d="M5 18v-6"/><circle cx="9" cy="12" r="1.2" fill="currentColor"/><circle cx="13" cy="9" r="1.2" fill="currentColor"/><circle cx="17" cy="6" r="1.2" fill="currentColor"/><path d="M5 12l4-3 4-3 4-3"/>',
    rl:       '<path d="M9 4a3.5 3.5 0 00-3.5 3.5c0 .5.1.9.3 1.3A3.5 3.5 0 008 15.5V18a2 2 0 002 2h4a2 2 0 002-2v-2.5a3.5 3.5 0 002.2-6.7c.2-.4.3-.8.3-1.3A3.5 3.5 0 0015 4a3.5 3.5 0 00-3 1.5A3.5 3.5 0 009 4z"/>',
    results:  '<rect x="4" y="13" width="4" height="7"/><rect x="10" y="8" width="4" height="12"/><rect x="16" y="4" width="4" height="16"/>',
    ros:      '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="12" r="2"/><path d="M8 6l8 5"/><path d="M8 18l8-5"/>',
    sim2real: '<path d="M3 10h13a4 4 0 014 4v0a4 4 0 01-4 4H10"/><path d="M7 7l-4 3 4 3"/><path d="M17 21l4-3-4-3"/>',
    team:     '<path d="M5 19l3-9h8l3 9"/><circle cx="12" cy="6" r="2.4"/><circle cx="6" cy="11" r="1.6"/><circle cx="18" cy="11" r="1.6"/>',
  };

  var navHTML =
    '<nav class="nav">' +
      '<div class="nav-inner">' +
        '<a href="/" class="brand">' +
          '<span class="dot"></span>' +
          '<span>Hold <span class="brand-acc">UR</span> Pla<span class="brand-acc">7</span>e</span>' +
        '</a>' +
        '<button class="nav-toggle" type="button" aria-label="Open menu" aria-expanded="false" aria-controls="nav-links">' +
          '<span class="nav-toggle-bar"></span>' +
          '<span class="nav-toggle-bar"></span>' +
          '<span class="nav-toggle-bar"></span>' +
        '</button>' +
        '<ul id="nav-links" class="nav-links">' +
          '<li><a href="/" data-nav="home">' + ic(ICONS.home) + 'Home</a></li>' +
          '<li><a href="/setup/" data-nav="setup">' + ic(ICONS.setup) + 'Setup</a></li>' +
          '<li><a href="/physics/" data-nav="physics">' + ic(ICONS.physics) + 'Physics</a></li>' +
          '<li><a href="/vision/" data-nav="vision">' + ic(ICONS.vision) + 'Vision</a></li>' +
          '<li class="nav-dropdown">' +
            '<span class="nav-dropdown-label">' + ic(ICONS.control) + 'Control <span class="nav-caret">▾</span></span>' +
            '<ul class="nav-dropdown-menu">' +
              '<li><a href="/control/" data-nav="control">' + ic(ICONS.pid) + 'PID</a></li>' +
              '<li><a href="/mpc/" data-nav="mpc">' + ic(ICONS.mpc) + 'MPC</a></li>' +
              '<li><a href="/rl/" data-nav="rl">' + ic(ICONS.rl) + 'Residual PPO</a></li>' +
              '<li><a href="/results/" data-nav="results">' + ic(ICONS.results) + 'Domain randomisation</a></li>' +
            '</ul>' +
          '</li>' +
          '<li><a href="/ros/" data-nav="ros">' + ic(ICONS.ros) + 'ROS</a></li>' +
          '<li><a href="/sim2real/" data-nav="sim2real">' + ic(ICONS.sim2real) + 'Sim2Real</a></li>' +
          '<li><a href="/team/" data-nav="team">' + ic(ICONS.team) + 'Code</a></li>' +
        '</ul>' +
        '<div class="nav-backdrop" aria-hidden="true"></div>' +
      '</div>' +
    '</nav>';

  var ghPath = 'M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z';

  var footerHTML =
    '<footer class="site-footer">' +
      '<div class="footer-bg" aria-hidden="true">' +
        '<svg viewBox="0 0 360 220" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">' +
          '<ellipse cx="60" cy="200" rx="32" ry="5"/>' +
          '<rect x="40" y="165" width="40" height="35" rx="4"/>' +
          '<circle cx="60" cy="150" r="14"/>' +
          '<rect x="56" y="85" width="8" height="65"/>' +
          '<circle cx="60" cy="85" r="10"/>' +
          '<rect x="60" y="81" width="120" height="8"/>' +
          '<circle cx="180" cy="85" r="8"/>' +
          '<rect x="180" y="60" width="6" height="20"/>' +
          '<rect x="140" y="55" width="80" height="5"/>' +
          '<circle cx="210" cy="50" r="4"/>' +
        '</svg>' +
      '</div>' +
      '<div class="container">' +
        '<div class="footer-grid">' +
          '<div class="footer-brand-col">' +
            '<a href="/" class="footer-logo">' +
              '<span class="dot"></span>' +
              '<span>Hold <span class="brand-acc">UR</span> Pla<span class="brand-acc">7</span>e</span>' +
            '</a>' +
            '<p class="footer-tagline">A ping-pong ball balanced on a tilted plate held by a UR7e arm. PID, MPC, and a learned residual.</p>' +
            '<a class="footer-gh-cta" href="https://github.com/riccardocolletti-berkeley/hold-ur-pla7e" target="_blank" rel="noopener">' +
              '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="' + ghPath + '"/></svg>' +
              'Source on GitHub' +
            '</a>' +
          '</div>' +
          '<div class="footer-col">' +
            '<h5>Theory</h5>' +
            '<ul>' +
              '<li><a href="/setup/">Setup</a></li>' +
              '<li><a href="/physics/">Physics</a></li>' +
              '<li><a href="/vision/">Vision</a></li>' +
            '</ul>' +
          '</div>' +
          '<div class="footer-col">' +
            '<h5>Control</h5>' +
            '<ul>' +
              '<li><a href="/control/">PID</a></li>' +
              '<li><a href="/mpc/">MPC</a></li>' +
              '<li><a href="/rl/">Residual PPO</a></li>' +
              '<li><a href="/results/">Domain randomisation</a></li>' +
            '</ul>' +
          '</div>' +
          '<div class="footer-col">' +
            '<h5>System</h5>' +
            '<ul>' +
              '<li><a href="/ros/">ROS 2</a></li>' +
              '<li><a href="/sim2real/">Sim2Real</a></li>' +
              '<li><a href="/team/">Code</a></li>' +
            '</ul>' +
          '</div>' +
        '</div>' +
        '<div class="footer-bottom">' +
          '<div class="footer-berkeley"><span class="footer-bear">🐻</span> UC Berkeley &middot; Spring 2026</div>' +
          '<div class="footer-authors">' +
            '<a href="https://collettiriccardo.com/" target="_blank" rel="noopener">Riccardo Colletti</a>' +
            '<span class="sep">·</span>' +
            '<a href="https://alexanderremmerie.com/" target="_blank" rel="noopener">Alexander Remmerie</a>' +
            '<span class="sep">·</span>' +
            'Xutao Ma' +
            '<span class="sep">·</span>' +
            'Alex Gasca Rosas' +
          '</div>' +
        '</div>' +
      '</div>' +
    '</footer>';

  function inject(id, html) {
    var el = document.getElementById(id);
    if (el) el.outerHTML = html;
  }
  inject('site-nav', navHTML);
  inject('site-footer', footerHTML);

  // Highlight the active nav item by URL slug. Pages live at /<slug>/ or
  // /<slug>/index.html; the root and bare /index.html count as home.
  var path = window.location.pathname;
  var slug;
  if (path === '/' || /\/index\.html$/.test(path)) slug = 'home';
  else slug = path.replace(/\/$/, '').replace(/^.*\//, '').replace(/\.html$/, '');
  document.querySelectorAll('[data-nav]').forEach(function (a) {
    if (a.dataset.nav === slug) a.classList.add('active');
  });

  // Mobile hamburger: toggle drawer, close on link click / outside tap / ESC.
  var navEl     = document.querySelector('.nav');
  var toggleEl  = document.querySelector('.nav-toggle');
  var linksEl   = document.querySelector('.nav-links');
  var backdrop  = document.querySelector('.nav-backdrop');
  if (navEl && toggleEl && linksEl) {
    function setOpen(open) {
      navEl.classList.toggle('is-open', open);
      toggleEl.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggleEl.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
      document.body.classList.toggle('nav-locked', open);
    }
    toggleEl.addEventListener('click', function () {
      setOpen(!navEl.classList.contains('is-open'));
    });
    if (backdrop) backdrop.addEventListener('click', function () { setOpen(false); });
    linksEl.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') setOpen(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && navEl.classList.contains('is-open')) setOpen(false);
    });
    // Auto-close if the viewport grows past the mobile breakpoint while open.
    window.matchMedia('(min-width: 921px)').addEventListener('change', function (m) {
      if (m.matches) setOpen(false);
    });
  }
})();

// =============================================================
// Reveal-on-scroll using IntersectionObserver
// All reveal classes: .reveal, .reveal-l, .reveal-r, .reveal-scale, .stagger
// =============================================================
const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    }
  },
  { rootMargin: '0px 0px -8% 0px', threshold: 0.12 }
);

document.querySelectorAll(
  '.reveal, .reveal-l, .reveal-r, .reveal-scale, .stagger'
).forEach((el) => revealObserver.observe(el));

// =============================================================
// Sticky nav scroll state + top progress bar
// =============================================================
const nav = document.querySelector('.nav');
const progress = document.createElement('div');
progress.className = 'scroll-progress';
document.body.appendChild(progress);

let scrolled = false;
function onScroll() {
  const y = window.scrollY;
  // nav state
  const isScrolled = y > 8;
  if (nav && isScrolled !== scrolled) {
    scrolled = isScrolled;
    nav.classList.toggle('scrolled', scrolled);
  }
  // progress
  const h = document.documentElement.scrollHeight - window.innerHeight;
  const pct = h > 0 ? (y / h) * 100 : 0;
  progress.style.width = pct + '%';
}
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

// =============================================================
// Subtle parallax for hero scene (mouse-follow)
// =============================================================
const scene = document.querySelector('.scene-frame');
if (scene && window.matchMedia('(min-width: 980px)').matches) {
  const heroEl = document.querySelector('.hero');
  let rafPending = false;
  let mx = 0, my = 0;
  heroEl.addEventListener('mousemove', (e) => {
    const r = heroEl.getBoundingClientRect();
    mx = ((e.clientX - r.left) / r.width  - 0.5) * 2;
    my = ((e.clientY - r.top)  / r.height - 0.5) * 2;
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        scene.style.setProperty('--mx', `${mx * 8}deg`);
        scene.style.setProperty('--my', `${-my * 8}deg`);
      });
    }
  });
}

// =============================================================
// Animated counter for stats
// =============================================================
const counters = document.querySelectorAll('[data-counter]');
if (counters.length) {
  const counterObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const el = entry.target;
      const target = parseFloat(el.dataset.counter);
      const decimals = parseInt(el.dataset.decimals || '0', 10);
      const duration = 1400;
      const start = performance.now();
      const ease = (t) => 1 - Math.pow(1 - t, 3);
      const tick = (now) => {
        const t = Math.min(1, (now - start) / duration);
        const v = target * ease(t);
        el.textContent = v.toLocaleString(undefined, {
          minimumFractionDigits: decimals,
          maximumFractionDigits: decimals,
        });
        if (t < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
      counterObserver.unobserve(el);
    }
  }, { threshold: 0.4 });
  counters.forEach((el) => counterObserver.observe(el));
}

// =============================================================
// Hero arm video → start closed-loop diagram once the 2 laps finish
// =============================================================
(function () {
  const video   = document.getElementById('hero-arm-video');
  const diagram = document.querySelector('.loop-diagram');
  if (!video || !diagram) return;

  // SMIL <animate> / <animateMotion> elements: pause the entire SVG clock.
  // pauseAnimations / unpauseAnimations are SVG SVGSVGElement methods.
  try { diagram.pauseAnimations(); } catch (_) {}

  function startDiagram() {
    diagram.classList.add('running');         // CSS-driven gear/tilt/pulse
    try { diagram.unpauseAnimations(); } catch (_) {}
    video.removeEventListener('ended', startDiagram);
  }

  video.addEventListener('ended', startDiagram);
  // Safety net: if the video can't autoplay or errors out, kick off the
  // diagram after a short delay so the page isn't left half-static.
  video.addEventListener('error',   () => setTimeout(startDiagram, 500));
  setTimeout(() => {
    if (video.readyState < 2) startDiagram();   // video never even loaded
  }, 38000);
})();
