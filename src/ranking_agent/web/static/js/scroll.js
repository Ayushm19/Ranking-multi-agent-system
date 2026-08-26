/* Scroll reveal — IntersectionObserver, no GSAP dependency.
   Matches the ui-ux-pro-max subtle scroll-reveal guidance (small y offset). */
(() => {
  'use strict';
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const nodes = document.querySelectorAll('.reveal');
  if (!nodes.length) return;
  if (reduced || !('IntersectionObserver' in window)) {
    nodes.forEach((n) => n.classList.add('in'));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add('in');
          io.unobserve(e.target);
        }
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.12 },
  );
  nodes.forEach((n) => io.observe(n));
})();
