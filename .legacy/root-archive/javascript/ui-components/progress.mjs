/** Bridge the existing progressDiv to a more composable interface. */

function queryProgress() {
  return /** @type {HTMLElement|null} */ (document.querySelector('.progressDiv'));
}

export function bootProgressBridge() {
  const bar = queryProgress();
  if (!bar) return;
  // Example: expose getter on window for extensions to subscribe
  // @ts-ignore
  window.sdw = Object.assign(window.sdw || {}, {
    progress: {
      element: bar,
      get percent() {
        const p = bar.querySelector('.progress');
        if (!p) return 0;
        const w = /** @type {HTMLElement} */(p).style.width || '0%';
        const n = parseFloat(w);
        return Number.isFinite(n) ? n / 100 : 0;
      },
    },
  });
}

