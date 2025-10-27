/**
 * Hook helpers to coordinate with existing global hooks injected by the WebUI.
 * Keeps runtime JS-only, no Svelte/Vite required.
 */

/** Ensure a callback runs when the WebUI is ready. */
/** @param {() => void} cb */
export function runOnUiLoaded(cb) {
  try {
    // @ts-ignore
    const h = (window.onUiLoaded || window.onUiUpdate || ((fn)=>document.addEventListener('DOMContentLoaded', fn)));
    h(typeof cb === 'function' ? cb : () => {});
  } catch {
    document.addEventListener('DOMContentLoaded', () => (typeof cb === 'function' && cb()));
  }
}
