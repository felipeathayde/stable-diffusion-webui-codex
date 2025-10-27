// Clipboard helpers: install resolution paste handler for width/height inputs
import { getAppElementById as $id } from './readers.mjs';
import { runOnUiLoaded } from './hooks.mjs';
import { updateInput } from './dom.mjs';

/** Configure the width and height elements on `tab` to accept pasting of resolutions. */
/** @param {string} tab */
export function setupResolutionPasting(tab) {
  const w = /** @type {HTMLInputElement|null} */ ($id(`${tab}_width`)?.querySelector('input[type=number]'));
  const h = /** @type {HTMLInputElement|null} */ ($id(`${tab}_height`)?.querySelector('input[type=number]'));
  [w, h].forEach((el) => {
    if (!(el instanceof HTMLInputElement)) return;
    el.addEventListener('paste', (event) => {
      const text = event.clipboardData?.getData('text/plain') ?? '';
      const match = text.match(/^\s*(\d+)\D+(\d+)\s*$/);
      if (match && match[1] && match[2]) {
        if (w) { w.value = match[1]; updateInput(w); }
        if (h) { h.value = match[2]; updateInput(h); }
        event.preventDefault();
      }
    });
  });
}

export function install() {
  runOnUiLoaded(() => {
    try { setupResolutionPasting('txt2img'); } catch {}
    try { setupResolutionPasting('img2img'); } catch {}
  });
}
