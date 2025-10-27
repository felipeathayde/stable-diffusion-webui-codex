// Gallery helpers (JS + JSDoc types for TS checking)

/** @returns {Document | ShadowRoot | HTMLElement} */
function grRoot() {
  try { return gradioApp(); } catch { return document; }
}

/** @returns {HTMLElement[]} */
export function allThumbButtons() {
  const buttons = grRoot().querySelectorAll('[style="display: block;"] .thumbnail-item.thumbnail-small');
  /** @type {HTMLElement[]} */
  const visible = [];
  buttons.forEach((elem) => { if (elem instanceof HTMLElement && elem.parentElement?.offsetParent) visible.push(elem); });
  return visible;
}

import { publish, types } from './events.mjs';

/** @returns {HTMLElement | null} */
export function selectedButton() { return allThumbButtons().find((el) => el.classList.contains('selected')) ?? null; }

/** @returns {number} */
export function selectedIndex() { return allThumbButtons().findIndex((el) => el.classList.contains('selected')); }

/** @param {string} containerId */
export function containerButtons(containerId) { return grRoot().querySelectorAll(`#${containerId} .thumbnail-item.thumbnail-small`); }

/** @param {string} containerId */
export function selectedIndexIn(containerId) {
  return Array.from(containerButtons(containerId)).findIndex((el) => el.classList.contains('selected'));
}

/** @template T @param {T[]} gallery */
export function extractSelected(gallery) {
  if (!Array.isArray(gallery) || gallery.length === 0) return [[null]];
  let idx = selectedIndex();
  if (idx < 0 || idx >= gallery.length) idx = 0;
  // @ts-ignore
  return [[gallery[idx] ?? null]];
}

// Install a delegated click listener to emit gallery:select events.
let _installed = false;
export function install() {
  if (_installed) return; _installed = true;
  const root = grRoot();
  try {
    root.addEventListener('click', (ev) => {
      const path = ev.composedPath?.() || [];
      const target = /** @type {Element|undefined} */ (path.find((n) => n instanceof Element));
      if (!target) return;
      const item = /** @type {HTMLElement|null} */ (target.closest?.('.thumbnail-item.thumbnail-small'));
      if (!item) return;
      // Compute index and container id
      const container = /** @type {HTMLElement|null} */ (item.closest?.('[id]'));
      let containerId = container?.id || '';
      let index = -1;
      if (container) {
        const buttons = Array.from(container.querySelectorAll('.thumbnail-item.thumbnail-small'));
        index = buttons.findIndex((el) => el === item);
      }
      try { publish(types.GallerySelect, { containerId, index }); } catch {}
    }, { capture: true });
  } catch {}
}
