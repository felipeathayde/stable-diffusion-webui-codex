// Tab helpers extracted from ui.js (JS + JSDoc, no build)

/** @returns {Document | ShadowRoot | HTMLElement} */
function root() {
  try { return gradioApp(); } catch { return document; }
}

/**
 * Retrieve an element by id from the Gradio root with document fallback.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
export function getAppElementById(id) {
  const r = root();
  if ('getElementById' in r && typeof r.getElementById === 'function') {
    const el = /** @type {any} */(r).getElementById(id);
    if (el instanceof HTMLElement) return el;
  }
  const fallback = document.getElementById(id);
  return fallback instanceof HTMLElement ? fallback : null;
}

/** Clicks the main "txt2img" tab. */
import { publish, types } from './events.mjs';

export function switchToTxt2Img() {
  const tabs = root().querySelector('#tabs');
  const buttons = tabs ? tabs.querySelectorAll('button') : null;
  buttons?.[0]?.click();
  try { publish(types.TabsChange, { tab: 'txt2img', index: 0 }); } catch {}
}

/** Clicks the main "img2img" tab. */
export function switchToImg2Img() {
  switchToImg2ImgTab(0);
}

/** Clicks the img2img sub-tab at a given index.
 *  @param {number} index
 */
export function switchToImg2ImgTab(index) {
  const tabs = root().querySelector('#tabs');
  const buttons = tabs ? tabs.querySelectorAll('button') : null;
  buttons?.[1]?.click();
  const mode = getAppElementById('mode_img2img');
  if (mode) {
    const modeButtons = mode.querySelectorAll('button');
    const button = modeButtons?.[index];
    if (button instanceof HTMLElement) button.click();
  }
  try { publish(types.TabsChange, { tab: 'img2img', index }); } catch {}
}

/** Convenience switches for common img2img modes. */
export function switchToSketch() { switchToImg2ImgTab(1); }
export function switchToInpaint() { switchToImg2ImgTab(2); }
export function switchToInpaintSketch() { switchToImg2ImgTab(3); }

/** Clicks the main "Extras" tab. */
export function switchToExtras() {
  const tabs = root().querySelector('#tabs');
  const buttons = tabs ? tabs.querySelectorAll('button') : null;
  buttons?.[3]?.click();
  try { publish(types.TabsChange, { tab: 'extras', index: 3 }); } catch {}
}

/**
 * Reads the selected index within a button tab group.
 * @param {string} tabId
 * @returns {number}
 */
export function getTabIndex(tabId) {
  const tab = getAppElementById(tabId);
  const buttons = tab ? tab.querySelector('div')?.querySelectorAll('button') : null;
  if (!buttons) return 0;
  const arr = Array.from(buttons);
  for (let i = 0; i < arr.length; i += 1) {
    const btn = arr[i];
    if (btn instanceof HTMLElement && btn.classList.contains('selected')) return i;
  }
  return 0;
}

/**
 * Sets the first argument to the selected index for a given tab.
 * @param {string} tabId
 * @param {IArguments | ArrayLike<unknown>} args
 * @returns {unknown[]}
 */
export function createTabIndexArgs(tabId, args) {
  const res = Array.from(/** @type {any} */(args));
  res[0] = getTabIndex(tabId);
  return res;
}

/**
 * Legacy helper used by img2img to pass the current sub-tab index.
 * It mirrors ui.js behaviour: drop last two items and set index at [0].
 * @param {IArguments | ArrayLike<unknown>} args
 * @returns {unknown[]}
 */
export function getImg2ImgTabIndex(args) {
  const res = Array.from(/** @type {any} */(args));
  // keep parity with ui.js
  res.splice(-2);
  res[0] = getTabIndex('mode_img2img');
  return res;
}
