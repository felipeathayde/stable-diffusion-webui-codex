// DOM readers extracted from ui.js (JS + JSDoc, no build)

/** @returns {Document | ShadowRoot | HTMLElement} */
function root() { try { return gradioApp(); } catch { return document; } }

/** @param {string} id */
export function getAppElementById(id) {
  const r = root();
  if ('getElementById' in r && typeof r.getElementById === 'function') {
    const el = /** @type {any} */(r).getElementById(id);
    if (el instanceof HTMLElement) return el;
  }
  const fallback = document.getElementById(id);
  return fallback instanceof HTMLElement ? fallback : null;
}

/** @param {string} id */
export function readText(id) {
  const el = getAppElementById(id);
  if (!el) return '';
  const ta = el.querySelector('textarea');
  if (ta && ta instanceof HTMLTextAreaElement) return ta.value ?? '';
  const inp = el.querySelector('input[type=text]');
  if (inp && inp instanceof HTMLInputElement) return inp.value ?? '';
  return '';
}

/** @param {string} id */
export function readNumber(id) {
  const el = getAppElementById(id);
  if (!el) return 0;
  const num = el.querySelector('input[type=number]');
  if (num && num instanceof HTMLInputElement) return Number(num.value || 0);
  const rng = el.querySelector('input[type=range]');
  if (rng && rng instanceof HTMLInputElement) return Number(rng.value || 0);
  return 0;
}

/** @param {string} id */
export const readFloat = (id) => Number(readNumber(id));
/** @param {string} id */
export const readInt = (id) => Math.trunc(Number(readNumber(id)));

/** @param {string} id */
export function isInteractive(id) {
  const el = getAppElementById(id);
  if (!el) return false;
  if (el instanceof HTMLElement && el.offsetParent === null) return false;
  const disabled = el.querySelector('input[disabled], select[disabled], textarea[disabled]');
  return !disabled;
}

/** @param {string} id */
export function readCheckbox(id) {
  const el = getAppElementById(id);
  if (!el) return false;
  const cb = el.querySelector('input[type=checkbox]');
  return !!(cb && cb instanceof HTMLInputElement && cb.checked);
}

/** @param {string} id */
export function readDropdownValue(id) {
  const el = getAppElementById(id);
  if (!el) return '';
  // Our HTML dropdown helper mounts either a <select> (single/multi)
  // or an <input class="sdw-dropdown-input" list="..."> for allow_custom.
  const sel = el.querySelector('select');
  if (sel instanceof HTMLSelectElement) {
    if (sel.multiple) return Array.from(sel.selectedOptions).map(o => o.value);
    return sel.value;
  }
  const inp = el.querySelector('input.sdw-dropdown-input, .sdw-dropdown-datalist input');
  if (inp instanceof HTMLInputElement) {
    return inp.value ?? '';
  }
  return '';
}

/** @param {string} id */
export function readRadioIndex(id) {
  const el = getAppElementById(id);
  if (!el) return 0;
  const buttons = el.querySelectorAll('button');
  let idx = 0;
  buttons.forEach((btn, i) => {
    if (btn instanceof HTMLElement && btn.classList.contains('selected')) idx = i;
  });
  return idx;
}

/** @param {string} id */
export function readRadioValue(id) {
  const el = getAppElementById(id);
  if (!el) return '';
  const buttons = el.querySelectorAll('button');
  let value = '';
  buttons.forEach((btn) => {
    if (btn instanceof HTMLElement && btn.classList.contains('selected')) value = btn.textContent?.trim() ?? '';
  });
  return value;
}

/** @param {string} id */
export function readDropdownOrRadioValue(id) {
  const dd = readDropdownValue(id);
  if (typeof dd === 'string' && dd !== '') return dd;
  return readRadioValue(id);
}

/** @param {string} id */
export function readSeedValue(id) {
  const el = getAppElementById(id);
  if (!el) return -1;
  const num = el.querySelector('input[type=number]');
  if (num && num instanceof HTMLInputElement) {
    const v = Number(num.value || -1);
    return Number.isFinite(v) ? Math.trunc(v) : -1;
  }
  const text = el.querySelector('input[type=text]');
  if (text && text instanceof HTMLInputElement) {
    const t = (text.value || '').trim();
    if (/^-?\d+$/.test(t)) return Number.parseInt(t, 10);
    return -1;
  }
  return -1;
}
