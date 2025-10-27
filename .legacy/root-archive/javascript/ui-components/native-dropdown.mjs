import { runOnUiLoaded } from './hooks.mjs';

/** @type {Map<Element, { disconnect: ()=>void }>} */
const hydrated = new Map();

/** @param {HTMLInputElement | HTMLTextAreaElement} el */
function dispatchValue(el) {
  try {
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  } catch {}
}

/** @param {Element} node */
function hydrate(node) {
  if (!(node instanceof HTMLElement)) return;
  const targetId = node.dataset.target;
  if (!targetId) return;
  const app = typeof gradioApp === 'function' ? gradioApp() : document;
  const targetRoot = app?.querySelector(`#${CSS.escape(targetId)}`);
  if (!(targetRoot instanceof HTMLElement)) return;
  const carrier = /** @type {HTMLInputElement | HTMLTextAreaElement | null} */ (targetRoot.querySelector('input, textarea'));
  if (!carrier) return;

  const input = /** @type {HTMLInputElement | null} */ (node.querySelector('.sdw-native-dropdown__input'));
  const toggle = /** @type {HTMLButtonElement | null} */ (node.querySelector('.sdw-native-dropdown__toggle'));
  const datalist = /** @type {HTMLDataListElement | null} */ (node.querySelector('datalist'));
  if (!input || !toggle || !datalist) return;

  if (hydrated.has(node)) {
    hydrated.get(node)?.disconnect();
    hydrated.delete(node);
  }

  const allowCustom = node.dataset.allowCustom === '1';
  const optionValues = () => new Set(Array.from(datalist.options || []).map((opt) => opt.value ?? ''));

  const syncFromCarrier = () => {
    if (carrier.value !== input.value) {
      input.value = carrier.value;
    }
  };

  /** @param {string} nextValue */
  const applyValue = (nextValue) => {
    carrier.value = nextValue;
    dispatchValue(carrier);
  };

  const ensureValue = () => {
    const val = input.value;
    const values = optionValues();
    if (!allowCustom && val && !values.has(val)) {
      const fallback = carrier.value || Array.from(values)[0] || '';
      input.value = fallback;
      applyValue(fallback);
      return;
    }
    applyValue(val);
  };

  input.addEventListener('change', ensureValue);
  input.addEventListener('blur', ensureValue);
  input.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      ev.preventDefault();
      ensureValue();
      input.blur();
    }
  });

  toggle.addEventListener('click', (ev) => {
    ev.preventDefault();
    input.focus();
    if (typeof input.showPicker === 'function') {
      try { input.showPicker(); } catch {}
    }
  });

  syncFromCarrier();

  const carrierObserver = new MutationObserver(syncFromCarrier);
  carrierObserver.observe(carrier, { attributes: true, attributeFilter: ['value'] });

  const datalistObserver = new MutationObserver(() => {
    if (!allowCustom) ensureValue();
  });
  datalistObserver.observe(datalist, { childList: true });

  hydrated.set(node, {
    disconnect() {
      carrierObserver.disconnect();
      datalistObserver.disconnect();
    }
  });
}

function scan() {
  const app = typeof gradioApp === 'function' ? gradioApp() : document;
  if (!app) return;
  app.querySelectorAll('.sdw-native-dropdown').forEach((node) => {
    if (!hydrated.has(node)) {
      hydrate(node);
    }
  });
  hydrated.forEach((/** @type {{disconnect?:()=>void}} */ handle, /** @type {Element} */ node) => {
    if (!node.isConnected) {
      handle?.disconnect?.();
      hydrated.delete(node);
    }
  });
}

export function install() {
  runOnUiLoaded(() => {
    scan();
    const app = typeof gradioApp === 'function' ? gradioApp() : document;
    if (!app) return;
    const observer = new MutationObserver(() => scan());
    observer.observe(app, { childList: true, subtree: true });
  });
}
