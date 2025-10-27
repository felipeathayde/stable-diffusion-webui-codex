// Options watcher: parse settings_json, expose window.opts, update dependent UI bits
import { runOnUiLoaded } from './hooks.mjs';
import { publish, on, types } from './events.mjs';
import { getAppElementById as $id } from './readers.mjs';

export function install() {
  runOnUiLoaded(() => {
    try {
      // Avoid duplicate installs
      if (typeof window !== 'undefined' && window.__SDW_OPTS_INSTALLED__) return;
      // @ts-ignore
      window.__SDW_OPTS_INSTALLED__ = true;
    } catch {}

    const jsonElem = $id('settings_json');
    if (!jsonElem) return;
    const textarea = jsonElem.querySelector('textarea');
    if (!(textarea instanceof HTMLTextAreaElement)) return;

    // Initialize opts
    try {
      // @ts-ignore
      window.opts = JSON.parse(textarea.value) || {};
      try { publish(types.OptionsUpdate, { scope: 'init', keys: Object.keys(window.opts || {}) }); } catch {}
    } catch (e) {
      console.error('[Options] Failed to parse settings_json:', e);
      // @ts-ignore
      window.opts = window.opts || {};
    }

    // Fire legacy callbacks if present
    try {
      // @ts-ignore
      if (typeof window.executeCallbacks === 'function') {
        // @ts-ignore
        window.executeCallbacks(window.optionsAvailableCallbacks, undefined, 'onOptionsAvailable');
        // @ts-ignore
        window.executeCallbacks(window.optionsChangedCallbacks, undefined, 'onOptionsChanged');
      }
    } catch {}

    // Intercept value set to keep opts in sync
    try {
      const desc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
      if (desc) {
        const originalGet = desc.get?.bind(textarea);
        const originalSet = desc.set?.bind(textarea);
        Object.defineProperty(textarea, 'value', {
          set(newValue) {
            const oldValue = typeof originalGet === 'function' ? originalGet() : textarea.value;
            if (typeof originalSet === 'function') originalSet(newValue);
            if (oldValue !== newValue) {
              try {
                // @ts-ignore
                window.opts = JSON.parse(textarea.value) || window.opts || {};
                try { publish(types.OptionsUpdate, { scope: 'change', keys: Object.keys(window.opts || {}) }); } catch {}
              } catch (e) {
                console.error('[Options] Failed to parse updated settings_json:', e);
              }
            }
            try {
              // @ts-ignore
              if (typeof window.executeCallbacks === 'function') window.executeCallbacks(window.optionsChangedCallbacks, undefined, 'onOptionsChanged');
            } catch {}
            updateCheckpointHash();
          },
          get() {
            return typeof originalGet === 'function' ? originalGet() : textarea.value;
          }
        });
      }
    } catch {}

    // Hide raw settings JSON area
    try { if (jsonElem.parentElement?.style) jsonElem.parentElement.style.display = 'none'; } catch {}

    // Initial UI updates dependent on opts
    updateCheckpointHash();
    try { applyUiDefaults('init'); } catch {}

    // React to OptionsUpdate events (from ourselves or others)
    try { on(types.OptionsUpdate, (ev)=>{ try { applyUiDefaults(ev?.detail?.scope || 'change', ev?.detail); } catch {} }); } catch {}
  });
}

function updateCheckpointHash() {
  try {
    const elem = $id('sd_checkpoint_hash');
    // @ts-ignore
    const hash = typeof window.opts?.sd_checkpoint_hash === 'string' ? window.opts.sd_checkpoint_hash : '';
    const shortHash = hash.substring(0, 10);
    if (elem && elem.textContent !== shortHash) {
      elem.textContent = shortHash;
      elem.title = hash;
      elem.setAttribute('href', `https://google.com/search?q=${hash}`);
    }
  } catch {}
}

/** @param {HTMLElement | null} selectOrRadioContainer @param {unknown} value */
function _selectValue(selectOrRadioContainer, value) {
  if (!(selectOrRadioContainer instanceof HTMLElement)) return;
  // Handle <select>
  const select = selectOrRadioContainer instanceof HTMLSelectElement ? selectOrRadioContainer : selectOrRadioContainer.querySelector('select');
  if (select instanceof HTMLSelectElement) {
    const prev = select.value;
    if (prev !== value && typeof value === 'string' && value.length) {
      // avoid stomping manual user choice: honor a marker when user interacted
      if (select.dataset && select.dataset.sdwUserSet === '1') return;
      const opt = Array.from(select.options).find(o => o.value === value || o.textContent === value);
      if (opt) { select.value = opt.value; select.dispatchEvent(new Event('change', { bubbles: true })); }
    }
    // mark manual changes
    select.addEventListener('change', ()=>{ try { select.dataset.sdwUserSet = '1'; } catch {} }, { once: false });
    return;
  }
  // Handle radio group inside container
  const inputs = selectOrRadioContainer.querySelectorAll('input[type=radio][value]');
  if (inputs && inputs.length) {
    const arr = Array.from(inputs);
    const curr = arr.find(i => i instanceof HTMLInputElement && i.checked);
    if (typeof value === 'string' && value.length) {
      const target = /** @type {HTMLInputElement | undefined} */ (arr.find(i => i instanceof HTMLInputElement && (i.value === value || i.getAttribute('aria-label') === value)));
      if (target && target !== curr) {
        // mark user-set to avoid override if changed later
        const key = 'data-sdw-user-set';
        if (selectOrRadioContainer.getAttribute(key) === '1') return;
        if (target instanceof HTMLInputElement && typeof target.click === 'function') target.click();
      }
    }
    selectOrRadioContainer.addEventListener('change', ()=>{ try { selectOrRadioContainer.setAttribute('data-sdw-user-set','1'); } catch {} }, { once: false });
  }
}

/** @typedef {{ sampler: unknown; scheduler: unknown }} PresetPair */
/** @typedef {{ t2i: PresetPair; i2i: PresetPair }} PresetDefaults */

/** @param {'init'|'change'} [scope] @param {{ keys?: string[] }} [detail] */
function applyUiDefaults(scope = 'change', detail) {
  // Read current options
  /** @type {Record<string, unknown>} */
  // @ts-ignore - global opts comes from backend
  const opts = window.opts || {};
  const preset = String(opts.forge_preset || 'sd'); // 'sd' | 'xl' | 'flux' | 'all'
  /** @type {Record<string, PresetDefaults>} */
  const presetMap = {
    sd: {
      t2i: { sampler: opts.sd_t2i_sampler, scheduler: opts.sd_t2i_scheduler },
      i2i: { sampler: opts.sd_i2i_sampler, scheduler: opts.sd_i2i_scheduler },
    },
    xl: {
      t2i: { sampler: opts.xl_t2i_sampler, scheduler: opts.xl_t2i_scheduler },
      i2i: { sampler: opts.xl_i2i_sampler, scheduler: opts.xl_i2i_scheduler },
    },
    flux: {
      t2i: { sampler: opts.flux_t2i_sampler, scheduler: opts.flux_t2i_scheduler },
      i2i: { sampler: opts.flux_i2i_sampler, scheduler: opts.flux_i2i_scheduler },
    },
  };
  const def = /** @type {PresetDefaults | undefined} */ (presetMap[preset] || presetMap.sd);
  if (!def) return;
  const t2iDefaults = def.t2i || /** @type {PresetPair} */ ({ sampler: '', scheduler: '' });
  const i2iDefaults = def.i2i || /** @type {PresetPair} */ ({ sampler: '', scheduler: '' });

  // Only set on init or when relevant keys change
  const detailKeys = detail && Array.isArray(detail.keys) ? detail.keys : null;
  const keys = detailKeys ? detailKeys.slice() : [];
  const relevant = scope === 'init' || keys.some(k => /^(forge_preset|sd_|xl_|flux_).*(sampler|scheduler)$/i.test(String(k)));
  if (!relevant) return;

  try {
    // txt2img controls
    const tSampler = $id('txt2img_sampling');
    const tScheduler = $id('txt2img_scheduler');
    _selectValue(tSampler, String(t2iDefaults.sampler || ''));
    _selectValue(tScheduler, String(t2iDefaults.scheduler || ''));
  } catch {}
  try {
    // img2img controls
    const iSampler = $id('img2img_sampling');
    const iScheduler = $id('img2img_scheduler');
    _selectValue(iSampler, String(i2iDefaults.sampler || ''));
    _selectValue(iScheduler, String(i2iDefaults.scheduler || ''));
  } catch {}
}
