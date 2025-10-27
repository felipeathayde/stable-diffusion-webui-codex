import { runOnUiLoaded } from './hooks.mjs';

const HOST_SELECTOR = '.sdw-dropdown-host[data-sdw-dropdown]';

/** @typedef {{ id: string, label?: string, choices: Array<{value:string,label:string}>, value?: string | string[], allow_custom?: boolean, multiselect?: boolean, interactive?: boolean, placeholder?: string }} DropdownConfig */

/** @type {Map<string, { value: string|string[], node: HTMLElement, control: HTMLElement, config: DropdownConfig }>} */
const registry = new Map();

/** @type {Map<string, Set<(detail: {id:string,value:string|string[],source:string})=>void>>} */
const subscribers = new Map();

function emitChange(id, value, source) {
  const detail = { id, value, source };
  document.dispatchEvent(new CustomEvent('sdw:dropdown-change', { detail }));
  const subs = subscribers.get(id);
  if (subs) {
    for (const fn of subs) {
      try { fn(detail); } catch (err) { console.warn('[SDW:dropdown] subscriber error', err); }
    }
  }
}

function setState(entry, value, source) {
  const normalized = entry.config.multiselect ? (Array.isArray(value) ? value : [value].filter(Boolean)) : (Array.isArray(value) ? value[0] ?? '' : value ?? '');
  if (JSON.stringify(entry.value) === JSON.stringify(normalized)) return;
  entry.value = normalized;
  emitChange(entry.config.id, normalized, source);
}

function buildSelectSingle(config, entry) {
  const select = document.createElement('select');
  select.className = 'sdw-dropdown-control';
  select.disabled = config.interactive === false;

  for (const choice of config.choices) {
    const option = document.createElement('option');
    option.value = choice.value;
    option.textContent = choice.label;
    select.appendChild(option);
  }

  if (typeof config.value === 'string') {
    select.value = config.value;
  }

  select.addEventListener('change', () => {
    setState(entry, select.value, 'user');
  });

  return select;
}

function buildSelectMultiple(config, entry) {
  const select = document.createElement('select');
  select.className = 'sdw-dropdown-control';
  select.multiple = true;
  select.disabled = config.interactive === false;

  for (const choice of config.choices) {
    const option = document.createElement('option');
    option.value = choice.value;
    option.textContent = choice.label;
    select.appendChild(option);
  }

  if (Array.isArray(config.value)) {
    for (const v of config.value) {
      const option = select.querySelector(`option[value="${CSS.escape(v)}"]`);
      if (option) option.selected = true;
    }
  }

  select.addEventListener('change', () => {
    const selected = Array.from(select.selectedOptions).map(opt => opt.value);
    setState(entry, selected, 'user');
  });

  return select;
}

function buildInputWithDatalist(config, entry) {
  const wrapper = document.createElement('div');
  wrapper.className = 'sdw-dropdown-datalist';

  const input = document.createElement('input');
  input.className = 'sdw-dropdown-input';
  input.autocomplete = 'off';
  input.placeholder = config.placeholder || '';
  input.disabled = config.interactive === false;

  const listId = `${config.id}-list`;
  const datalist = document.createElement('datalist');
  datalist.id = listId;

  for (const choice of config.choices) {
    const option = document.createElement('option');
    option.value = choice.value;
    option.label = choice.label;
    datalist.appendChild(option);
  }

  if (typeof config.value === 'string') {
    input.value = config.value;
  }

  input.setAttribute('list', listId);

  input.addEventListener('change', () => {
    setState(entry, input.value, 'user');
  });
  input.addEventListener('blur', () => {
    setState(entry, input.value, 'user');
  });

  wrapper.appendChild(input);
  wrapper.appendChild(datalist);
  return { wrapper, input };
}

function mountDropdown(node, config) {
  if (!config?.id) {
    console.warn('[SDW:dropdown] configuration missing id', config);
    return;
  }

  const existing = registry.get(config.id);
  if (existing && existing.node !== node) {
    existing.node.replaceChildren();
    registry.delete(config.id);
  }

  node.innerHTML = '';
  node.classList.add('sdw-dropdown-mounted');

  /** @type {{ value: string|string[], node: HTMLElement, control: HTMLElement, config: DropdownConfig }} */
  const entry = {
    value: config.multiselect ? (Array.isArray(config.value) ? [...config.value] : []) : (typeof config.value === 'string' ? config.value : ''),
    node,
    control: node,
    config,
  };

  let control;

  if (config.multiselect) {
    control = buildSelectMultiple(config, entry);
  } else if (config.allow_custom) {
    const { wrapper, input } = buildInputWithDatalist(config, entry);
    control = wrapper;
    entry.control = input;
  } else {
    control = buildSelectSingle(config, entry);
  }

  node.appendChild(control);
  entry.control = config.allow_custom ? control.querySelector('input') || control : control;

  registry.set(config.id, entry);
  setState(entry, entry.value, 'init');
}

function parsePayload(node) {
  try {
    const raw = node.getAttribute('data-sdw-dropdown');
    if (!raw) return null;
    /** @type {DropdownConfig} */
    const payload = JSON.parse(raw);
    payload.choices = Array.isArray(payload.choices) ? payload.choices.map(opt => ({ value: String(opt.value), label: String(opt.label ?? opt.value) })) : [];
    return payload;
  } catch (err) {
    console.warn('[SDW:dropdown] failed to parse payload', err);
    return null;
  }
}

function scan(root = document) {
  const nodes = root.querySelectorAll(HOST_SELECTOR);
  nodes.forEach((node) => {
    const payload = parsePayload(node);
    if (!payload) return;
    mountDropdown(/** @type {HTMLElement} */ (node), payload);
  });
}

export function install() {
  runOnUiLoaded(() => {
    scan();
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        mutation.addedNodes.forEach((node) => {
          if (node instanceof HTMLElement) {
            if (node.matches(HOST_SELECTOR)) {
              const payload = parsePayload(node);
              if (payload) mountDropdown(node, payload);
            } else if (node.querySelector) {
              scan(node);
            }
          }
        });
        if (mutation.target instanceof HTMLElement && mutation.target.matches(HOST_SELECTOR)) {
          const payload = parsePayload(mutation.target);
          if (payload) mountDropdown(mutation.target, payload);
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-sdw-dropdown'] });
  });
}

export function getValue(id) {
  const entry = registry.get(id);
  return entry ? entry.value : undefined;
}

export function setValue(id, value) {
  const entry = registry.get(id);
  if (!entry) return;
  if (entry.config.multiselect) {
    const select = entry.node.querySelector('select');
    if (select) {
      const arr = Array.isArray(value) ? value : [value];
      Array.from(select.options).forEach(opt => { opt.selected = arr.includes(opt.value); });
      setState(entry, arr, 'programmatic');
    }
  } else if (entry.config.allow_custom) {
    const input = entry.node.querySelector('input');
    if (input) {
      input.value = typeof value === 'string' ? value : Array.isArray(value) ? (value[0] ?? '') : '';
      setState(entry, input.value, 'programmatic');
    }
  } else {
    const select = entry.node.querySelector('select');
    if (select) {
      const val = typeof value === 'string' ? value : Array.isArray(value) ? (value[0] ?? '') : '';
      select.value = val;
      setState(entry, val, 'programmatic');
    }
  }
}

export function update(id, config) {
  const entry = registry.get(id);
  const node = entry?.node || document.querySelector(`.sdw-dropdown-host[data-sdw-dropdown*="\"${id}\""]`);
  if (!node) return;
  const payload = { ...(entry?.config || parsePayload(node) || {}), ...config, id };
  node.setAttribute('data-sdw-dropdown', JSON.stringify(payload));
  mountDropdown(node, payload);
}

export function on(id, handler) {
  const set = subscribers.get(id) || new Set();
  set.add(handler);
  subscribers.set(id, set);
  return () => {
    set.delete(handler);
    if (set.size === 0) subscribers.delete(id);
  };
}

export function list() {
  return Array.from(registry.entries()).map(([id, entry]) => ({ id, value: entry.value, config: entry.config }));
}

export default {
  install,
  getValue,
  setValue,
  update,
  on,
  list,
};
