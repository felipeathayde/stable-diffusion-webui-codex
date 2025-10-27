/** DOM helpers with JSDoc types (checked by TS) */

/** @param {string} id */
export function ensureRoot(id = 'sdw-ui-root') {
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement('div');
    el.id = id;
    el.className = 'sdw-ui-root';
    document.body.appendChild(el);
  }
  return el;
}

/** @param {HTMLElement} root @param {string} name */
export function mountAt(root, name) {
  const id = `sdw-${name}`;
  let slot = root.querySelector(`#${id}`);
  if (!slot) {
    slot = document.createElement('div');
    slot.id = id;
    slot.className = `sdw-slot sdw-${name}`;
    root.appendChild(slot);
  }
  return /** @type {HTMLElement} */ (slot);
}

/** Trigger an input event for Gradio textboxes after programmatic edits. */
/** @param {HTMLElement} target */
export function updateInput(target) {
  try {
    const event = new Event('input', { bubbles: true });
    target.dispatchEvent(event);
  } catch {}
}
