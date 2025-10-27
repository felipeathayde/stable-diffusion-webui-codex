// Submit button visibility helpers (JS + JSDoc)

/** @param {string} id */
function getById(id) {
  try {
    // @ts-ignore
    const root = gradioApp();
    if (root && typeof root === 'object' && 'getElementById' in root && typeof root.getElementById === 'function') {
      const el = root.getElementById(id);
      if (el instanceof HTMLElement) return el;
    }
  } catch {}
  const fb = document.getElementById(id);
  return fb instanceof HTMLElement ? fb : null;
}

/** @param {string} tab @param {boolean} interrupt @param {boolean} skip @param {boolean} interrupting */
export function setButtons(tab, interrupt, skip, interrupting) {
  const a = getById(`${tab}_interrupt`);
  const b = getById(`${tab}_skip`);
  const c = getById(`${tab}_interrupting`);
  if (a) a.style.display = interrupt ? 'block' : 'none';
  if (b) b.style.display = skip ? 'block' : 'none';
  if (c) c.style.display = interrupting ? 'block' : 'none';
}

/** @param {string} tab @param {boolean} show */
export function showRestore(tab, show) {
  const btn = getById(`${tab}_restore_progress`);
  if (btn) btn.style.setProperty('display', show ? 'flex' : 'none', 'important');
}
