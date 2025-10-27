// Submit/restore flow helpers with safe fallbacks to legacy globals.

import { getAppElementById as $id } from './readers.mjs';
import { publish, types } from './events.mjs';

/** @typedef {import('./queue.mjs').ProgressResponse} ProgressResponse */

/** @param {string} k @param {string} v */
function localSet(k, v) { try { localStorage.setItem(k, v); } catch {} }
/** @param {string} k */
function localGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
/** @param {string} k */
function localRemove(k) { try { localStorage.removeItem(k); } catch {} }

/** @param {string} tab @param {boolean} show */
function showButtons(tab, show) {
  const a = $id(`${tab}_interrupt`);
  const b = $id(`${tab}_skip`);
  const c = $id(`${tab}_interrupting`);
  if (a) a.style.display = show ? 'none' : 'block';
  if (b) b.style.display = show ? 'none' : 'block';
  if (c) c.style.display = 'none';
}

/** @param {string} tab */
function showInterrupting(tab) {
  const a = $id(`${tab}_interrupt`);
  const b = $id(`${tab}_skip`);
  const c = $id(`${tab}_interrupting`);
  if (a) a.style.display = 'none';
  if (b) b.style.display = 'block';
  if (c) c.style.display = 'block';
}

/** @param {string} tab @param {boolean} show */
function showRestore(tab, show) {
  const btn = $id(`${tab}_restore_progress`);
  if (btn) btn.style.setProperty('display', show ? 'flex' : 'none', 'important');
}

/**
 * @param {string} tab
 * @param {string} id
 * @param {boolean} [resumed]
 * @returns {(res: ProgressResponse) => void}
 */
function makeProgressHandler(tab, id, resumed = false) {
  return function progressHandler(res) {
    try {
      const detail = resumed ? { tab, id, resumed: true, res } : { tab, id, res };
      publish(types.ProgressUpdate, detail);
    } catch {}
  };
}

/** Start progress for a tab and return the id. */
/**
 * @param {string} tab
 * @param {string} galleryContainerId
 * @param {string} galleryId
 * @param {() => void} [onDone]
 */
export function start(tab, galleryContainerId, galleryId, onDone) {
  try {
    const q = window?.sdw?.helpers?.queue;
    const id = typeof q?.randomId === 'function' ? q.randomId() : `task(${Math.random().toString(36).slice(2)})`;
    localSet(`${tab}_task_id`, id);
    showButtons(tab, false);
    const container = /** @type {HTMLElement|null} */ ($id(galleryContainerId));
    const gallery = /** @type {HTMLElement|null} */ ($id(galleryId));
    publish(types.ProgressStart, { tab, id });
    const done = () => {
      try { showButtons(tab, true); localRemove(`${tab}_task_id`); showRestore(tab, false); } finally {
        try { publish(types.ProgressEnd, { tab, id }); } catch {}
        try { onDone && onDone(); } catch {}
      }
    };
    if (typeof q?.requestProgress === 'function' && container) {
      q.requestProgress(id, container, gallery, done,
        /** @type {(res: ProgressResponse) => void} */ (makeProgressHandler(tab, id)));
    } else if (container) {
      // fallback to legacy global if present
      // @ts-ignore
      if (typeof window.requestProgress === 'function') {
        /** @type {(res: ProgressResponse) => void} */
        // @ts-ignore - legacy global
        const handler = makeProgressHandler(tab, id);
        // @ts-ignore
        window.requestProgress(id, container, gallery, done, handler);
      } else {
        done();
      }
    }
    return id;
  } catch (e) {
    try { onDone && onDone(); } catch {}
    return `task(${Date.now()})`;
  }
}

/** Resume progress for a tab if there is a saved id; return id or null. */
/** @param {string} tab */
export function resume(tab) {
  const id = localGet(`${tab}_task_id`);
  if (typeof id !== 'string' || !id) return null;
  showInterrupting(tab);
  try { publish(types.ProgressStart, { tab, id, resumed: true }); } catch {}
  try {
    const q = window?.sdw?.helpers?.queue;
    const container = /** @type {HTMLElement|null} */ ($id(`${tab}_gallery_container`));
    const gallery = /** @type {HTMLElement|null} */ ($id(`${tab}_gallery`));
    const done = () => { showButtons(tab, true); try { publish(types.ProgressEnd, { tab, id, resumed: true }); } catch {} };
    if (typeof q?.requestProgress === 'function' && container) {
      q.requestProgress(id, container, gallery, done,
        /** @type {(res: ProgressResponse) => void} */ (makeProgressHandler(tab, id, true)), 0);
    } else if (container) {
      // @ts-ignore
      if (typeof window.requestProgress === 'function') {
        /** @type {(res: ProgressResponse) => void} */
        // @ts-ignore - legacy global
        const handler = makeProgressHandler(tab, id, true);
        // @ts-ignore
        window.requestProgress(id, container, gallery, done, handler, 0);
      } else {
        done();
      }
    }
  } catch {}
  return id;
}

export const buttons = { show: showButtons, showInterrupting, showRestore };
