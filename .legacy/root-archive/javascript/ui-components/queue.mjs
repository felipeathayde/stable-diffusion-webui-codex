// Queue helpers: task id + built-in progress polling (no legacy dependency)

/** Generate a task id (parity with legacy format). */
export function randomId() {
  const rnd = () => Math.random().toString(36).slice(2, 7);
  return `task(${rnd()}${rnd()}${rnd()})`;
}

/** @typedef {{ [key: string]: unknown }} JsonLike */

/**
 * @typedef {Object} ProgressResponse
 * @property {number=} progress
 * @property {number=} eta
 * @property {string=} textinfo
 * @property {boolean=} completed
 * @property {boolean=} active
 * @property {boolean=} queued
 * @property {number=} id_live_preview
 * @property {string=} live_preview
 */

/** @param {number} n */
function pad2(n) { return n < 10 ? `0${n}` : String(n); }
/** @param {number} s */
function fmtTime(s) {
  if (s > 3600) return `${pad2(Math.floor(s/3600))}:${pad2(Math.floor(s/60)%60)}:${pad2(Math.floor(s)%60)}`;
  if (s > 60) return `${pad2(Math.floor(s/60))}:${pad2(Math.floor(s)%60)}`;
  return `${Math.floor(s)}s`;
}

let _baseTitle = document.title;
try { onUiLoaded?.(()=>{ _baseTitle = document.title; }); } catch {}
/** @param {string} progressText */
function setTitle(progressText) {
  try {
    // @ts-ignore
    const show = !!(window.opts && window.opts.show_progress_in_title);
    const next = show && progressText ? `[${String(progressText).trim()}] ${_baseTitle}` : _baseTitle;
    if (document.title !== next) document.title = next;
  } catch {}
}

/**
 * @template T
 * @param {string} url
 * @param {unknown} data
 * @param {(value: T) => void} ok
 * @param {() => void} [err]
 */
function postJSON(url, data, ok, err) {
  try {
    fetch(url, { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(data), cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((value) => { ok(/** @type {T} */ (value)); })
      .catch(() => { if (err) { try { err(); } catch {} } });
  } catch {
    if (err) { try { err(); } catch {} }
  }
}

/**
 * Polls backend progress and manages the inline UI bar. Mirrors legacy contract.
 * @param {string} idTask
 * @param {HTMLElement} progressbarContainer
 * @param {HTMLElement|null} gallery
 * @param {() => void} atEnd
 * @param {(res: ProgressResponse) => void} [onProgress]
 * @param {number} [inactivityTimeoutSeconds]
 */
export function requestProgress(idTask, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeoutSeconds = 40) {
  const started = Date.now(); let activeOnce = false;
  const parent = progressbarContainer?.parentElement; if (!parent) { try { atEnd&&atEnd(); } catch {} return; }

  // Optional Screen Wake Lock
  /** @type {any} */ let wakeLock = null;
  const wantsWake = () => { try { return !!(window.opts && window.opts.prevent_screen_sleep_during_generation); } catch { return false; } };
  const acquireWake = async()=>{ try { if (wantsWake() && !wakeLock && navigator.wakeLock) wakeLock = await navigator.wakeLock.request('screen'); } catch(e) { /* no-op */ } };
  const releaseWake = async()=>{ try { if (wakeLock) { await wakeLock.release(); wakeLock = null; } } catch(e) { /* no-op */ } };

  // DOM elements
  /** @type {HTMLDivElement | null} */
  let div = document.createElement('div'); div.className = 'progressDiv';
  try { /* @ts-ignore */ div.style.display = window.opts?.show_progressbar ? 'block' : 'none'; } catch {}
  const inner = document.createElement('div'); inner.className = 'progress'; div.appendChild(inner);
  parent.insertBefore(div, progressbarContainer);
  /** @type {HTMLElement|null} */ let liveBox = null;

  const cleanup = () => { setTitle(''); releaseWake(); try { if (div && div.parentElement) div.parentElement.removeChild(div); } catch {}
    try { if (gallery && liveBox) gallery.removeChild(liveBox); } catch {} try { atEnd&&atEnd(); } catch {} div = null; };

  const refreshPeriod = () => { try { /* @ts-ignore */ return Number(window.opts?.live_preview_refresh_period ?? 500) || 500; } catch { return 500; } };
  const inactivity = () => inactivityTimeoutSeconds;

  const tick = () => {
    acquireWake(); if (!div) return;
    postJSON('./internal/progress', { id_task: idTask, live_preview: false }, /** @param {ProgressResponse} res */ (res) => {
      if (!div) return;
      if (res && res.completed) { cleanup(); return; }
      const p = Number(res?.progress ?? 0);
      inner.style.width = `${(p*100)||0}%`; inner.style.background = p ? '' : 'transparent';
      let text = p>0 ? `${(p*100).toFixed(0)}%` : '';
      if (res?.eta) text += ` ETA: ${fmtTime(Number(res.eta)||0)}`;
      setTitle(text);
      if (res?.textinfo && !String(res.textinfo).includes('\n')) text = `${res.textinfo} ${text}`;
      inner.textContent = text;

      const elapsed = (Date.now()-started)/1000; if (res?.active) activeOnce = true;
      if ((!res?.active && activeOnce) || (elapsed > inactivity() && !res?.queued && !res?.active)) { cleanup(); return; }
      try { onProgress && onProgress(res); } catch {}
      setTimeout(tick, refreshPeriod());
    }, cleanup);
  };

  const tickLive = (liveId = 0) => {
    postJSON('./internal/progress', { id_task: idTask, id_live_preview: liveId }, /** @param {ProgressResponse} res */ (res) => {
      if (!div) return;
      if (res?.live_preview && gallery) {
        try {
          const img = new Image();
          img.onload = () => {
            if (!liveBox) { liveBox = document.createElement('div'); liveBox.className = 'livePreview'; gallery.insertBefore(liveBox, gallery.firstElementChild); }
            liveBox.appendChild(img); while (liveBox.childElementCount > 2) { const first = liveBox.firstElementChild; if (first) liveBox.removeChild(first); }
          };
          img.src = res.live_preview;
        } catch {}
      }
      const next = typeof res?.id_live_preview === 'number' ? res.id_live_preview : Number(liveId) || 0;
      setTimeout(()=>tickLive(next), refreshPeriod());
    }, cleanup);
  };

  tick(); if (gallery) tickLive();
}
