// Lightbox modal helpers extracted from imageviewer.js (JS + JSDoc, no build)
import { allThumbButtons } from './galleries.mjs';

/** @returns {Document | ShadowRoot | HTMLElement} */
function grRoot() { try { return gradioApp(); } catch { return document; } }

/** @param {string} id */
function byId(id) {
  const root = grRoot();
  if ('getElementById' in root && typeof root.getElementById === 'function') {
    const el = root.getElementById(id);
    if (el instanceof HTMLElement) return el;
  }
  const fb = document.getElementById(id);
  return fb instanceof HTMLElement ? fb : null;
}

/** @returns {HTMLImageElement | null} */
function modalImg() { const el = byId('modalImage'); return el instanceof HTMLImageElement ? el : null; }

export function close() { const m = byId('lightboxModal'); if (m) m.style.display = 'none'; }

/** @param {MouseEvent} event */
function show(event) {
  const target = event.currentTarget instanceof HTMLImageElement
    ? event.currentTarget
    : (event.target instanceof HTMLImageElement ? event.target : null);
  if (!target) return;
  const img = modalImg(); const modal = byId('lightboxModal'); const toggle = byId('modal_toggle_live_preview');
  if (!img || !modal) return;
  try { if (toggle) { /* @ts-ignore */ toggle.innerHTML = (opts?.js_live_preview_in_modal_lightbox ? '&#x1F5C7;' : '&#x1F5C6;'); } } catch {}
  img.src = target.src;
  if (img.style.display === 'none') modal.style.setProperty('background-image', `url(${target.src})`);
  modal.style.display = 'flex';
  modal.focus();
  event.stopPropagation();
}

/** @param {number} n @param {number} m */
function negmod(n, m) { return ((n % m) + m) % m; }

function updateBg() {
  const img = modalImg(); if (!img || !img.offsetParent) return;
  const currentButton = (typeof selected_gallery_button === 'function') ? selected_gallery_button() : null;
  const previews = grRoot().querySelectorAll('.livePreview > img');
  try {
    // @ts-ignore
    if (opts?.js_live_preview_in_modal_lightbox && previews.length) {
      const last = previews[previews.length - 1]; if (last instanceof HTMLImageElement) img.src = last.src; return;
    }
  } catch {}
  if (currentButton && currentButton.children?.length) {
    const child = currentButton.children[0];
    if (child instanceof HTMLImageElement && img.src !== child.src) {
      img.src = child.src;
      if (img.style.display === 'none') { const m = byId('lightboxModal'); m?.style.setProperty('background-image', `url(${img.src})`); }
    }
  }
}

/** @param {number} offset */
export function switchRelative(offset) {
  /** @param {Element} btn @returns {btn is HTMLElement} */
  const isButton = (btn) => btn instanceof HTMLElement;
  const buttons = Array.from(allThumbButtons()).filter(isButton);
  if (!buttons.length) return;
  // find selected
  let selected = 0;
  for (let i = 0; i < buttons.length; i += 1) {
    const candidate = buttons[i];
    if (candidate && candidate.classList.contains('selected')) { selected = i; break; }
  }
  const next = negmod(selected + offset, buttons.length);
  const button = buttons[next];
  if (button && typeof button.click === 'function') button.click();
  updateBg();
}

/** @param {KeyboardEvent} e */
function keyHandler(e) {
  switch (e.key) {
    case 'ArrowLeft': switchRelative(-1); e.stopPropagation(); break;
    case 'ArrowRight': switchRelative(1); e.stopPropagation(); break;
    case 'Escape': close(); e.stopPropagation(); break;
    case 's': try { /* @ts-ignore */ saveImage(); } catch {} e.stopPropagation(); break;
  }
}

/** @param {HTMLImageElement | null} img @param {boolean} on */
function zoomSet(img, on) { if (img) img.classList.toggle('modalImageFullscreen', !!on); }
/** @param {MouseEvent} e */
function zoomToggle(e) { const img = modalImg(); zoomSet(img, img ? !img.classList.contains('modalImageFullscreen') : false); e.stopPropagation(); }
/** @param {MouseEvent} e */
function livePreviewToggle(e) { try { /* @ts-ignore */ opts.js_live_preview_in_modal_lightbox = !opts.js_live_preview_in_modal_lightbox; } catch {} const t = byId('modal_toggle_live_preview'); if (t) t.innerHTML = '&#x1F5C6;'; e.stopPropagation(); }
/** @param {MouseEvent} e */
function tileToggle(e) {
  const img = modalImg(); const modal = byId('lightboxModal'); if (!img || !modal) return;
  const tiling = img.style.display === 'none';
  if (tiling) { img.style.display = 'block'; modal.style.setProperty('background-image', 'none'); }
  else { img.style.display = 'none'; modal.style.setProperty('background-image', `url(${img.src})`); }
  e.stopPropagation();
}

function build() {
  if (byId('lightboxModal')) return; // idempotent
  const modal = document.createElement('div'); modal.id = 'lightboxModal'; modal.tabIndex = 0;
  modal.addEventListener('click', () => close()); modal.addEventListener('keydown', keyHandler, true);
  const controls = document.createElement('div'); controls.className = 'modalControls gradio-container'; modal.append(controls);
  /** @param {string} cls @param {string} html @param {string} title @param {(e: MouseEvent) => void} cb */
  const mkBtn = (cls, html, title, cb) => { const s = document.createElement('span'); s.className = `${cls} cursor`; s.innerHTML = html; s.title = title; s.addEventListener('click', cb, true); controls.appendChild(s); };
  mkBtn('modalZoom', '&#10529;', 'Toggle zoomed view', zoomToggle);
  mkBtn('modalTileImage', '&#8862;', 'Preview tiling', tileToggle);
  const save = document.createElement('span'); save.className = 'modalSave cursor'; save.id = 'modal_save'; save.innerHTML = '&#x1F5AB;'; save.title = 'Save Image(s)'; save.addEventListener('click', (e)=>{ try { /* @ts-ignore */ modalSaveImage(e); } catch {} }, true); controls.appendChild(save);
  const live = document.createElement('span'); live.className = 'modalToggleLivePreview cursor'; live.id = 'modal_toggle_live_preview'; live.innerHTML = '&#x1F5C6;'; live.title = 'Toggle live preview'; live.addEventListener('click', livePreviewToggle, true); controls.appendChild(live);
  const closeBtn = document.createElement('span'); closeBtn.className = 'modalClose cursor'; closeBtn.innerHTML = '&times;'; closeBtn.title = 'Close image viewer'; closeBtn.addEventListener('click', ()=>close(), true); controls.appendChild(closeBtn);
  const img = document.createElement('img'); img.id = 'modalImage'; img.tabIndex = 0; img.addEventListener('click', ()=>close()); img.addEventListener('keydown', keyHandler, true); modal.appendChild(img);
  /** @param {string} cls @param {string} html @param {(e: MouseEvent)=>void} cb */
  const mkNav = (cls, html, cb) => { const a = document.createElement('a'); a.className = cls; a.innerHTML = html; a.tabIndex = 0; a.addEventListener('click', cb, true); a.addEventListener('keydown', keyHandler, true); modal.appendChild(a); };
  mkNav('modalPrev', '&#10094;', ()=>switchRelative(-1));
  mkNav('modalNext', '&#10095;', ()=>switchRelative(1));
  try { grRoot().appendChild(modal); } catch { document.body.appendChild(modal); }
}

export function afterUpdate() {
  const previews = grRoot().querySelectorAll('.gradio-gallery > button > button > img, .gradio-gallery > .livePreview');
  previews.forEach((node) => { if (node instanceof HTMLImageElement) setupImage(node); });
  updateBg();
}

/** @param {HTMLImageElement} img */
function setupImage(img) {
  if (img.dataset.modded) return; img.dataset.modded = 'true'; img.style.cursor = 'pointer'; img.style.userSelect = 'none';
  img.addEventListener('mousedown', (evt) => { if (evt.button === 1 && typeof window.open === 'function' && img.src) { window.open(img.src); evt.preventDefault(); } }, true);
  img.addEventListener('click', (evt) => { try { /* @ts-ignore */ if (!opts?.js_modal_lightbox || evt.button !== 0) return; } catch {} zoomSet(modalImg(), !!(/* @ts-ignore */ opts?.js_modal_lightbox_initially_zoomed)); evt.preventDefault(); show(evt); }, true);
}

export function install() {
  build();
  document.addEventListener('DOMContentLoaded', build);
  try { onAfterUiUpdate?.(afterUpdate); } catch {}
}
