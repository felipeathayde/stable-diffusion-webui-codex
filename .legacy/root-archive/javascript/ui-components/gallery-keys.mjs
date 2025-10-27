// Keyboard navigation for galleries (left/right). JS + JSDoc; no build.
import { containerButtons } from './galleries.mjs';

/** @returns {Document | ShadowRoot | HTMLElement} */
function grRoot() { try { return gradioApp(); } catch { return document; } }

/** @param {string} containerId @param {KeyboardEvent} ev */
function handleKey(containerId, ev) {
  if (!(ev instanceof KeyboardEvent)) return;
  if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
  /** @param {Element} el @returns {el is HTMLElement} */
  const isElement = (el) => el instanceof HTMLElement;
  const list = Array.from(containerButtons(containerId)).filter(isElement);
  if (!list.length) return;
  let idx = list.findIndex((el)=> el.classList?.contains('selected'));
  if (idx < 0) idx = 0;
  idx = ev.key === 'ArrowLeft' ? (idx - 1 + list.length) % list.length : (idx + 1) % list.length;
  const next = list[idx]; if (next && typeof next.click === 'function') next.click();
  ev.preventDefault(); ev.stopPropagation();
}

/** @param {string} tab */
function installOne(tab) {
  const id = `${tab}_gallery`;
  const g = grRoot().querySelector(`#${id}`);
  if (!(g instanceof HTMLElement) || g.dataset.sdwGalleryKeys==='1') return;
  g.dataset.sdwGalleryKeys='1';
  g.addEventListener('keydown', (ev)=> handleKey(id, ev), true);
}

export function install() {
  ['txt2img','img2img','txt2vid','img2vid'].forEach(installOne);
  try { onAfterUiUpdate?.(()=>['txt2img','img2img','txt2vid','img2vid'].forEach(installOne)); } catch {}
}
