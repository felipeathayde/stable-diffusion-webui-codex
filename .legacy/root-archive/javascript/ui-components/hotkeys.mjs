// Generate/Skip/Interrupt hotkeys (JS + JSDoc), parity with codex.components.prompt.js

import { getAppElementById as $id } from './readers.mjs';

/** @returns {Document | ShadowRoot | HTMLElement} */
function grRoot() { try { return gradioApp(); } catch { return document; } }

/** @param {string} id */
function byId(id){ return $id(id) || document.getElementById(id); }

/** @param {HTMLElement} el @param {string} tab */
function bind(el, tab){ if (!el || el.dataset.sdwHotkey==='1') return; el.dataset.sdwHotkey='1'; el.addEventListener('keydown', (ev)=>{
  try {
    if (ev.key==='Enter' && (ev.ctrlKey||ev.metaKey)) {
      const btn = grRoot().querySelector(`#${tab}_generate button, #${tab}_generate`) || document.getElementById(`${tab}_generate`);
      if (btn instanceof HTMLElement && typeof btn.click === 'function') { ev.preventDefault(); btn.click(); return; }
    }
    if (ev.key==='Enter' && ev.altKey) {
      const btn = grRoot().querySelector(`#${tab}_skip`) || document.getElementById(`${tab}_skip`);
      if (btn instanceof HTMLElement && typeof btn.click === 'function') { ev.preventDefault(); btn.click(); return; }
    }
    if (ev.key==='Escape') {
      const btn = grRoot().querySelector(`#${tab}_interrupt`) || document.getElementById(`${tab}_interrupt`);
      if (btn instanceof HTMLElement && typeof btn.click === 'function') { ev.preventDefault(); btn.click(); return; }
    }
  } catch (e) { console.warn('[SDW:UI] hotkeys error', e); }
}, { capture:true }); }

function scan(){ ['txt2img','img2img','txt2vid','img2vid'].forEach((tab)=>{ const pos = byId(`${tab}_prompt`); const neg = byId(`${tab}_neg_prompt`); const p = pos && pos.querySelector('textarea'); const n = neg && neg.querySelector('textarea'); if (p instanceof HTMLTextAreaElement) bind(p, tab); if (n instanceof HTMLTextAreaElement) bind(n, tab); }); }

export function install(){ try{ onUiLoaded?.(scan); } catch { document.addEventListener('DOMContentLoaded', scan); } }
