// Context menu helpers extracted from contextMenus.js (JS + JSDoc)

/** @returns {Document | ShadowRoot | HTMLElement} */
function grRoot() { try { return gradioApp(); } catch { return document; } }

function dismiss() { const existing = grRoot().querySelector('#context-menu'); if (existing instanceof HTMLElement) existing.remove(); }

/** @param {{ pageX: number; pageY: number }} position @param {{name:string,fn:()=>void}[]} items */
function render(position, items) {
  dismiss();
  const base = (typeof uiCurrentTab !== 'undefined' && uiCurrentTab instanceof HTMLElement) ? uiCurrentTab : document.body;
  const style = window.getComputedStyle(base);
  const nav = document.createElement('nav'); nav.id = 'context-menu';
  nav.style.background = style.background; nav.style.color = style.color; nav.style.fontFamily = style.fontFamily;
  nav.style.top = `${position.pageY}px`; nav.style.left = `${position.pageX}px`;
  const ul = document.createElement('ul'); ul.className = 'context-menu-items'; nav.append(ul);
  items.forEach((it)=>{ const a = document.createElement('a'); a.textContent = it.name; a.addEventListener('click', (e)=>{ e.preventDefault(); e.stopPropagation(); try { it.fn(); } finally { dismiss(); } }); ul.append(a); });
  grRoot().appendChild(nav);
}

/** @type {Map<string, {name:string, fn:()=>void, id:string}[]>} */
const specs = new Map(); let applied = false;

/** @param {string} selector @param {string} name @param {()=>void} fn */
export function append(selector, name, fn) {
  const list = specs.get(selector) || []; const id = `${selector}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  list.push({ name, fn, id }); specs.set(selector, list); return id;
}

/** @param {string} id */
export function remove(id) { specs.forEach((arr)=>{ const i = arr.findIndex((e)=>e.id===id); if (i>=0) arr.splice(i,1); }); }

function addListeners() {
  if (applied) return; applied = true;
  grRoot().addEventListener('click', (e)=>{ if (e.isTrusted) dismiss(); });
  /** @param {Event} event */
  const handle = (event) => {
    if (event instanceof TouchEvent) {
      const touch = event.touches.item(0);
      if (!touch) return;
      process(event, { pageX: touch.pageX, pageY: touch.pageY });
    } else if (event instanceof MouseEvent) {
      process(event, { pageX: event.pageX, pageY: event.pageY });
    }
  };
  /** @param {Event} event @param {{pageX:number,pageY:number}} position */
  function process(event, position) {
    dismiss(); const path = event.composedPath(); const target = path.length>0 ? path[0] : null;
    specs.forEach((entries, selector)=>{ if (target instanceof Element && target.matches(selector)) { render(position, entries); event.preventDefault(); } });
  }
  grRoot().addEventListener('contextmenu', handle);
  grRoot().addEventListener('touchstart', handle);
}

function installDefaults() {
  // Generate forever entries for txt2img/img2img (parity with legacy)
  /** @type {ReturnType<typeof setInterval> | null} */ let regen_txt2img = null;
  /** @type {ReturnType<typeof setInterval> | null} */ let regen_img2img = null;
  /** @param {string} genId @param {string} intId @param {'txt'|'img'} guard */
  const genForever = (genId, intId, guard) => () => {
    if (regen_txt2img !== null || regen_img2img !== null) return;
    const gen = grRoot().querySelector(genId); const intr = grRoot().querySelector(intId);
    if (!(gen instanceof HTMLElement) || !(intr instanceof HTMLElement)) return;
    if (!intr.offsetParent) gen.click();
    const timer = setInterval(()=>{ if (intr.style.display === 'none') { gen.click(); intr.style.display = 'block'; } }, 500);
    if (guard==='txt') regen_txt2img = timer; else regen_img2img = timer;
  };
  /** @param {'txt'|'img'} guard */
  const cancel = (guard) => () => { const h = (guard==='txt') ? 'regen_txt2img' : 'regen_img2img'; if (guard==='txt' && regen_txt2img) { clearInterval(regen_txt2img); regen_txt2img=null; } if (guard==='img' && regen_img2img) { clearInterval(regen_img2img); regen_img2img=null; } };
  append('#txt2img_generate','Generate forever',genForever('#txt2img_generate','#txt2img_interrupt','txt'));
  append('#txt2img_interrupt','Generate forever',genForever('#txt2img_generate','#txt2img_interrupt','txt'));
  append('#txt2img_interrupt','Cancel generate forever',cancel('txt'));
  append('#txt2img_generate','Cancel generate forever',cancel('txt'));
  append('#img2img_generate','Generate forever',genForever('#img2img_generate','#img2img_interrupt','img'));
  append('#img2img_interrupt','Generate forever',genForever('#img2img_generate','#img2img_interrupt','img'));
  append('#img2img_interrupt','Cancel generate forever',cancel('img'));
  append('#img2img_generate','Cancel generate forever',cancel('img'));

  // Viewer lightbox context menu (modal image)
  const openInNewTab = () => { const img = document.getElementById('modalImage'); if (img instanceof HTMLImageElement && img.src) window.open(img.src, '_blank'); };
  const toggleZoom = () => { const img = document.getElementById('modalImage'); if (img instanceof HTMLImageElement) img.classList.toggle('modalImageFullscreen'); };
  const closeViewer = () => { const modal = document.getElementById('lightboxModal'); if (modal) modal.style.display = 'none'; };
  append('#lightboxModal #modalImage', 'Open image in new tab', openInNewTab);
  append('#lightboxModal #modalImage', 'Toggle zoom', toggleZoom);
  append('#lightboxModal #modalImage', 'Close viewer', closeViewer);
}

export function install() {
  addListeners();
  installDefaults();
  try { onAfterUiUpdate?.(addListeners); } catch {}
}
