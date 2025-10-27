/**
 * SD WebUI — Modular UI runtime (ESM, no build required).
 * Entry file loaded as <script type="module"> by the injector.
 *
 * Notes:
 * - Keep source as JavaScript with JSDoc types; TypeScript checks run via tsconfig (checkJs:true).
 * - Only this file lives in javascript/ root; all other modules stay under subfolders to avoid double-injection.
 */

import { runOnUiLoaded } from './ui-components/hooks.mjs';
import { ensureRoot, mountAt, updateInput as domUpdateInput } from './ui-components/dom.mjs';
import { installDropdownPortal } from './ui-components/overlay.mjs';
import { bootProgressBridge } from './ui-components/progress.mjs';
import * as galleries from './ui-components/galleries.mjs';
import * as submit from './ui-components/submit.mjs';
import * as lightbox from './ui-components/lightbox.mjs';
import * as contextMenu from './ui-components/context-menu.mjs';
import * as hotkeys from './ui-components/hotkeys.mjs';
import * as lightboxGamepad from './ui-components/lightbox-gamepad.mjs';
import * as galleryKeys from './ui-components/gallery-keys.mjs';
import * as tabs from './ui-components/tabs.mjs';
import * as readers from './ui-components/readers.mjs';
import * as queue from './ui-components/queue.mjs';
import * as flow from './ui-components/flow.mjs';
import * as clipboard from './ui-components/clipboard.mjs';
import * as strict from './ui-components/strict.mjs';
import * as options from './ui-components/options.mjs';
import * as events from './ui-components/events.mjs';
import * as canvas from './ui-components/canvas.mjs';
import * as hires from './ui-components/hires.mjs';
import * as prompt from './ui-components/prompt.mjs';
import * as sampler from './ui-components/sampler.mjs';
import * as dropdowns from './ui-components/dropdowns.mjs';
import * as nativeDropdown from './ui-components/native-dropdown.mjs';
import * as htmlDropdown from './ui-components/html-dropdown.mjs';

/** @typedef {{ mount: (root: HTMLElement)=>void, name: string }} SdwComponent */

/** @type {{ components: Record<string, SdwComponent>, register:(c:SdwComponent)=>void }} */
const registry = {
  components: {},
  register(c) {
    if (!c || !c.name || typeof c.mount !== 'function') return;
    this.components[c.name] = c;
  }
};

function bootstrap() {
  const root = ensureRoot('sdw-ui-root');
  // Install dropdown/lightbox overlay helpers (CSS handles z-index; JS handles positioning if needed)
  installDropdownPortal();
  // Bridge progress updates to an optional overlay component API (no-op if not present)
  bootProgressBridge();
  // Install optional behaviour modules idempotently
  try { lightbox.install(); } catch (e) { console.warn('[SDW:UI] lightbox install failed', e); }
  try { contextMenu.install(); } catch (e) { console.warn('[SDW:UI] context-menu install failed', e); }
  try { hotkeys.install(); } catch (e) { console.warn('[SDW:UI] hotkeys install failed', e); }
  try { lightboxGamepad.install(); } catch (e) { console.warn('[SDW:UI] lightbox-gamepad install failed', e); }
  try { galleryKeys.install(); } catch (e) { console.warn('[SDW:UI] gallery-keys install failed', e); }
  try { galleries.install(); } catch (e) { console.warn('[SDW:UI] galleries install failed', e); }
  try { clipboard.install(); } catch (e) { console.warn('[SDW:UI] clipboard install failed', e); }
  try { options.install(); } catch (e) { console.warn('[SDW:UI] options install failed', e); }
  try { dropdowns.install(); } catch (e) { console.warn('[SDW:UI] dropdowns install failed', e); }
  try { nativeDropdown.install(); } catch (e) { console.warn('[SDW:UI] native-dropdown install failed', e); }
  try { htmlDropdown.install(); } catch (e) { console.warn('[SDW:UI] html-dropdown install failed', e); }
  // Mount any registered components
  Object.values(registry.components).forEach((c) => {
    try { c.mount(mountAt(root, c.name)); } catch (e) { console.warn('[SDW:UI] component failed', c?.name, e); }
  });
}

// Defer to Gradio’s onUiLoaded hook if present; fallback to DOMContentLoaded
runOnUiLoaded(bootstrap);

// Expose a minimal API for extensions
// @ts-ignore
window.sdw = Object.assign(window.sdw || {}, {
  ui: { registry },
  helpers: {
    dom: { updateInput: domUpdateInput },
    galleries,
    submit,
    tabs,
    readers,
    queue,
    flow,
    strict,
    options,
    events,
    canvas,
    hires,
    prompt,
    sampler,
    lightbox: { install: lightbox.install, afterUpdate: lightbox.afterUpdate, switchRelative: lightbox.switchRelative },
    lightboxGamepad: { install: lightboxGamepad.install },
    contextMenu: { install: contextMenu.install, append: contextMenu.append, remove: contextMenu.remove },
    hotkeys: { install: hotkeys.install },
    galleryKeys: { install: galleryKeys.install },
    dropdown: { getValue: htmlDropdown.getValue, setValue: htmlDropdown.setValue, update: htmlDropdown.update, on: htmlDropdown.on, list: htmlDropdown.list },
  },
});

// Provide legacy globals for broad compatibility with existing scripts
try {
  // @ts-ignore
  if (typeof window.requestProgress !== 'function') window.requestProgress = queue.requestProgress;
  // @ts-ignore
  if (typeof window.randomId !== 'function') window.randomId = queue.randomId;
  // @ts-ignore
  if (typeof window.modalImageSwitch !== 'function') window.modalImageSwitch = (d) => { try { lightbox.switchRelative(d|0); } catch {} };
} catch {}

export { registry };

// Bridge to legacy Codex namespace so existing ui.js builders can consume ESM readers without code churn
try {
  // @ts-ignore
  const NS = (window.Codex = window.Codex || {});
  // @ts-ignore
  const C = (NS.Components = NS.Components || {});
  // Prefer the ESM helpers as the canonical implementation
  // @ts-ignore
  C.Readers = readers;
  // @ts-ignore
  C.Canvas = canvas;
  // @ts-ignore
  C.Hires = hires;
  // @ts-ignore
  C.Prompt = prompt;
  // @ts-ignore
  C.Sampler = sampler;
  // Lightweight bus for legacy consumers
  // @ts-ignore
  NS.bus = NS.bus || {
    /** @param {string} event @param {(payload?: unknown) => void} fn */
    on(event, fn) {
      const unsubscribe = events.on(event, (ev) => {
        try { fn?.(ev?.detail); } catch (error) { console.warn('[SDW:UI] Codex bus handler failed', error); }
      });
      return unsubscribe;
    },
    /** @param {string} event @param {unknown} payload */
    emit(event, payload) {
      events.emit(event, payload);
    }
  };
} catch (e) {
  console.warn('[SDW:UI] Failed to expose ESM readers under window.Codex.Components.Readers', e);
}
