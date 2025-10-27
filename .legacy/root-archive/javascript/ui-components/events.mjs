// Tiny event bus for UI helpers and extensions (no dependencies)

/** @typedef {CustomEvent & { detail: any }} SdwEvent */

const bus = new EventTarget();

/**
 * Subscribe to an event. Returns an unsubscribe function.
 * @param {string} type
 * @param {(ev: SdwEvent)=>void} handler
 * @returns {() => void}
 */
export function on(type, handler) {
  const fn = /** @param {Event} e */ (e) => {
    try { handler(/** @type {any} */(e)); } catch (err) { console.error('[SDW:events] handler error', err); }
  };
  bus.addEventListener(type, fn);
  return () => bus.removeEventListener(type, fn);
}

/**
 * Publish an event with an optional detail payload.
 * @param {string} type
 * @param {any} [detail]
 */
export function emit(type, detail) {
  try {
    const ev = new CustomEvent(type, { detail });
    bus.dispatchEvent(ev);
  } catch (e) {
    // Fallback for environments without CustomEvent constructor
    try { bus.dispatchEvent(new Event(type)); } catch {}
  }
}

/**
 * Utility to emit a typed event with a consistent shape.
 * @param {string} type
 * @param {Record<string, any>} fields
 */
export function publish(type, fields = {}) {
  emit(type, Object.assign({ ts: Date.now() }, fields));
}

export const types = Object.freeze({
  ProgressStart: 'progress:start',
  ProgressUpdate: 'progress:update',
  ProgressEnd: 'progress:end',
  TabsChange: 'tabs:change',
  GallerySelect: 'gallery:select',
  OptionsUpdate: 'options:update',
});

