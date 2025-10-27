import { runOnUiLoaded } from './hooks.mjs';

let installed = false;

/** @param {HTMLElement} dropdownIcon @param {MouseEvent|PointerEvent} originalEvent */
function triggerInput(dropdownIcon, originalEvent) {
  const dropdown = dropdownIcon.closest('.gradio-dropdown');
  if (!dropdown) return;
  const input = dropdown.querySelector('input[role="combobox"], input[role="listbox"]');
  if (!(input instanceof HTMLInputElement)) return;

  try {
    input.focus({ preventScroll: true });
  } catch {
    input.focus();
  }

  /** @type {PointerEventInit} */
  const pointerEventInit = {
    bubbles: true,
    pointerId: 1,
    pointerType: 'mouse',
  };

  if (originalEvent instanceof PointerEvent) {
    pointerEventInit.pointerType = originalEvent.pointerType || 'mouse';
    pointerEventInit.pointerId = originalEvent.pointerId || 1;
  }

  try {
    input.dispatchEvent(new PointerEvent('pointerdown', pointerEventInit));
    input.dispatchEvent(new PointerEvent('pointerup', pointerEventInit));
  } catch {
    // PointerEvent may not be supported; fall back to mouse events.
    input.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    input.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
  }

  input.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

/** @param {PointerEvent} event */
function handlePointerDown(event) {
  const target = event.target instanceof Element ? event.target : null;
  const icon = target ? target.closest('.gradio-dropdown .icon-wrap') : null;
  if (!(icon instanceof HTMLElement)) return;

  event.preventDefault();
  event.stopPropagation();

  triggerInput(icon, event);
}

export function install() {
  if (installed) return;
  installed = true;

  runOnUiLoaded(() => {
    document.addEventListener('pointerdown', handlePointerDown, true);
  });
}
