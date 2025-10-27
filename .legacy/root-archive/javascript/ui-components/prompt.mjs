// Prompt helpers: read/write prompt textboxes (parity with legacy Codex component)
import { getAppElementById as $id, readText } from './readers.mjs';
import { updateInput } from './dom.mjs';

/**
 * Read positive/negative prompt text for a tab.
 * @param {'txt2img'|'img2img'|'txt2vid'|'img2vid'|string} tab
 * @returns {{ prompt: string; negative: string }}
 */
export function get(tab) {
  return {
    prompt: readText(`${tab}_prompt`),
    negative: readText(`${tab}_neg_prompt`),
  };
}

/**
 * Write prompt values back to the UI (dispatching input events).
 * @param {'txt2img'|'img2img'|'txt2vid'|'img2vid'|string} tab
 * @param {{ prompt?: string; negative?: string }} values
 * @returns {boolean} true if at least one field updated
 */
export function set(tab, values) {
  let changed = false;
  /** @param {string} id @param {string | undefined} text */
  const apply = (id, text) => {
    const root = $id(id);
    if (!root) return false;
    const textarea = root.querySelector('textarea');
    if (textarea instanceof HTMLTextAreaElement) {
      textarea.value = String(text ?? '');
      updateInput(textarea);
      return true;
    }
    return false;
  };

  if ('prompt' in values) {
    changed = apply(`${tab}_prompt`, values.prompt) || changed;
  }
  if ('negative' in values) {
    changed = apply(`${tab}_neg_prompt`, values.negative) || changed;
  }
  return changed;
}
