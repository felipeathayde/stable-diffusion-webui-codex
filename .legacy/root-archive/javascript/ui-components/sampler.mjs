// Sampler helper: update sampler/scheduler choices based on backend data
import { getAppElementById as $id } from './readers.mjs';

/**
 * Replace the options for sampler/scheduler dropdowns.
 * @param {[string, string]} ids [samplerElemId, schedulerElemId]
 * @param {{ samplers?: string[]; schedulers?: string[] }} choices
 */
export function setChoices(ids, choices) {
  const [samplerId, schedulerId] = ids;
  const samplerRoot = $id(samplerId);
  const schedulerRoot = $id(schedulerId);

  /** @param {HTMLElement | null} root @param {string[] | undefined} items */
  const refreshSelect = (root, items) => {
    if (!(root instanceof HTMLElement) || !Array.isArray(items)) return;
    const select = root instanceof HTMLSelectElement ? root : root.querySelector('select');
    if (!(select instanceof HTMLSelectElement)) return;
    const current = select.value;
    select.innerHTML = '';
    items.forEach((label) => {
      const opt = document.createElement('option');
      opt.value = label;
      opt.textContent = label;
      select.appendChild(opt);
    });
    if (items.includes(current)) {
      select.value = current;
    }
    select.dispatchEvent(new Event('change', { bubbles: true }));
  };

  refreshSelect(samplerRoot, choices.samplers);
  refreshSelect(schedulerRoot, choices.schedulers);
}
