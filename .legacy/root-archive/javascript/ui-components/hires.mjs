// Hires configuration helper (parity with legacy Codex component)
import { readCheckbox, readInt, readFloat, readDropdownValue, readText } from './readers.mjs';

/**
 * Return a normalized hires configuration for the requested tab.
 * Currently only txt2img exposes hires settings; other tabs return { enable:false }.
 * @param {'txt2img'|'img2img'|'txt2vid'|'img2vid'|string} tab
 * @returns {Record<string, unknown>}
 */
export function get(tab) {
  if (tab !== 'txt2img' && tab !== '' && tab != null) {
    return { enable: false };
  }

  const enable = readCheckbox('txt2img_hr_enable');
  if (!enable) return { enable: false };

  const hrUpscaler = readDropdownValue('txt2img_hr_upscaler');
  const hrCheckpoint = readDropdownValue('hr_checkpoint');
  const hrVaeTeRaw = readDropdownValue('hr_vae_te');
  const hrSampler = readDropdownValue('hr_sampler');
  const hrScheduler = readDropdownValue('hr_scheduler');

  const hires = {
    enable: true,
    steps: readInt('txt2img_hires_steps'),
    denoise: readFloat('txt2img_denoising_strength'),
    scale: readFloat('txt2img_hr_scale'),
    upscaler: Array.isArray(hrUpscaler) ? hrUpscaler[0] ?? '' : hrUpscaler,
    resize_x: readInt('txt2img_hr_resize_x'),
    resize_y: readInt('txt2img_hr_resize_y'),
    hr_checkpoint: Array.isArray(hrCheckpoint) ? hrCheckpoint[0] ?? '' : hrCheckpoint,
    hr_vae_te: Array.isArray(hrVaeTeRaw) ? hrVaeTeRaw : (hrVaeTeRaw ? [hrVaeTeRaw] : []),
    hr_sampler: Array.isArray(hrSampler) ? hrSampler[0] ?? '' : hrSampler,
    hr_scheduler: Array.isArray(hrScheduler) ? hrScheduler[0] ?? '' : hrScheduler,
    hr_prompt: readText('hires_prompt'),
    hr_negative_prompt: readText('hires_neg_prompt'),
    hr_cfg: readFloat('txt2img_hr_cfg'),
    hr_distilled_cfg: readFloat('txt2img_hr_distilled_cfg'),
  };

  return hires;
}
