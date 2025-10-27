// Canvas helpers: surface lookup and inpaint normalization
import { getAppElementById as $id, readInt, readFloat, readRadioIndex } from './readers.mjs';

/**
 * Find the primary rendered surface (img or canvas) for a component id.
 * @param {string} id
 * @returns {HTMLImageElement | HTMLCanvasElement | null}
 */
export function getSurface(id) {
  const root = $id(id);
  if (!root) return null;
  const img = root.querySelector('img');
  if (img instanceof HTMLImageElement) return img;
  const canvas = root.querySelector('canvas');
  if (canvas instanceof HTMLCanvasElement) return canvas;
  return null;
}

/**
 * Collect normalized inpainting settings for img2img when an inpaint tab is active.
 * Matches the legacy Codex Canvas API to keep behaviours consistent.
 * @returns {{
 *   mask_blur: number;
 *   mask_alpha: number;
 *   inpainting_fill: number;
 *   inpaint_full_res: boolean;
 *   inpaint_full_res_padding: number;
 *   inpainting_mask_invert: number;
 * }}
 */
export function getInpaint() {
  const maskBlur = readInt('img2img_mask_blur');
  const maskAlpha = readFloat('img2img_mask_alpha');
  const inpaintFill = readRadioIndex('img2img_inpainting_fill');
  const fullResIdx = readRadioIndex('img2img_inpaint_full_res');
  const fullResPadding = readInt('img2img_inpaint_full_res_padding');
  const maskInvert = readRadioIndex('img2img_mask_mode');
  return {
    mask_blur: maskBlur,
    mask_alpha: maskAlpha,
    inpainting_fill: inpaintFill,
    inpaint_full_res: fullResIdx === 1,
    inpaint_full_res_padding: fullResPadding,
    inpainting_mask_invert: maskInvert,
  };
}

