// Strict JSON helpers: stringify and sync hidden field for server submit.
import { getAppElementById as $id } from './readers.mjs';
import { updateInput } from './dom.mjs';
import { getInpaint as getCanvasInpaint } from './canvas.mjs';

/** @typedef {{ [key: string]: unknown }} StrictPayload */

/**
 * Stringify a strict payload and sync it into the hidden JSON slot for a tab.
 * Returns the JSON string or null if serialization failed.
 * @param {string} tab
 * @param {StrictPayload} payload
 */
export function stringifyAndSync(tab, payload) {
  let json = null;
  try {
    json = JSON.stringify(payload);
  } catch (err) {
    console.warn('[Strict] JSON.stringify failed', err, payload);
    json = null;
  }
  try {
    if (json !== null) {
      const hiddenRoot = $id(`${tab}_named_active`);
      if (hiddenRoot) {
        const field = hiddenRoot.querySelector('textarea, input');
        if (field instanceof HTMLTextAreaElement || field instanceof HTMLInputElement) {
          field.value = json;
          updateInput(field);
        }
      }
    }
  } catch (e) {
    console.warn('[Strict] failed to sync hidden strict textbox', e);
  }
  return json;
}

// ---- Strict JSON builders (ESM) ----
import * as R from './readers.mjs';
import * as T from './tabs.mjs';

function _buildTxt2img() {
  const named = /** @type {StrictPayload} */ ({ __strict_version: 1, __source: 'txt2img' });
  /** @type {string[]} */
  const active = [];
  const rText = R.readText, rDD = R.readDropdownValue, rDDR = R.readDropdownOrRadioValue, rInt = R.readInt, rFloat = R.readFloat, rSeed = R.readSeedValue;
  named['txt2img_prompt'] = rText('txt2img_prompt'); active.push('txt2img_prompt');
  named['txt2img_neg_prompt'] = rText('txt2img_neg_prompt'); active.push('txt2img_neg_prompt');
  named['txt2img_styles'] = rDD('txt2img_styles') || []; active.push('txt2img_styles');
  named['txt2img_batch_count'] = rInt('txt2img_batch_count'); active.push('txt2img_batch_count');
  named['txt2img_batch_size'] = rInt('txt2img_batch_size'); active.push('txt2img_batch_size');
  named['txt2img_cfg_scale'] = rFloat('txt2img_cfg_scale'); active.push('txt2img_cfg_scale');
  if (R.isInteractive('txt2img_distilled_cfg_scale')) {
    named['txt2img_distilled_cfg_scale'] = rFloat('txt2img_distilled_cfg_scale'); active.push('txt2img_distilled_cfg_scale');
  }
  named['txt2img_height'] = rInt('txt2img_height'); active.push('txt2img_height');
  named['txt2img_width'] = rInt('txt2img_width'); active.push('txt2img_width');
  const rCB = R.readCheckbox;
  const hrEnabled = rCB('txt2img_hr_enable');
  named['txt2img_hr_enable'] = hrEnabled; active.push('txt2img_hr_enable');
  if (hrEnabled) {
    named['txt2img_denoising_strength'] = rFloat('txt2img_denoising_strength'); active.push('txt2img_denoising_strength');
    named['txt2img_hr_scale'] = rFloat('txt2img_hr_scale'); active.push('txt2img_hr_scale');
    named['txt2img_hr_upscaler'] = rDD('txt2img_hr_upscaler'); active.push('txt2img_hr_upscaler');
    named['txt2img_hires_steps'] = rInt('txt2img_hires_steps'); active.push('txt2img_hires_steps');
    named['txt2img_hr_resize_x'] = rInt('txt2img_hr_resize_x'); active.push('txt2img_hr_resize_x');
    named['txt2img_hr_resize_y'] = rInt('txt2img_hr_resize_y'); active.push('txt2img_hr_resize_y');
    named['hr_checkpoint'] = rDD('hr_checkpoint'); active.push('hr_checkpoint');
    // Prefer explicit compatibility multiselect, else derive union of VAE + Text Encoders selections
    let hrUnion = rDD('hr_vae_te') || [];
    if (!Array.isArray(hrUnion) || hrUnion.length === 0 || (hrUnion.length === 1 && String(hrUnion[0]) === 'Use same choices')) {
      const v = rDD('hr_vae');
      const t = rDD('hr_text_encoders') || [];
      const vaePick = (typeof v === 'string' && v && v !== 'Use same choices') ? [v] : [];
      const picks = Array.isArray(t) ? [...vaePick, ...t] : vaePick;
      hrUnion = picks.length ? picks : ['Use same choices'];
    }
    named['hr_vae_te'] = hrUnion; active.push('hr_vae_te');
    named['hr_sampler'] = rDD('hr_sampler'); active.push('hr_sampler');
    named['hr_scheduler'] = rDD('hr_scheduler'); active.push('hr_scheduler');
    named['txt2img_hr_prompt'] = rText('hires_prompt'); active.push('txt2img_hr_prompt');
    named['txt2img_hr_neg_prompt'] = rText('hires_neg_prompt'); active.push('txt2img_hr_neg_prompt');
    named['txt2img_hr_cfg'] = rFloat('txt2img_hr_cfg'); active.push('txt2img_hr_cfg');
    if (R.isInteractive('txt2img_hr_distilled_cfg')) {
      named['txt2img_hr_distilled_cfg'] = rFloat('txt2img_hr_distilled_cfg'); active.push('txt2img_hr_distilled_cfg');
    }
  }
  named['txt2img_steps'] = rInt('txt2img_steps'); active.push('txt2img_steps');
  named['txt2img_sampling'] = rDDR('txt2img_sampling'); active.push('txt2img_sampling');
  named['txt2img_scheduler'] = rDD('txt2img_scheduler'); active.push('txt2img_scheduler');
  named['txt2img_seed'] = rSeed('txt2img_seed'); active.push('txt2img_seed');
  const showVar = rCB('txt2img_subseed_show');
  if (showVar) {
    named['txt2img_subseed_show'] = true; active.push('txt2img_subseed_show');
    named['txt2img_subseed'] = rInt('txt2img_subseed'); active.push('txt2img_subseed');
    named['txt2img_subseed_strength'] = rFloat('txt2img_subseed_strength'); active.push('txt2img_subseed_strength');
    named['txt2img_seed_resize_from_w'] = rInt('txt2img_seed_resize_from_w'); active.push('txt2img_seed_resize_from_w');
    named['txt2img_seed_resize_from_h'] = rInt('txt2img_seed_resize_from_h'); active.push('txt2img_seed_resize_from_h');
  }
  named['__active_ids'] = active;
  try {
    const dynEnabled = rCB('dynthres_enabled');
    if (dynEnabled) {
      named['dynthres_enabled'] = true; active.push('dynthres_enabled');
      named['dynthres_mimic_scale'] = R.readFloat('dynthres_mimic_scale'); active.push('dynthres_mimic_scale');
      named['dynthres_threshold_percentile'] = R.readFloat('dynthres_threshold_percentile'); active.push('dynthres_threshold_percentile');
      named['dynthres_mimic_mode'] = R.readRadioValue('dynthres_mimic_mode'); active.push('dynthres_mimic_mode');
      named['dynthres_mimic_scale_min'] = R.readFloat('dynthres_mimic_scale_min'); active.push('dynthres_mimic_scale_min');
      named['dynthres_cfg_mode'] = R.readRadioValue('dynthres_cfg_mode'); active.push('dynthres_cfg_mode');
      named['dynthres_cfg_scale_min'] = R.readFloat('dynthres_cfg_scale_min'); active.push('dynthres_cfg_scale_min');
      named['dynthres_sched_val'] = R.readFloat('dynthres_sched_val'); active.push('dynthres_sched_val');
      named['dynthres_separate_feature_channels'] = R.readRadioValue('dynthres_separate_feature_channels'); active.push('dynthres_separate_feature_channels');
      named['dynthres_scaling_startpoint'] = R.readRadioValue('dynthres_scaling_startpoint'); active.push('dynthres_scaling_startpoint');
      named['dynthres_variability_measure'] = R.readRadioValue('dynthres_variability_measure'); active.push('dynthres_variability_measure');
      named['dynthres_interpolate_phi'] = R.readFloat('dynthres_interpolate_phi'); active.push('dynthres_interpolate_phi');
    }
    const scriptList = $id('script_list');
    if (scriptList && scriptList.querySelector('select') instanceof HTMLSelectElement) {
      const sel = /** @type {HTMLSelectElement} */ (scriptList.querySelector('select'));
      named['__script_index'] = sel.selectedIndex;
    }
  } catch (e) { console.warn('[Strict] build txt2img extras failed', e); }
  return named;
}

function _buildImg2img() {
  const named = /** @type {StrictPayload} */ ({ __strict_version: 1, __source: 'img2img' });
  /** @type {string[]} */
  const active = [];
  const rText = R.readText, rDD = R.readDropdownValue, rDDR = R.readDropdownOrRadioValue, rInt = R.readInt, rFloat = R.readFloat, rSeed = R.readSeedValue, rRadioIndex = R.readRadioIndex;
  named['img2img_prompt'] = rText('img2img_prompt'); active.push('img2img_prompt');
  named['img2img_neg_prompt'] = rText('img2img_neg_prompt'); active.push('img2img_neg_prompt');
  named['img2img_styles'] = rDD('img2img_styles') || []; active.push('img2img_styles');
  named['img2img_batch_count'] = rInt('img2img_batch_count'); active.push('img2img_batch_count');
  named['img2img_batch_size'] = rInt('img2img_batch_size'); active.push('img2img_batch_size');
  named['img2img_cfg_scale'] = rFloat('img2img_cfg_scale'); active.push('img2img_cfg_scale');
  named['img2img_distilled_cfg_scale'] = rFloat('img2img_distilled_cfg_scale'); active.push('img2img_distilled_cfg_scale');
  named['img2img_image_cfg_scale'] = rFloat('img2img_image_cfg_scale'); active.push('img2img_image_cfg_scale');
  named['img2img_denoising_strength'] = rFloat('img2img_denoising_strength'); active.push('img2img_denoising_strength');
  named['img2img_selected_scale_tab'] = T.getTabIndex('img2img_tabs_resize'); active.push('img2img_selected_scale_tab');
  named['img2img_height'] = rInt('img2img_height'); active.push('img2img_height');
  named['img2img_width'] = rInt('img2img_width'); active.push('img2img_width');
  named['img2img_scale_by'] = rFloat('img2img_scale'); active.push('img2img_scale_by');
  named['img2img_resize_mode'] = rRadioIndex('resize_mode'); active.push('img2img_resize_mode');
  named['img2img_steps'] = rInt('img2img_steps'); active.push('img2img_steps');
  named['img2img_sampling'] = rDDR('img2img_sampling'); active.push('img2img_sampling');
  named['img2img_scheduler'] = rDD('img2img_scheduler'); active.push('img2img_scheduler');
  named['img2img_seed'] = rSeed('img2img_seed'); active.push('img2img_seed');
  const showVarI2I = R.readCheckbox('img2img_subseed_show');
  if (showVarI2I) {
    named['img2img_subseed_show'] = true; active.push('img2img_subseed_show');
    named['img2img_subseed'] = rInt('img2img_subseed'); active.push('img2img_subseed');
    named['img2img_subseed_strength'] = rFloat('img2img_subseed_strength'); active.push('img2img_subseed_strength');
    named['img2img_seed_resize_from_w'] = rInt('img2img_seed_resize_from_w'); active.push('img2img_seed_resize_from_w');
    named['img2img_seed_resize_from_h'] = rInt('img2img_seed_resize_from_h'); active.push('img2img_seed_resize_from_h');
  }
  const tab = T.getTabIndex('mode_img2img');
  if ([2,3,4].includes(tab)) {
    named['img2img_mask_blur'] = rInt('img2img_mask_blur'); active.push('img2img_mask_blur');
    named['img2img_mask_alpha'] = rFloat('img2img_mask_alpha'); active.push('img2img_mask_alpha');
    named['img2img_inpainting_fill'] = rRadioIndex('img2img_inpainting_fill'); active.push('img2img_inpainting_fill');
    named['img2img_inpaint_full_res'] = (rRadioIndex('img2img_inpaint_full_res') === 1); active.push('img2img_inpaint_full_res');
    named['img2img_inpaint_full_res_padding'] = rInt('img2img_inpaint_full_res_padding'); active.push('img2img_inpaint_full_res_padding');
    named['img2img_inpainting_mask_invert'] = rRadioIndex('img2img_mask_mode'); active.push('img2img_inpainting_mask_invert');
    const ci = getCanvasInpaint();
    if (ci) {
      named['img2img_mask_blur'] = ci.mask_blur;
      named['img2img_mask_alpha'] = ci.mask_alpha;
      named['img2img_inpainting_fill'] = ci.inpainting_fill;
      named['img2img_inpaint_full_res'] = ci.inpaint_full_res;
      named['img2img_inpaint_full_res_padding'] = ci.inpaint_full_res_padding;
      named['img2img_inpainting_mask_invert'] = ci.inpainting_mask_invert;
    }
  }
  named['__active_ids'] = active;
  try {
    const dynEnabled = R.readCheckbox('dynthres_enabled');
    if (dynEnabled) {
      named['dynthres_enabled'] = true; active.push('dynthres_enabled');
      named['dynthres_mimic_scale'] = R.readFloat('dynthres_mimic_scale'); active.push('dynthres_mimic_scale');
      named['dynthres_threshold_percentile'] = R.readFloat('dynthres_threshold_percentile'); active.push('dynthres_threshold_percentile');
      named['dynthres_mimic_mode'] = R.readRadioValue('dynthres_mimic_mode'); active.push('dynthres_mimic_mode');
      named['dynthres_mimic_scale_min'] = R.readFloat('dynthres_mimic_scale_min'); active.push('dynthres_mimic_scale_min');
      named['dynthres_cfg_mode'] = R.readRadioValue('dynthres_cfg_mode'); active.push('dynthres_cfg_mode');
      named['dynthres_cfg_scale_min'] = R.readFloat('dynthres_cfg_scale_min'); active.push('dynthres_cfg_scale_min');
      named['dynthres_sched_val'] = R.readFloat('dynthres_sched_val'); active.push('dynthres_sched_val');
      named['dynthres_separate_feature_channels'] = R.readRadioValue('dynthres_separate_feature_channels'); active.push('dynthres_separate_feature_channels');
      named['dynthres_scaling_startpoint'] = R.readRadioValue('dynthres_scaling_startpoint'); active.push('dynthres_scaling_startpoint');
      named['dynthres_variability_measure'] = R.readRadioValue('dynthres_variability_measure'); active.push('dynthres_variability_measure');
      named['dynthres_interpolate_phi'] = R.readFloat('dynthres_interpolate_phi'); active.push('dynthres_interpolate_phi');
    }
    const scriptList = $id('script_list');
    if (scriptList && scriptList.querySelector('select') instanceof HTMLSelectElement) {
      const sel = /** @type {HTMLSelectElement} */ (scriptList.querySelector('select'));
      named['__script_index'] = sel.selectedIndex;
    }
  } catch (e) { console.warn('[Strict] build img2img extras failed', e); }
  return named;
}

function _buildTxt2Vid() {
  const named = /** @type {StrictPayload} */ ({ __strict_version: 1, __source: 'txt2vid' });
  /** @type {string[]} */
  const active = [];
  const rText = R.readText, rDD = R.readDropdownValue, rDDR = R.readDropdownOrRadioValue, rInt = R.readInt, rSeed = R.readSeedValue;
  named['txt2vid_prompt'] = rText('txt2vid_prompt'); active.push('txt2vid_prompt');
  named['txt2vid_neg_prompt'] = rText('txt2vid_neg_prompt'); active.push('txt2vid_neg_prompt');
  named['txt2vid_styles'] = rDD('txt2vid_styles') || []; active.push('txt2vid_styles');
  named['txt2vid_width'] = rInt('txt2vid_width'); active.push('txt2vid_width');
  named['txt2vid_height'] = rInt('txt2vid_height'); active.push('txt2vid_height');
  named['txt2vid_steps'] = rInt('txt2vid_steps'); active.push('txt2vid_steps');
  named['txt2vid_fps'] = rInt('txt2vid_fps'); active.push('txt2vid_fps');
  named['txt2vid_num_frames'] = rInt('txt2vid_num_frames'); active.push('txt2vid_num_frames');
  named['txt2vid_sampling'] = rDDR('txt2vid_sampling'); active.push('txt2vid_sampling');
  named['txt2vid_scheduler'] = rDD('txt2vid_scheduler'); active.push('txt2vid_scheduler');
  named['txt2vid_seed'] = rSeed('txt2vid_seed'); active.push('txt2vid_seed');
  named['__active_ids'] = active; return named;
}

function _buildImg2Vid() {
  const named = /** @type {StrictPayload} */ ({ __strict_version: 1, __source: 'img2vid' });
  /** @type {string[]} */
  const active = [];
  const rText = R.readText, rDD = R.readDropdownValue, rDDR = R.readDropdownOrRadioValue, rInt = R.readInt, rSeed = R.readSeedValue;
  named['img2vid_prompt'] = rText('img2vid_prompt'); active.push('img2vid_prompt');
  named['img2vid_neg_prompt'] = rText('img2vid_neg_prompt'); active.push('img2vid_neg_prompt');
  named['img2vid_styles'] = rDD('img2vid_styles') || []; active.push('img2vid_styles');
  named['img2vid_width'] = rInt('img2vid_width'); active.push('img2vid_width');
  named['img2vid_height'] = rInt('img2vid_height'); active.push('img2vid_height');
  named['img2vid_steps'] = rInt('img2vid_steps'); active.push('img2vid_steps');
  named['img2vid_fps'] = rInt('img2vid_fps'); active.push('img2vid_fps');
  named['img2vid_num_frames'] = rInt('img2vid_num_frames'); active.push('img2vid_num_frames');
  named['img2vid_sampling'] = rDDR('img2vid_sampling'); active.push('img2vid_sampling');
  named['img2vid_scheduler'] = rDD('img2vid_scheduler'); active.push('img2vid_scheduler');
  named['img2vid_seed'] = rSeed('img2vid_seed'); active.push('img2vid_seed');
  named['__active_ids'] = active; return named;
}

export const build = {
  txt2img: _buildTxt2img,
  img2img: _buildImg2img,
  txt2vid: _buildTxt2Vid,
  img2vid: _buildImg2Vid,
};
