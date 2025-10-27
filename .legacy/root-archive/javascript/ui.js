"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: UI glue for submit flows, gallery helpers, and small UX toggles.
 - Safety: Use getAppElementById for all ID lookups; never mutate submit arg order.
 - Helpers: submitWithProgress, normalizeSubmitArgs, updateInput, restoreProgress.
 - Events: listeners annotated (Keyboard/Mouse), guarded against nulls.
*/

/**
 * Various client-side helpers used by the Gradio UI build in `ui.py`.
 */

/** @typedef {Window & {
 *   submit?: (...args: unknown[]) => unknown[];
 *   submit_named?: (...args: unknown[]) => unknown[];
 *   submit_txt2img_upscale?: (...args: unknown[]) => unknown[];
 *   submit_img2img?: (...args: unknown[]) => unknown[];
 *   submit_img2img_named?: (...args: unknown[]) => unknown[];
 *   submit_txt2vid_named?: (...args: unknown[]) => unknown[];
 *   submit_img2vid_named?: (...args: unknown[]) => unknown[];
 *   restoreProgressTxt2img?: () => string | null;
 *   restoreProgressImg2img?: () => string | null;
 *   args_to_array?: typeof Array.from;
 *   selectCheckpoint?: (name: string) => void;
 *   selectVAE?: (value: unknown) => void;
 *   build?: {
 *     txt2img?: StrictBuilder;
 *     img2img?: StrictBuilder;
 *     txt2vid?: StrictBuilder;
 *     img2vid?: StrictBuilder;
 *   };
 *   __STRICT_CHECK_OK__?: boolean;
 *   __STRICT_CHECK_REASON__?: string;
 *   STRICT_CHECK_OK?: boolean;
 *   STRICT_CHECK_REASON?: string;
 * }} UIWindow */

/** @typedef {(args?: unknown[]) => Record<string, unknown> | null | undefined} StrictBuilder */
/** @typedef {{ txt2img?: StrictBuilder; img2img?: StrictBuilder; txt2vid?: StrictBuilder; img2vid?: StrictBuilder }} StrictBuilderMap */

/** @type {UIWindow} */
const uiWindow = window;

/**
 * Shortcut to the Gradio app root.
 * @returns {Document | ShadowRoot | HTMLElement}
 */
function gradioRoot() {
    return gradioApp();
}

/**
 * Retrieve an element by id, looking inside the Gradio root first with a DOM fallback.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getAppElementById(id) {
    const root = gradioRoot();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const el = root.getElementById(id);
        if (el instanceof HTMLElement) return el;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

/**
 * @returns {HTMLElement[]}
 */
function all_gallery_buttons() {
    try {
        // Prefer modular helper when available (loaded later as ESM)
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.allThumbButtons;
        if (typeof h === 'function') return h();
    } catch {}
    const buttons = gradioRoot().querySelectorAll('[style="display: block;"] .thumbnail-item.thumbnail-small');
    /** @type {HTMLElement[]} */
    const visible = [];
    buttons.forEach((elem) => {
        if (elem instanceof HTMLElement && elem.parentElement?.offsetParent) {
            visible.push(elem);
        }
    });
    return visible;
}

/**
 * @returns {HTMLElement | null}
 */
function selected_gallery_button() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.selectedButton;
        if (typeof h === 'function') return h();
    } catch {}
    return all_gallery_buttons().find((elem) => elem.classList.contains('selected')) ?? null;
}

/**
 * @returns {number}
 */
function selected_gallery_index() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.selectedIndex;
        if (typeof h === 'function') return h();
    } catch {}
    return all_gallery_buttons().findIndex((elem) => elem.classList.contains('selected'));
}

/**
 * @param {string} galleryContainer
 * @returns {NodeListOf<HTMLElement>}
 */
function gallery_container_buttons(galleryContainer) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.containerButtons;
        if (typeof h === 'function') return h(galleryContainer);
    } catch {}
    return gradioRoot().querySelectorAll(`#${galleryContainer} .thumbnail-item.thumbnail-small`);
}

/**
 * @param {string} galleryContainer
 * @returns {number}
 */
function selected_gallery_index_id(galleryContainer) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.selectedIndexIn;
        if (typeof h === 'function') return h(galleryContainer);
    } catch {}
    return Array.from(gallery_container_buttons(galleryContainer)).findIndex((elem) => elem.classList.contains('selected'));
}

/**
 * @template T
 * @param {T[]} gallery
 * @returns {[T | null][]}
 */
function extract_image_from_gallery(gallery) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.galleries?.extractSelected;
        if (typeof h === 'function') return h(gallery);
    } catch {}
    if (gallery.length === 0) {
        return [[null]];
    }
    let index = selected_gallery_index();
    if (index < 0 || index >= gallery.length) {
        index = 0;
    }
    return [[gallery[index] ?? null]];
}

uiWindow.args_to_array = Array.from;

/**
 * Extracts a useful message from unknown error values.
 * @param {unknown} error
 * @returns {string}
 */
function formatErrorMessage(error) {
    if (error && typeof error === 'object' && 'message' in error) {
        const maybeMessage = /** @type {{ message?: unknown }} */ (error).message;
        if (typeof maybeMessage === 'string' && maybeMessage.length > 0) {
            return maybeMessage;
        }
    }
    if (typeof error === 'string') {
        return error;
    }
    try {
        return JSON.stringify(error);
    } catch (_) {
        return String(error);
    }
}

/** @param {string} theme */
function set_theme(theme) {
    const gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(`${gradioURL}?__theme=${theme}`);
    }
}

function switch_to_txt2img() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToTxt2Img;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[0]?.click();
    return Array.from(arguments);
}

/** @param {number} index */
function switch_to_img2img_tab(index) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToImg2ImgTab;
        if (typeof h === 'function') return void h(index);
    } catch {}
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[1]?.click();
    const mode = getAppElementById('mode_img2img');
    if (mode) {
        const modeButtons = mode.querySelectorAll('button');
        const button = modeButtons?.[index];
        if (button instanceof HTMLElement) button.click();
    }
}

function switch_to_img2img() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToImg2Img;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToSketch;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToInpaint;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToInpaintSketch;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.switchToExtras;
        if (typeof h === 'function') return h.apply(null, arguments);
    } catch {}
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[3]?.click();
    return Array.from(arguments);
}

/**
 * @param {string} tabId
 * @returns {number}
 */
function get_tab_index(tabId) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.getTabIndex;
        if (typeof h === 'function') return h(tabId);
    } catch {}
    const tab = getAppElementById(tabId);
    const buttons = tab ? tab.querySelector('div')?.querySelectorAll('button') : null;
    if (!buttons) return 0;
    const arr = Array.from(buttons);
    for (let i = 0; i < arr.length; i += 1) {
        const btn = arr[i];
        if (btn instanceof HTMLElement && btn.classList.contains('selected')) return i;
    }
    return 0;
}

/**
 * @param {string} tabId
 * @param {IArguments} args
 */
function create_tab_index_args(tabId, args) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.createTabIndexArgs;
        if (typeof h === 'function') return h(tabId, args);
    } catch {}
    const res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.tabs?.getImg2ImgTabIndex;
        if (typeof h === 'function') return h(arguments);
    } catch {}
    const res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

/**
 * @param {IArguments | unknown[]} args
 * @returns {unknown[]}
 */
function create_submit_args(args) {
    return Array.from(/** @type {any} */ (args));
}

/**
 * @param {string} tabname
 * @param {unknown[]} res
 * @returns {unknown[]}
 */
function normalizeSubmitArgs(tabname, res) {
    try {
        for (let i = 0; i < res.length; i += 1) {
            const value = res[i];
            if (typeof value === 'string') {
                const trimmed = value.trim();
                if (/^-?\d+$/.test(trimmed)) {
                    const numeric = Number.parseInt(trimmed, 10);
                    if (!Number.isNaN(numeric)) res[i] = numeric;
                } else if (/^-?\d*\.\d+$/.test(trimmed)) {
                    const floatValue = Number.parseFloat(trimmed);
                    if (!Number.isNaN(floatValue)) res[i] = floatValue;
                }
            }
        }
    } catch (error) {
        console.warn('normalizeSubmitArgs failed:', error);
    }
    return res;
}

/**
 * @param {string} tabname
 * @param {boolean} showInterrupt
 * @param {boolean} showSkip
 * @param {boolean} showInterrupting
 */
function setSubmitButtonsVisibility(tabname, showInterrupt, showSkip, showInterrupting) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.submit?.setButtons;
        if (typeof h === 'function') return void h(tabname, showInterrupt, showSkip, showInterrupting);
    } catch {}
    const interrupt = getAppElementById(`${tabname}_interrupt`);
    const skip = getAppElementById(`${tabname}_skip`);
    const interrupting = getAppElementById(`${tabname}_interrupting`);
    if (interrupt) interrupt.style.display = showInterrupt ? 'block' : 'none';
    if (skip) skip.style.display = showSkip ? 'block' : 'none';
    if (interrupting) interrupting.style.display = showInterrupting ? 'block' : 'none';
}

/** @param {string} tabname @param {boolean} show */
function showSubmitButtons(tabname, show) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.submit?.setButtons;
        if (typeof h === 'function') return void h(tabname, !show, !show, false);
    } catch {}
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

/** @param {string} tabname */
function showSubmitInterruptingPlaceholder(tabname) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.submit?.setButtons;
        if (typeof h === 'function') return void h(tabname, false, true, true);
    } catch {}
    setSubmitButtonsVisibility(tabname, false, true, true);
}

/**
 * @param {string} tabname
 * @param {boolean} show
 */
function showRestoreProgressButton(tabname, show) {
    try {
        // @ts-ignore
        const h = window?.sdw?.helpers?.submit?.showRestore;
        if (typeof h === 'function') return void h(tabname, show);
    } catch {}
    const button = getAppElementById(`${tabname}_restore_progress`);
    if (button) button.style.setProperty('display', show ? 'flex' : 'none', 'important');
}

/**
 * @param {IArguments} args
 * @param {string} galleryContainerId
 * @param {string} galleryId
 * @param {StrictBuilder} [strictBuilder]
 * @returns {unknown[]}
 */
function submitWithProgress(args, galleryContainerId, galleryId, strictBuilder) {
    const argsArray = create_submit_args(args);
    const tabname = /** @type {string} */ (galleryContainerId.split('_', 1)[0]);
    let id;
    try {
        // Prefer ESM flow helper if available
        // @ts-ignore
        const f = window?.sdw?.helpers?.flow;
        if (f && typeof f.start === 'function') {
            id = f.start(tabname, galleryContainerId, galleryId, () => {});
        } else {
            // Legacy fallback
            showSubmitButtons(tabname, false);
            // @ts-ignore
            const q = window?.sdw?.helpers?.queue;
            id = typeof q?.randomId === 'function' ? q.randomId() : randomId();
            localSet(`${tabname}_task_id`, id);
            requestProgress(
                id,
                /** @type {HTMLElement} */ (getAppElementById(galleryContainerId)),
                getAppElementById(galleryId),
                () => {
                    showSubmitButtons(tabname, true);
                    localRemove(`${tabname}_task_id`);
                    showRestoreProgressButton(tabname, false);
                }
            );
        }
    } catch {
        // Last-resort legacy path
        showSubmitButtons(tabname, false);
        id = randomId();
        localSet(`${tabname}_task_id`, id);
        requestProgress(
            id,
            /** @type {HTMLElement} */ (getAppElementById(galleryContainerId)),
            getAppElementById(galleryId),
            () => {
                showSubmitButtons(tabname, true);
                localRemove(`${tabname}_task_id`);
                showRestoreProgressButton(tabname, false);
            }
        );

    }

    let res = normalizeSubmitArgs(tabname, Array.from(argsArray));
    res[0] = id;
    let strictPayloadString = null;
    if (typeof strictBuilder === 'function' && Array.isArray(res) && res.length > 0) {
        /** @type {Record<string, unknown> | null | undefined} */
        let strictPayload;
        try {
            strictPayload = strictBuilder(argsArray);
        } catch (error) {
            console.warn('submitWithProgress(): strict builder failed, sending diagnostic strict JSON', error);
            strictPayload = {
                __strict_version: 1,
                __source: tabname,
                __builder_error: formatErrorMessage(error),
            };
        }
        if (strictPayload && typeof strictPayload === 'object') {
            try {
                // Prefer ESM strict helper to stringify + sync hidden
                // @ts-ignore
                const s = window?.sdw?.helpers?.strict;
                if (s && typeof s.stringifyAndSync === 'function') {
                    strictPayloadString = s.stringifyAndSync(tabname, strictPayload);
                } else {
                    strictPayloadString = JSON.stringify(strictPayload);
                    // legacy sync
                    const hiddenRoot = getAppElementById(`${tabname}_named_active`);
                    if (hiddenRoot) {
                        const field = hiddenRoot.querySelector('textarea, input');
                        if (field instanceof HTMLTextAreaElement || field instanceof HTMLInputElement) {
                            field.value = strictPayloadString;
                            updateInput(field);
                        }
                    }
                }
            } catch (err) {
                console.warn('submitWithProgress(): JSON.stringify/sync failed', err, strictPayload);
                strictPayloadString = null;
            }
        } else {
            console.warn('submitWithProgress(): strict builder returned invalid payload', strictPayload);
            try {
                strictPayloadString = JSON.stringify({
                    __strict_version: 1,
                    __source: tabname,
                    __builder_error: 'strict builder returned invalid payload',
                });
            } catch {
                strictPayloadString = null;
            }
        }
        if (tabname !== 'txt2img' && Array.isArray(res) && res.length > 0 && strictPayloadString !== null) {
            res[res.length - 1] = strictPayloadString;
        }
    }
    // Sanitize positional scalars: for txt2img we don't need any; keep only id_task and strict payload string
    try {
        if (tabname === 'txt2img' && Array.isArray(res) && res.length > 0) {
            const idTask = res[0];
            const payload = strictPayloadString !== null ? strictPayloadString : JSON.stringify({
                __strict_version: 1,
                __source: tabname,
                __builder_error: 'strict payload unavailable',
            });
            res = [idTask, payload];
            return res;
        }
    } catch (e) {
        console.warn('submitWithProgress(): sanitize failed', e);
    }
    return res;
}

function submit() {
    /** @param {IArguments | unknown[] | undefined} submitArgs */
    const builder = (submitArgs) => {
        const argsArray = Array.isArray(submitArgs) ? submitArgs : Array.from(submitArgs || []);
        try {
            const strict = buildNamedTxt2img(argsArray);
            if (strict && typeof strict === 'object') {
                return strict;
            }
            console.warn('submit(): builder returned invalid payload', strict);
        } catch (error) {
            console.warn('submit(): failed to attach strict JSON', error);
            return { __strict_version: 1, __source: 'txt2img', __builder_error: formatErrorMessage(error) };
        }
        return { __strict_version: 1, __source: 'txt2img', __builder_error: 'builder returned non-object' };
    };
    return submitWithProgress(
        arguments,
        'txt2img_gallery_container',
        'txt2img_gallery',
        builder
    );
}

// ---- Strict JSON builders (DOM-based) ----
/** @param {string} id */
function readText(id) {
    const root = getAppElementById(id);
    if (!root) return '';
    const ta = root.querySelector('textarea');
    if (ta && ta instanceof HTMLTextAreaElement) return ta.value ?? '';
    const inp = root.querySelector('input[type=text]');
    if (inp && inp instanceof HTMLInputElement) return inp.value ?? '';
    return '';
}
/** @param {string} id */
function readNumber(id) {
    const root = getAppElementById(id);
    if (!root) return 0;
    const num = root.querySelector('input[type=number]');
    if (num && num instanceof HTMLInputElement) return Number(num.value || 0);
    const rng = root.querySelector('input[type=range]');
    if (rng && rng instanceof HTMLInputElement) return Number(rng.value || 0);
    return 0;
}
/** @param {string} id */
function readFloat(id) { return Number(readNumber(id)); }
/** @param {string} id */
function readInt(id) { return Math.trunc(Number(readNumber(id))); }
/** @param {string} id */
function isInteractive(id) {
    const root = getAppElementById(id);
    if (!root) return false;
    if (root instanceof HTMLElement && root.offsetParent === null) return false; // hidden
    const disabledInput = root.querySelector('input[disabled], select[disabled], textarea[disabled]');
    if (disabledInput) return false;
    return true;
}
/** @param {string} id */
function readCheckbox(id) {
    const root = getAppElementById(id);
    if (!root) return false;
    const cb = root.querySelector('input[type=checkbox]');
    return !!(cb && cb instanceof HTMLInputElement && cb.checked);
}
/** @param {string} id */
function readDropdownValue(id) {
    const root = getAppElementById(id);
    if (!root) return '';
    const sel = root.querySelector('select');
    if (!sel || !(sel instanceof HTMLSelectElement)) return '';
    if (sel.multiple) {
        return Array.from(sel.selectedOptions).map(o => o.value);
    }
    return sel.value;
}
/** @param {string} id */
function readRadioIndex(id) {
    const root = getAppElementById(id);
    if (!root) return 0;
    const buttons = root.querySelectorAll('button');
    let idx = 0;
    buttons.forEach((btn, i) => {
        if (btn instanceof HTMLElement && btn.classList.contains('selected')) idx = i;
    });
    return idx;
}
/** @param {string} id */
function readRadioValue(id) {
    const root = getAppElementById(id);
    if (!root) return '';
    const buttons = root.querySelectorAll('button');
    let value = '';
    buttons.forEach((btn) => {
        if (btn instanceof HTMLElement && btn.classList.contains('selected')) value = btn.textContent?.trim() ?? '';
    });
    return value;
}
/** @param {string} id */
function readDropdownOrRadioValue(id) {
    const dd = readDropdownValue(id);
    if (typeof dd === 'string' && dd !== '') return dd;
    return readRadioValue(id);
}
/** @param {string} id */
function readSeedValue(id) {
    const root = getAppElementById(id);
    if (!root) return -1;
    const num = root.querySelector('input[type=number]');
    if (num && num instanceof HTMLInputElement) {
        const v = Number(num.value || -1);
        return Number.isFinite(v) ? Math.trunc(v) : -1;
    }
    const text = root.querySelector('input[type=text]');
    if (text && text instanceof HTMLInputElement) {
        const t = (text.value || '').trim();
        if (/^-?\d+$/.test(t)) return Number.parseInt(t, 10);
        return -1;
    }
    return -1;
}
/** @param {IArguments | unknown[]} _args */
function buildNamedTxt2img(_args) {
    try {
        // @ts-ignore
        const strictHelper = window?.sdw?.helpers?.strict;
        const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
        if (buildMap && typeof buildMap.txt2img === 'function') {
            const argArray = Array.isArray(_args) ? _args : Array.from(_args || []);
            const built = buildMap.txt2img(argArray);
            if (built && typeof built === 'object') return built;
        }
    } catch {}
    console.warn('buildNamedTxt2img(): ESM strict builder not available');
    return { __strict_version: 1, __source: 'txt2img', __builder_error: 'esm_strict_builder_unavailable' };
}

// legacyBuildNamedTxt2img removed (ESM strict builders are canonical)

// Builders are used by *_named submit paths; no global exposure.

function submit_named() {
    /** @param {IArguments | unknown[] | undefined} submitArgs */
    const builder = (submitArgs) => {
        const argsArray = Array.isArray(submitArgs) ? submitArgs : Array.from(submitArgs || []);
        try {
            // Prefer ESM builders if available
            try {
                // @ts-ignore
                const strictHelper = window?.sdw?.helpers?.strict;
                const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
                if (buildMap && typeof buildMap.txt2img === 'function') {
                    const strict = buildMap.txt2img(argsArray);
                    if (strict && typeof strict === 'object') return strict;
                }
            } catch {}
            const strict = buildNamedTxt2img(argsArray);
            if (strict && typeof strict === 'object') {
                return strict;
            }
            console.warn('submit_named(): builder returned invalid payload', strict);
        } catch (error) {
            console.warn('submit_named(): builder failed, sending minimal strict JSON for diagnostics', error);
            return { __strict_version: 1, __source: 'txt2img', __builder_error: formatErrorMessage(error) };
        }
        return { __strict_version: 1, __source: 'txt2img', __builder_error: 'builder returned non-object' };
    };
    return submitWithProgress(
        arguments,
        'txt2img_gallery_container',
        'txt2img_gallery',
        builder
    );
}

function submit_txt2img_upscale() {
    // Use strict-named path; legacy `submit()` was removed.
    const res = submit_named(...arguments);
    res[2] = selected_gallery_index();
    return res;
}

// Removed legacy submit_img2img(); use submit_img2img_named.

/** @param {IArguments | unknown[]} _args */
function buildNamedImg2img(_args) {
    try {
        // @ts-ignore
        const strictHelper = window?.sdw?.helpers?.strict;
        const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
        if (buildMap && typeof buildMap.img2img === 'function') {
            const argArray = Array.isArray(_args) ? _args : Array.from(_args || []);
            const built = buildMap.img2img(argArray);
            if (built && typeof built === 'object') return built;
        }
    } catch {}
    console.warn('buildNamedImg2img(): ESM strict builder not available');
    return { __strict_version: 1, __source: 'img2img', __builder_error: 'esm_strict_builder_unavailable' };
}

// legacyBuildNamedImg2img removed

function submit_img2img_named() {
    /** @param {IArguments | unknown[] | undefined} submitArgs */
    const builder = (submitArgs) => {
        const argsArray = Array.isArray(submitArgs) ? submitArgs : Array.from(submitArgs || []);
        try {
            try {
                // @ts-ignore
                const strictHelper = window?.sdw?.helpers?.strict;
                const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
                if (buildMap && typeof buildMap.img2img === 'function') {
                    const strict = buildMap.img2img(argsArray);
                    if (strict && typeof strict === 'object') return strict;
                }
            } catch {}
            const strict = buildNamedImg2img(argsArray);
            if (strict && typeof strict === 'object') {
                return strict;
            }
            console.warn('submit_img2img_named(): builder returned invalid payload', strict);
        } catch (error) {
            console.warn('submit_img2img_named(): builder failed, sending minimal strict JSON for diagnostics', error);
            return { __strict_version: 1, __source: 'img2img', __builder_error: formatErrorMessage(error) };
        }
        return { __strict_version: 1, __source: 'img2img', __builder_error: 'builder returned non-object' };
    };
    return submitWithProgress(
        arguments,
        'img2img_gallery_container',
        'img2img_gallery',
        builder
    );
}

function submit_extras() {
    return submitWithProgress(arguments, 'extras_gallery_container', 'extras_gallery');
}

// -------- Video (Txt2Vid / Img2Vid) --------

/** @param {unknown[] | IArguments} _args */
function buildNamedTxt2Vid(_args) {
    try {
        // @ts-ignore
        const strictHelper = window?.sdw?.helpers?.strict;
        const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
        if (buildMap && typeof buildMap.txt2vid === 'function') {
            const argArray = Array.isArray(_args) ? _args : Array.from(_args || []);
            const built = buildMap.txt2vid(argArray);
            if (built && typeof built === 'object') return built;
        }
    } catch {}
    console.warn('buildNamedTxt2Vid(): ESM strict builder not available');
    return { __strict_version: 1, __source: 'txt2vid', __builder_error: 'esm_strict_builder_unavailable' };
}

// legacyBuildNamedTxt2Vid removed

function submit_txt2vid_named() {
    /** @param {IArguments | unknown[] | undefined} submitArgs */
    const builder = (submitArgs) => {
        const argsArray = Array.isArray(submitArgs) ? submitArgs : Array.from(submitArgs || []);
        try {
            try {
                // @ts-ignore
                const strictHelper = window?.sdw?.helpers?.strict;
                const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
                if (buildMap && typeof buildMap.txt2vid === 'function') {
                    const strict = buildMap.txt2vid(argsArray);
                    if (strict && typeof strict === 'object') return strict;
                }
            } catch {}
            const strict = buildNamedTxt2Vid(argsArray);
            if (strict && typeof strict === 'object') return strict;
        } catch (error) {
            console.warn('submit_txt2vid_named(): builder failed', error);
            return { __strict_version: 1, __source: 'txt2vid', __builder_error: formatErrorMessage(error) };
        }
        return { __strict_version: 1, __source: 'txt2vid', __builder_error: 'builder returned non-object' };
    };
    return submitWithProgress(arguments, 'txt2vid_gallery_container', 'txt2vid_gallery', builder);
}

/** @param {unknown[] | IArguments} _args */
function buildNamedImg2Vid(_args) {
    try {
        // @ts-ignore
        const strictHelper = window?.sdw?.helpers?.strict;
        const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
        if (buildMap && typeof buildMap.img2vid === 'function') {
            const argArray = Array.isArray(_args) ? _args : Array.from(_args || []);
            const built = buildMap.img2vid(argArray);
            if (built && typeof built === 'object') return built;
        }
    } catch {}
    console.warn('buildNamedImg2Vid(): ESM strict builder not available');
    return { __strict_version: 1, __source: 'img2vid', __builder_error: 'esm_strict_builder_unavailable' };
}

// legacyBuildNamedImg2Vid removed

function submit_img2vid_named() {
    /** @param {IArguments | unknown[] | undefined} submitArgs */
    const builder = (submitArgs) => {
        const argsArray = Array.isArray(submitArgs) ? submitArgs : Array.from(submitArgs || []);
        try {
            try {
                // @ts-ignore
                const strictHelper = window?.sdw?.helpers?.strict;
                const buildMap = strictHelper && typeof strictHelper.build === 'object' ? /** @type {StrictBuilderMap} */ (strictHelper.build) : null;
                if (buildMap && typeof buildMap.img2vid === 'function') {
                    const strict = buildMap.img2vid(argsArray);
                    if (strict && typeof strict === 'object') return strict;
                }
            } catch {}
            const strict = buildNamedImg2Vid(argsArray);
            if (strict && typeof strict === 'object') return strict;
        } catch (error) {
            console.warn('submit_img2vid_named(): builder failed', error);
            return { __strict_version: 1, __source: 'img2vid', __builder_error: formatErrorMessage(error) };
        }
        return { __strict_version: 1, __source: 'img2vid', __builder_error: 'builder returned non-object' };
    };
    return submitWithProgress(arguments, 'img2vid_gallery_container', 'img2vid_gallery', builder);
}

/** @param {string} tabname */
function restoreProgress(tabname) {
    showRestoreProgressButton(tabname, false);
    try {
        // @ts-ignore
        const f = window?.sdw?.helpers?.flow;
        if (f && typeof f.resume === 'function') return f.resume(tabname);
    } catch {}
    const id = localGet(`${tabname}_task_id`);
    if (typeof id !== 'string') return null;
    showSubmitInterruptingPlaceholder(tabname);
    requestProgress(
        id,
        /** @type {HTMLElement} */ (getAppElementById(`${tabname}_gallery_container`)),
        getAppElementById(`${tabname}_gallery`),
        () => showSubmitButtons(tabname, true),
        undefined,
        0
    );
    return id;
}

function restoreProgressTxt2img() {
    return restoreProgress('txt2img');
}

function restoreProgressImg2img() {
    return restoreProgress('img2img');
}

// Export strict-named submitters
uiWindow.submit_txt2img_upscale = submit_txt2img_upscale;
uiWindow.submit_named = submit_named;
uiWindow.restoreProgressTxt2img = restoreProgressTxt2img;
uiWindow.submit_txt2vid_named = submit_txt2vid_named;
uiWindow.submit_img2vid_named = submit_img2vid_named;
uiWindow.restoreProgressImg2img = restoreProgressImg2img;
uiWindow.submit_img2img_named = submit_img2img_named;

// Strict-compat aliases: keep legacy names but route to strict builders.
// This does NOT reintroduce legacy server behavior; it only maps old _js hooks to strict JSON submitters.
function submit() {
    const args = create_submit_args(arguments);
    return submit_named(...args);
}
function submit_img2img() {
    const args = create_submit_args(arguments);
    return submit_img2img_named(...args);
}
uiWindow.submit = submit;
uiWindow.submit_img2img = submit_img2img;

/**
 * Configure the width and height elements on `tabname` to accept pasting of resolutions.
 * @param {string} tabname
 */
function setupResolutionPasting(tabname) {
    try {
        // @ts-ignore
        const c = window?.sdw?.helpers?.clipboard;
        if (c && typeof c.setupResolutionPasting === 'function') return void c.setupResolutionPasting(tabname);
    } catch {}
    const width = gradioRoot().querySelector(`#${tabname}_width input[type=number]`);
    const height = gradioRoot().querySelector(`#${tabname}_height input[type=number]`);
    [width, height].forEach((el) => {
        if (!(el instanceof HTMLInputElement)) return;
        el.addEventListener('paste', (event) => {
            const text = event.clipboardData?.getData('text/plain') ?? '';
            const match = text.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (match && match[1] && match[2] && width instanceof HTMLInputElement && height instanceof HTMLInputElement) {
                width.value = match[1];
                height.value = match[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    });
}

// ---- Self-checks for strict submit wiring ----
onUiLoaded(() => {
    try {
        const hasTxt = !!getAppElementById('txt2img_named_active');
        const hasImg = !!getAppElementById('img2img_named_active');
        const hasNamed = (typeof uiWindow.submit_named === 'function');
        const hasImgNamed = (typeof uiWindow.submit_img2img_named === 'function');
        if (!hasTxt || !hasImg || !hasNamed || !hasImgNamed) {
            console.warn('[StrictSubmitCheck] Missing pieces', {
                txt2img_named_active_present: hasTxt,
                img2img_named_active_present: hasImg,
                submit_named_exported: hasNamed,
                submit_img2img_named_exported: hasImgNamed,
            });
            try {
                uiWindow.__STRICT_CHECK_OK__ = false;
                uiWindow.__STRICT_CHECK_REASON__ = 'missing pieces';
                uiWindow.STRICT_CHECK_OK = false;
                uiWindow.STRICT_CHECK_REASON = 'missing pieces';
            } catch {}
        } else {
            console.info('[StrictSubmitCheck] OK: strict submit handlers and hidden JSON slots detected.');
            try {
                uiWindow.__STRICT_CHECK_OK__ = true;
                uiWindow.__STRICT_CHECK_REASON__ = '';
                uiWindow.STRICT_CHECK_OK = true;
                uiWindow.STRICT_CHECK_REASON = '';
            } catch {}
        }
        // Harden legacy hook names to strict submitters and freeze them
        try {
            /** @param {string} name @param {(...args: unknown[]) => unknown} fn */
            const bindStrict = (name, fn) => {
                const desc = Object.getOwnPropertyDescriptor(uiWindow, name);
                const same = !!desc && typeof desc.value === 'function' && desc.value === fn;
                if (!same) {
                    Object.defineProperty(uiWindow, name, { value: fn, writable: false, configurable: false });
                    console.info(`[StrictSubmitCheck] Bound and froze window.${name} -> strict submitter.`);
                }
            };
            if (typeof uiWindow.submit_named === 'function') bindStrict('submit', uiWindow.submit_named);
            if (typeof uiWindow.submit_img2img_named === 'function') bindStrict('submit_img2img', uiWindow.submit_img2img_named);
        } catch (e) {
            console.warn('[StrictSubmitCheck] Failed to freeze legacy hook aliases', e);
        }
    } catch (e) {
        console.warn('[StrictSubmitCheck] Failed to run startup checks', e);
        try {
            uiWindow.__STRICT_CHECK_OK__ = false;
            uiWindow.__STRICT_CHECK_REASON__ = String(e);
            uiWindow.STRICT_CHECK_OK = false;
            uiWindow.STRICT_CHECK_REASON = String(e);
        } catch {}
    }
});

onUiLoaded(() => {
    showRestoreProgressButton('txt2img', Boolean(restoreProgressTxt2img()));
    showRestoreProgressButton('img2img', Boolean(restoreProgressImg2img()));
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});

function modelmerger() {
    const id = randomId();
    try {
        // @ts-ignore
        const q = window?.sdw?.helpers?.queue;
        if (typeof q?.requestProgress === 'function') {
            q.requestProgress(id, /** @type {HTMLElement} */ (getAppElementById('modelmerger_results_panel')), null, () => {});
        } else {
            requestProgress(id, /** @type {HTMLElement} */ (getAppElementById('modelmerger_results_panel')), null, () => {});
        }
    } catch {
        requestProgress(id, /** @type {HTMLElement} */ (getAppElementById('modelmerger_results_panel')), null, () => {});
    }
    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

/** @param {unknown} _ @param {string} promptText @param {string} negativePromptText */
function ask_for_style_name(_, promptText, negativePromptText) {
    const name = prompt('Style name:') ?? '';
    return [name, promptText, negativePromptText];
}

/** @param {string} promptValue @param {string} negativePromptValue */
function confirm_clear_prompt(promptValue, negativePromptValue) {
    if (confirm('Delete prompt?')) {
        promptValue = '';
        negativePromptValue = '';
    }
    return [promptValue, negativePromptValue];
}

/** @type {Record<string, unknown>} */
var opts = {};

onAfterUiUpdate(() => {
    try {
        // If ESM options installer is active, skip legacy wiring
        // @ts-ignore
        if (window?.sdw?.helpers?.options) return;
    } catch {}
    if (Object.keys(opts).length !== 0) return;

    const jsonElem = getAppElementById('settings_json');
    if (!jsonElem) return;

    const textarea = jsonElem.querySelector('textarea');
    if (!(textarea instanceof HTMLTextAreaElement)) return;

    try {
        opts = JSON.parse(textarea.value);
    } catch (error) {
        console.error('Failed to parse settings_json:', error);
        return;
    }

    executeCallbacks(optionsAvailableCallbacks, undefined, 'onOptionsAvailable');
    executeCallbacks(optionsChangedCallbacks, undefined, 'onOptionsChanged');

    const valueDescriptor = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
    if (valueDescriptor) {
        const originalGet = valueDescriptor.get?.bind(textarea);
        const originalSet = valueDescriptor.set?.bind(textarea);
        Object.defineProperty(textarea, 'value', {
            set(newValue) {
                const oldValue = typeof originalGet === 'function' ? originalGet() : textarea.value;
                if (typeof originalSet === 'function') originalSet(newValue);
                if (oldValue !== newValue) {
                    try {
                        opts = JSON.parse(textarea.value);
                    } catch (error) {
                        console.error('Failed to parse updated settings_json:', error);
                    }
                }
                executeCallbacks(optionsChangedCallbacks, undefined, 'onOptionsChanged');
            },
            get() {
                return typeof originalGet === 'function' ? originalGet() : textarea.value;
            }
        });
    }

    jsonElem.parentElement?.style && (jsonElem.parentElement.style.display = 'none');
});

onOptionsChanged(() => {
    try {
        // If ESM options is managing UI updates, no-op here
        // @ts-ignore
        if (window?.sdw?.helpers?.options) return;
    } catch {}
    const elem = getAppElementById('sd_checkpoint_hash');
    const hash = typeof opts.sd_checkpoint_hash === 'string' ? opts.sd_checkpoint_hash : '';
    const shortHash = hash.substring(0, 10);
    if (elem && elem.textContent !== shortHash) {
        elem.textContent = shortHash;
        elem.title = hash;
        elem.setAttribute('href', `https://google.com/search?q=${hash}`);
    }
});

/** @type {HTMLTextAreaElement | null} */ let txt2img_textarea = null;
/** @type {HTMLTextAreaElement | null} */ let img2img_textarea = null;

function restart_reload() {
    document.body.style.backgroundColor = 'var(--background-fill-primary)';
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    const requestPing = () => {
        requestGet('./internal/ping', {}, () => {
            location.reload();
        }, () => {
            setTimeout(requestPing, 500);
        });
    };
    setTimeout(requestPing, 2000);
    return [];
}

/**
 * Trigger an input event for Gradio textboxes after programmatic edits.
 * @param {HTMLInputElement | HTMLTextAreaElement} target
 */
function updateInput(target) {
    try {
        // @ts-ignore
        const d = window?.sdw?.helpers?.dom;
        if (d && typeof d.updateInput === 'function') return void d.updateInput(target);
    } catch {}
    const event = new Event('input', { bubbles: true });
    target.dispatchEvent(event);
}

/** @type {string | null} */
let desiredCheckpointName = null;
/** @param {string} name */
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    getAppElementById('change_checkpoint')?.click();
}

/** @type {string | null} */
let desiredVAEName = null;
/** @type {string[] | null} */
let desiredVAEExtras = null;
/** @param {unknown} vae */
function selectVAE(vae) {
    let selected = null;
    let extras = [];

    if (Array.isArray(vae)) {
        const first = vae.length > 0 ? vae[0] : null;
        selected = typeof first === 'string' ? first : null;
        extras = vae.slice(1).filter((item) => typeof item === 'string');
    } else if (vae && typeof vae === 'object') {
        const obj = /** @type {{ vae?: unknown, modules?: unknown }} */ (vae);
        selected = typeof obj.vae === 'string' ? obj.vae : null;
        if (Array.isArray(obj.modules)) {
            extras = obj.modules.filter((item) => typeof item === 'string');
        }
    } else if (typeof vae === 'string') {
        selected = vae;
    }

    if (selected === 'Built in') {
        selected = 'Automatic';
    }

    desiredVAEName = selected;
    desiredVAEExtras = extras;
}

/** @param {number} w @param {number} h @param {number} r */
function currentImg2imgSourceResolution(w, h, r) {
    const img = gradioRoot().querySelector('#mode_img2img > div[style="display: block;"] img, #mode_img2img > div[style="display: block;"] canvas');
    if (img instanceof HTMLImageElement) {
        return [img.naturalWidth || img.width, img.naturalHeight || img.height, r];
    } else if (img instanceof HTMLCanvasElement) {
        return [img.width, img.height, r];
    }
    return [w, h, r];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    setTimeout(() => {
        getAppElementById('img2img_update_resize_to')?.click();
    }, 500);
    return [];
}

/** @param {string} elemId */
function setRandomSeed(elemId) {
    const input = gradioRoot().querySelector(`#${elemId} input`);
    if (!(input instanceof HTMLInputElement)) return [];
    input.value = '-1';
    updateInput(input);
    return [];
}

/** @param {string} tabname */
function switchWidthHeight(tabname) {
    const width = gradioRoot().querySelector(`#${tabname}_width input[type=number]`);
    const height = gradioRoot().querySelector(`#${tabname}_height input[type=number]`);
    if (!(width instanceof HTMLInputElement) || !(height instanceof HTMLInputElement)) return [];
    const tmp = width.value;
    width.value = height.value;
    height.value = tmp;
    updateInput(width);
    updateInput(height);
    return [];
}

/** @type {Record<string, number>} */
const onEditTimers = {};

/**
 * Register a throttled input handler.
 * @param {string} editId
 * @param {HTMLInputElement | HTMLTextAreaElement | null} elem
 * @param {number} afterMs
 * @param {() => void} func
 * @returns {() => void}
 */
function onEdit(editId, elem, afterMs, func) {
    if (!elem) {
        return () => {};
    }
    const edited = () => {
        const existingTimer = onEditTimers[editId];
        if (existingTimer) window.clearTimeout(existingTimer);
        onEditTimers[editId] = window.setTimeout(func, afterMs);
    };
    elem.addEventListener('input', edited);
    return edited;
}

// expose globals for legacy hooks
Object.assign(uiWindow, {
    set_theme,
    all_gallery_buttons,
    selected_gallery_button,
    selected_gallery_index,
    gallery_container_buttons,
    selected_gallery_index_id,
    extract_image_from_gallery,
    switch_to_txt2img,
    switch_to_img2img,
    switch_to_sketch,
    switch_to_inpaint,
    switch_to_inpaint_sketch,
    switch_to_extras,
    get_tab_index,
    create_tab_index_args,
    get_img2img_tab_index,
    create_submit_args,
    normalizeSubmitArgs,
    setSubmitButtonsVisibility,
    showSubmitButtons,
    showSubmitInterruptingPlaceholder,
    showRestoreProgressButton,
    submit_extras,
    modelmerger,
    ask_for_style_name,
    confirm_clear_prompt,
    updateInput,
    selectCheckpoint,
    selectVAE,
    currentImg2imgSourceResolution,
    updateImg2imgResizeToTextAfterChangingImage,
    setRandomSeed,
    switchWidthHeight,
    onEdit,
});
