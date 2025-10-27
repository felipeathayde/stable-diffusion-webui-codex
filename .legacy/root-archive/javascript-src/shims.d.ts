// Ambient declarations for legacy global helpers injected by script.js
// Define a minimal DOM surface used across the JS codebase instead of the wide union
// so consumers can safely call getElementById/querySelector without extra narrowing.
type GradioLikeRoot = {
  getElementById(id: string): HTMLElement | null;
  querySelector<K extends keyof HTMLElementTagNameMap>(selector: K): HTMLElementTagNameMap[K] | null;
  querySelector(selector: string): Element | null;
  querySelectorAll(selector: string): NodeListOf<Element>;
};

declare function gradioApp(): GradioLikeRoot;

type UiMutationCallback = (mutations: MutationRecord[]) => void;
type UiCallback = () => void;

declare function onUiLoaded(cb: UiCallback): void;
declare function onUiUpdate(cb: UiMutationCallback): void;
declare function onAfterUiUpdate(cb: UiCallback): void;
declare function onUiTabChange(cb: UiCallback): void;
declare function onOptionsChanged(cb: UiCallback): void;
declare function onOptionsAvailable(cb: UiCallback): void;

declare function updateInput(el: HTMLElement): void;
declare function get_tab_index(tabId: string): number;

interface StableDiffusionOptions extends Record<string, unknown> {
  show_progress_in_title?: boolean;
  show_progressbar?: boolean;
  prevent_screen_sleep_during_generation?: boolean;
  notification_volume?: number;
  return_grid?: boolean;
  live_preview_refresh_period?: number;
  js_modal_lightbox?: boolean;
  js_modal_lightbox_initially_zoomed?: boolean;
  js_modal_lightbox_gamepad?: boolean;
  js_modal_lightbox_gamepad_repeat?: number;
  js_live_preview_in_modal_lightbox?: boolean;
  disable_token_counters?: boolean;
  keyedit_precision_attention?: number;
  keyedit_precision_extra?: number;
  keyedit_move?: string;
  keyedit_delimiters?: string;
  keyedit_delimiters_whitespace?: string;
  show_progress_type?: string;
  use_old_hires_fix_width_height?: boolean;
  extra_networks_add_text_separator?: string;
  lora_filter_disabled?: boolean;
  _categories?: Record<string, string[]>;
  _comments_before?: Record<string, string>;
  _comments_after?: Record<string, string>;
  sd_checkpoint_hash?: string | null;
}

declare var opts: StableDiffusionOptions;

interface LocalizationDictionary extends Record<string, string> {
  rtl?: boolean;
}

interface GradioComponentConfig {
  id: number;
  props: {
    elem_id?: string;
    webui_tooltip?: string;
    placeholder?: string;
    [key: string]: unknown;
  };
}

interface GradioConfig {
  components: GradioComponentConfig[];
}

declare const localization: LocalizationDictionary;

declare global {
  interface Window {
    localization?: LocalizationDictionary;
    gradio_config: GradioConfig;
    _uiUpdQ?: UiMutationCallback[];
    _uiAfterQ?: UiCallback[];
    _uiLoadQ?: UiCallback[];
    opts: StableDiffusionOptions;
    args_to_array?: typeof Array.from;
    inputAccordionChecked?: (id: string, checked: boolean) => void;
    __SDW_OPTS_INSTALLED__?: boolean;

    // Legacy Codex namespace used by some JS helpers
    Codex?: {
      Components?: Record<string, unknown> & {
        Readers?: Record<string, unknown>;
      };
      bus?: {
        on(event: string, fn: (payload?: unknown) => void): void;
        emit(event: string, payload?: unknown): void;
      };
    };

    // SD WebUI modular helpers exposed by codex.ui.app.mjs
    sdw?: {
      ui?: {
        registry?: { register(c: { name: string; mount(root: HTMLElement): void }): void };
      };
      helpers?: {
        dom?: { updateInput(el: HTMLElement): void };
        readers?: Record<string, unknown>;
        tabs?: Record<string, unknown>;
        queue?: { requestProgress: Function; randomId: Function };
        flow?: Record<string, unknown>;
        strict?: Record<string, unknown>;
        options?: Record<string, unknown>;
        events?: Record<string, unknown>;
        lightbox?: { install(): void; afterUpdate(): void; switchRelative(delta: number): void };
        lightboxGamepad?: { install(): void };
        contextMenu?: { install(): void; append?: Function; remove?: Function };
        hotkeys?: { install(): void };
        galleryKeys?: { install(): void };
      };
    };
  }
}

export {};
