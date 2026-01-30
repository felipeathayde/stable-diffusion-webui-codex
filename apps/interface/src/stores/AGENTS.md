# apps/interface/src/stores Overview
<!-- tags: frontend, stores, state -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-01-28
Status: Active

## Purpose
- Pinia stores encapsulating shared UI/application state (tabs, task options, session data).

## Notes
- Keep store interfaces aligned with backend schemas and avoid duplicating validation already enforced server-side.
- Derive computed state for components instead of mutating raw backend payloads.
- 2025-12-03: Result `info` now includes prompt, negative prompt, resolved seed, and default save directory so the UI surfaces real generation inputs/outputs.
- 2025-12-03: Stores track two refiner configs: a global `refiner` (for post-base pass) and `highres.refiner` nested under the hires options for a hires-coupled refiner stage.
- 2025-12-03: `xyz.ts` store runs frontend-driven XYZ sweeps (X/Y/Z axes) using the active image model tab as the baseline, with stop support and per-cell status.
- 2025-12-03: XYZ store now enqueues payload snapshots, supports stop-after-current vs stop-now (calling `/api/tasks/{id}/cancel`), and preserves hires/refiner in each job payload.
- 2025-12-04: `engine_capabilities.ts` hydrates `/engines/capabilities` (under `/api` via `API_BASE`) once and exposes a cached map keyed by semantic engine tag (sd15, sdxl, flux1, wan22, hunyuan_video, svd) so views/components can hide Highres/Refiner/video-specific UI when the backend declares a surface as unsupported.
- 2025-12-05: `quicksettings.ts` exposes flags `smartOffload`/`smartFallback`/`smartCache` from `/api/options`; model-tab payload builders propagate these flags for generation-time behavior.
- 2025-12-06: Model tabs keep per-tab Flux text encoder selections (`tab.params.textEncoders`) and `useGeneration(tabId)` sends them via `tenc_sha` (Flux.1 does not use `text_encoder_override` from the client).
- 2025-12-09: `quicksettings.ts` resolves SHA256 for path-prefixed text encoder labels (flux1/zimage), keeps `text_encoder_overrides` labels intact instead of truncating to basenames, and exposes `resolveTextEncoderSha` so composables can attach `tenc_sha` to GGUF payloads; model-tab `useGeneration` blocks runs when required TE SHAs are missing.
- 2026-01-24: `quicksettings.ts` stopped persisting UI-only globals via `/api/options` (`codex_engine`, `codex_mode`, `codex_device`, `sd_vae`); device + VAE + non-tab TE overrides now persist in localStorage (payload builders remain strict).
- 2026-01-24: Removed the global checkpoint default (`sd_model_checkpoint`); checkpoint selection is per-tab (`tab.params.checkpoint`) and request-driven (`model`/`extras.model_sha`).
- 2026-01-24: Attention backend QuickSettings no longer offers the unported `sage` option; backend enforces strict choices via `/api/settings/schema`.
- 2026-01-24: `xyz.ts` no longer tries to set `codex_engine` via `/api/options`; XYZ runs include `engine`/`model` per job payload.
- 2025-12-14: `model_tabs.ts` treats tab `type` as a UI tab kind (`sd15|sdxl|flux1|chroma|zimage|wan`) and normalizes legacy WAN types (`wan22_*` → `wan`); removed the legacy video Pinia store (`stores/video.ts`) now that WAN video runs exclusively via model tabs + typed payload builders.
- 2025-12-27: Image tabs now persist their checkpoint + text encoders in tab params (`checkpoint`, `textEncoders`) and `model_tabs.normalizeTab()` fills missing params with defaults at load time (so backend-saved tabs with partial `params` don’t render blank/undefined fields).
- 2025-12-16: `model_tabs.ts` WAN `video` params now include `vid2vid` controls (strength/method/chunk/flow toggles) plus optional `initVideoPath` for path-based inputs; uploaded video files are kept in-memory by `useVideoGeneration` (not persisted).
- 2026-01-21: `model_tabs.ts` WAN stage params store LoRA selection as `loraSha` (sha256) and payload builders emit `lora_sha` (no stage `lora_path`).
- 2025-12-17: Added `workflows.ts` store to keep `/workflows` list reactive (refresh after snapshot save/delete) and to centralize workflow persistence calls; WAN tabs also default `lowFollowsHigh=false` in `model_tabs.ts` for the Low Noise “Use High settings” toggle.
- 2025-12-28: Model-tab image params now persist `batchCount`/`batchSize` and `highres`/`refiner` in `tab.params` (defaults + normalization), enabling the legacy-style RunCard batch dropdown and Highres/Refiner controls for `/models/:tabId` image tabs.
- 2025-12-29: `model_tabs.load()` preserves the route-selected `activeId` when reloading tabs (reduces QuickSettings flicker on Vite reloads).
- 2025-12-29: `model_tabs.updateParams()` mutates the params object in-place (instead of replacing it) to reduce whole-view rerenders on small boolean toggles (e.g. LightX2V / Low “Use High settings”).
- 2026-01-01: `quicksettings.ts` now exposes a `refreshModelsList()` helper that calls `/api/models?refresh=1` so the UI can pick up new checkpoint files after changing `apps/paths.json` or copying weights into `*_ckpt` folders.
- 2026-01-02: `quicksettings.ts` now exposes `resolveModelSha` + `isModelCoreOnly` so callers can send checkpoint selection as a hash (instead of titles/paths) and still enforce core-only (GGUF) requirements.
- 2026-01-03: Added standardized file header blocks to stores (doc-only change; part of rollout).
- 2026-01-04: Flux family backend engine keys moved to `flux1*`; `chroma` is a first-class tab type mapped to `flux1_chroma` at request time.
- 2026-01-18: `engine_capabilities.ts` now also caches backend-provided `asset_contracts` (base + core-only) from `/api/engines/capabilities`; image payload builders use these contracts (plus `models[].core_only`) to enforce required VAE/text encoder selection without duplicating per-engine policy in the UI.
- 2026-01-06: Image tab defaults now use model_index-aligned canonical sampler/scheduler values (SD15: `pndm` + `ddim`; SDXL: `euler` + `euler_discrete`; flow: `euler` + `simple`) and normalization only fills blank/missing values (no alias/automatic shims).
- 2026-01-25: `model_tabs.ts` image-tab defaults now set `clipSkip=0` (auto/default sentinel). Payload builders can still send explicit clip skip values; 0 resets to engine defaults.
- 2026-01-27: `model_tabs.ts` WAN video params now include `returnFrames` (default false) to control whether frames are included in the final result payload (txt2vid/img2vid full frames; vid2vid preview frames).
- 2026-01-28: `model_tabs.ts` Z-Image params now include `zimageTurbo` (default true) to persist the Turbo/Base variant selection per tab.
- 2026-01-29: `model_tabs.ts` image-tab params now include masked img2img (“inpaint”) fields (`useMask`, `maskImageData`, and inpaint controls like enforcement/full-res/padding/invert/round/blur/fill mode) so the model tab can drive Codex-native masking end-to-end.
