# apps/interface/src/stores Overview
<!-- tags: frontend, stores, state -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-29
Status: Active

## Purpose
- Pinia stores encapsulating shared UI/application state (engine selections, task options, session data).

## Notes
- Keep store interfaces aligned with backend schemas and avoid duplicating validation already enforced server-side.
- Derive computed state for components instead of mutating raw backend payloads.
- 2025-12-03: Result `info` now includes prompt, negative prompt, resolved seed, and default save directory so the UI surfaces real generation inputs/outputs.
- 2025-12-03: Stores track two refiner configs: a global `refiner` (for post-base pass) and `highres.refiner` nested under the hires options for a hires-coupled refiner stage.
- 2025-12-03: `xyz.ts` store runs frontend-driven XYZ sweeps (X/Y/Z axes) using the active image model tab as the baseline, with stop support and per-cell status.
- 2025-12-03: XYZ store now enqueues payload snapshots, supports stop-after-current vs stop-now (calling `/api/tasks/{id}/cancel`), and preserves hires/refiner in each job payload.
- 2025-12-04: `engine_capabilities.ts` hydrates `/engines/capabilities` (under `/api` via `API_BASE`) once and exposes a cached map keyed by semantic engine tag (sd15, sdxl, flux, wan22, hunyuan_video, svd) so views/components can hide Highres/Refiner/video-specific UI when the backend declares a surface as unsupported.
- 2025-12-05: `quicksettings.ts` exposes flags `smartOffload`/`smartFallback`/`smartCache` from `/api/options`; model-tab payload builders propagate these flags for generation-time behavior.
- 2025-12-06: Model tabs derive Flux `textEncoderOverride` from `tab.params.textEncoders` using `deriveFluxTextEncoderOverrideFromLabels` (same contract as the Flux engine payload builder).
- 2025-12-09: `quicksettings.ts` resolves SHA256 for path-prefixed text encoder labels (flux/zimage), keeps `forge_additional_modules` labels intact instead of truncating to basenames, and exposes `resolveTextEncoderSha` so composables can attach `tenc_sha` to GGUF payloads; model-tab `useGeneration` blocks runs when required TE SHAs are missing.
- 2025-12-14: `model_tabs.ts` treats tab `type` as a UI tab kind (`sd15|sdxl|flux|zimage|wan`) and normalizes legacy WAN types (`wan22_*` → `wan`); removed the legacy video Pinia store (`stores/video.ts`) now that WAN video runs exclusively via model tabs + typed payload builders.
- 2025-12-27: Image tabs now persist their checkpoint + text encoders in tab params (`checkpoint`, `textEncoders`) and `model_tabs.normalizeTab()` fills missing params with defaults at load time (so backend-saved tabs with partial `params` don’t render blank/undefined fields).
- 2025-12-16: `model_tabs.ts` WAN `video` params now include `vid2vid` controls (strength/method/chunk/flow toggles) plus optional `initVideoPath` for path-based inputs; uploaded video files are kept in-memory by `useVideoGeneration` (not persisted).
- 2025-12-17: Added `workflows.ts` store to keep `/workflows` list reactive (refresh after snapshot save/delete) and to centralize workflow persistence calls; WAN tabs also default `lowFollowsHigh=false` in `model_tabs.ts` for the Low Noise “Use High settings” toggle.
- 2025-12-28: Model-tab image params now persist `batchCount`/`batchSize` and `highres`/`refiner` in `tab.params` (defaults + normalization), enabling the legacy-style RunCard batch dropdown and Highres/Refiner controls for `/models/:tabId` image tabs.
- 2025-12-29: `model_tabs.load()` preserves the route-selected `activeId` when reloading tabs (reduces QuickSettings flicker on Vite reloads).
- 2025-12-29: `model_tabs.updateParams()` mutates the params object in-place (instead of replacing it) to reduce whole-view rerenders on small boolean toggles (e.g. LightX2V / Low “Use High settings”).
