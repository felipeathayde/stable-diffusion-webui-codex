# apps/interface/src/stores Overview
<!-- tags: frontend, stores, state -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-05
Status: Active

## Purpose
- Pinia stores encapsulating shared UI/application state (engine selections, task options, session data).

## Notes
- Keep store interfaces aligned with backend schemas and avoid duplicating validation already enforced server-side.
- Derive computed state for components instead of mutating raw backend payloads.
- `sdxl.ts` mirrors `txt2img.ts` with Codex SDXL defaults; prefer reusing shared helpers when adjusting both stores.
- 2025-11-03: SDXL store persists profiles locally (`loadProfile`/`saveProfile`) so the `/sdxl` view can reuse saved parameters.
- 2025-12-03: Result `info` now includes prompt, negative prompt, resolved seed, and default save directory so the UI surfaces real generation inputs/outputs.
- 2025-12-03: Stores track two refiner configs: a global `refiner` (for post-base pass) and `highres.refiner` nested under the hires options for a hires-coupled refiner stage.
- 2025-12-03: `xyz.ts` store runs frontend-driven XYZ sweeps (X/Y/Z axes) using the current SDXL form as baseline, with stop support and per-cell status.
- 2025-12-03: XYZ store now enqueues payload snapshots, supports stop-after-current vs stop-now (calling `/api/tasks/{id}/cancel`), and preserves hires/refiner in each job payload.
- 2025-12-04: `engine_capabilities.ts` hydrates `/engines/capabilities` (under `/api` via `API_BASE`) once and exposes a cached map keyed by semantic engine tag (sd15, sdxl, flux, wan22, hunyuan_video, svd) so views/components can hide Highres/Refiner/video-specific UI when the backend declares a surface as unsupported.
- 2025-12-05: `quicksettings.ts`, `txt2img.ts` e `sdxl.ts` agora expõem flags `smartOffload`/`smartFallback`/`smartCache` alimentadas por `/api/options`; os stores propagam esses valores para os payloads de geração (`smart_offload`/`smart_fallback`/`smart_cache`) para controlar descarregamento entre estágios, fallback para CPU em caso de OOM e caches SDXL (TEnc/embeds).
- 2025-12-05: `flux.ts` passa a injetar `textEncoderOverride` nos payloads de `/flux` usando o primeiro text encoder selecionado nas QuickSettings (`currentTextEncoders[0]`, family=`'flux'`), alinhando o store Flux com a superfície de overrides de text encoder usada pelos model tabs de imagem; seleções adicionais de text encoder são persistidas como parte de `currentTextEncoders` para uso futuro, mas ainda não são consumidas pelo backend.
