<!-- tags: frontend, utils, xyz -->
# apps/interface/src/utils Overview
Date: 2025-12-03
Last Review: 2026-02-22
Status: Active

## Purpose
- Small utility helpers shared across frontend modules (parsers, formatters, pure functions).

## Notes
- Keep helpers pure and framework-agnostic so they remain easy to unit test with Vitest.
- New utilities should ship with targeted tests under `apps/interface/src/**/`.
- 2025-12-03: Added XYZ helpers (`xyz.ts`) for axis parsing/combo building used by the sweep view/store.
- 2026-01-03: Added standardized file header block to `xyz.ts` (doc-only change; part of rollout).
- 2026-01-29: Added PNG infotext parsing + sampler/scheduler mapping helpers (`pnginfo.ts`) with unit tests.
- 2026-02-18: `pnginfo.ts` parser now tokenizes KV blocks safely (quoted/bracketed comma support), captures additional A1111-style fields (`RNG`, `Eta`, `NGMS`, `Version`, `Hires Module 1`), and supports legacy JSON `parameters` fallback parsing.
- 2026-02-03: XYZ axis ids for hires are now `hires_scale` / `hires_steps`.
- 2026-02-08: XYZ keeps axis id `refiner_steps` for sweep compatibility, but the UI label now reflects swap-pointer semantics (`Swap at step`).
- 2026-02-06: Added `engine_taxonomy.ts` as canonical frontend engine taxonomy mapping (tab-family aliases, request engine-id resolution, semantic-engine resolution, and centralized sampler/scheduler fallback defaults).
- 2026-02-07: Updated Anima fallback sampling defaults to `euler/simple` in `engine_taxonomy.ts` and added `engine_taxonomy.test.ts` regression coverage.
- 2026-02-08: Added `image_params.ts` + `image_params.test.ts` for pure img2img/inpaint normalization helpers (`normalizeMaskEnforcement`, `normalizeInpaintingFill`, `normalizeNonNegativeInt`) used by `ImageModelTab.vue`.
- 2026-02-08: `image_params.ts` now also exposes hires policy/text helpers (`resolveTextOverride`, `resolveHiresModePolicy`) used by payload/view wiring to keep mode visibility and prompt fallback behavior deterministic.
- 2026-02-18: Added `img2img_resize.ts` as the canonical img2img resize-mode contract (`just_resize`, `crop_and_resize`, `resize_and_fill`, `just_resize_latent_upscale`, `upscaler`) with a shared normalizer for store/view/composable wiring.
- 2026-02-22: Added `wan_img2vid_temporal.ts` as the canonical WAN img2vid temporal helper module (`solo|chunk|sliding|svi2|svi2_pro` mode normalization + window stride/commit normalizers with `stride % 4 == 0` and `commit - stride >= 4` guards) reused by store/view/payload layers.
