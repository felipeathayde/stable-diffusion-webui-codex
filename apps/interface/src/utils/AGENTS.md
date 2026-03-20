<!-- tags: frontend, utils, xyz -->
# apps/interface/src/utils Overview
Date: 2025-12-03
Last Review: 2026-03-20
Status: Active

## Purpose
- Small utility helpers shared across frontend modules (parsers, formatters, pure functions).

## Notes
- Keep helpers pure and framework-agnostic so manual validation remains deterministic.
- For utility changes, run `cd apps/interface && npm run typecheck` and validate impacted UI flows manually.
- 2026-03-20: Added `image_request_contract.ts` as the canonical pure helper for frontend image request selector/extras resolution; `useGeneration.ts` and `xyz.ts` must both reuse it for checkpoint metadata, FLUX.2 guidance mode, asset-contract-backed `tenc_sha`/`vae_sha`, `vae_source`, and ZImage variant extras instead of duplicating that logic.
- 2025-12-03: Added XYZ helpers (`xyz.ts`) for axis parsing/combo building used by the sweep view/store.
- 2026-01-03: Added standardized file header block to `xyz.ts` (doc-only change; part of rollout).
- 2026-01-29: Added PNG infotext parsing + sampler/scheduler mapping helpers (`pnginfo.ts`).
- 2026-02-18: `pnginfo.ts` parser now tokenizes KV blocks safely (quoted/bracketed comma support), captures additional A1111-style fields (`RNG`, `Eta`, `NGMS`, `Version`, `Hires Module 1`), and supports legacy JSON `parameters` fallback parsing.
- 2026-03-07: `pnginfo.ts` sampler/scheduler mapping no longer rewrites `normal -> simple`; import mapping now preserves individually recognized global sampler/scheduler values while still rejecting incompatible sampler->allowed_scheduler pairs.
- 2026-03-07: `pnginfo.ts` warning text now distinguishes sampler-only vs scheduler-only failures so partial import patching stays technically honest.
- 2026-02-03: XYZ axis ids for hires are now `hires_scale` / `hires_steps`.
- 2026-02-08: XYZ keeps axis id `refiner_steps` for sweep compatibility, but the UI label now reflects swap-pointer semantics (`Swap at step`).
- 2026-02-06: Added `engine_taxonomy.ts` as canonical frontend engine taxonomy mapping (tab-family aliases, request engine-id resolution, semantic-engine resolution, and centralized sampler/scheduler fallback defaults).
- 2026-03-08: `engine_taxonomy.ts` fallback sampling defaults keep `sd15` on live executable defaults (`ddim`/`ddim`).
- 2026-02-07: Updated Anima fallback sampling defaults to `euler/simple` in `engine_taxonomy.ts`.
- 2026-02-08: Added `image_params.ts` for pure img2img/inpaint normalization helpers (`normalizeMaskEnforcement`, `normalizeInpaintingFill`, `normalizeNonNegativeInt`) used by `ImageModelTab.vue`.
- 2026-02-25: `image_params.normalizeMaskEnforcement(...)` now defaults unknown values to `per_step_clamp` (Forge-engine baseline), while preserving explicit `post_blend` as legacy opt-in.
- 2026-03-06: `image_params.ts` now also exposes hires policy/text helpers (`resolveTextOverride`, `resolveHiresModePolicy`) used by payload/view wiring to keep prompt fallback deterministic and hires visibility/reset mask-aware for img2img.
- 2026-02-18: Added `img2img_resize.ts` as the canonical img2img resize-mode contract (`just_resize`, `crop_and_resize`, `resize_and_fill`, `just_resize_latent_upscale`, `upscaler`) with a shared normalizer for store/view/composable wiring.
- 2026-02-22: Added `wan_img2vid_temporal.ts` as the canonical WAN img2vid temporal helper module (`solo|chunk|sliding|svi2|svi2_pro` mode normalization + window stride/commit normalizers with `stride % 4 == 0` and `commit - stride >= 4` guards) reused by store/view/payload layers.
- 2026-02-28: `wan_img2vid_temporal.ts` removed `chunk` from `WanImg2VidMode`; normalization is now fail-loud for `img2vid_mode='chunk'` and any unsupported value (`solo|sliding|svi2|svi2_pro` only).
- 2026-03-01: Added `wan_img2vid_frame_projection.ts` as the canonical WAN no-stretch frame-projection helper (free `imageScale > 0` + normalized crop offsets), including deterministic crop rect, slack bounds for drag clamping, and scaled-dimension metadata for init-image zoom overlays.
- 2026-03-05: `engine_taxonomy.ts` now includes first-class `flux2` semantic/request ids (no `flux1`/`flux1_kontext` alias), matching the backend FLUX.2 Klein 4B / base-4B slice, and `model_family_filters.ts` maps `flux2` to `flux2_ckpt` roots.
- 2026-03-12: `engine_taxonomy.ts` now includes first-class `ltx2` tab/semantic/request ids with fallback sampling defaults `euler/simple`; `resolveImageRequestEngineId(...)` explicitly rejects `ltx2` because image routes are not valid for the LTX video lane, and `model_family_filters.ts` maps `ltx2` to `ltx2_ckpt`/`ltx2_vae` style roots plus `ltx` filename fallbacks.
- 2026-03-06: `engine_taxonomy.ts` now also exports `supportsImg2ImgMaskingForEngineId(...)` as the shared frontend truth source for engines that expose init-image img2img but still reject mask/inpaint semantics (`flux1_kontext` remains blocked; `flux2` is now allowed after backend support landed).
