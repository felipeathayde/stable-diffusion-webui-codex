# apps/backend/engines/util Overview
<!-- tags: backend, engines, util, adapters -->
Date: 2025-10-28
Last Review: 2026-02-27
Status: Active

## Purpose
- Utility modules supporting engine execution (scheduler mapping, attention backend selection, etc.).

## Notes
- Add shared utilities here instead of duplicating helpers inside specific engine packages.
- `adapters.py` builds typed `RefinerConfig` for txt2img (global) and for hires payloads; `build_txt2img_processing` now populates `processing.refiner` and `CodexHiresConfig.refiner` when `extras.hires.refiner` is enabled.
- `build_txt2img_processing` also wires smart flags from `Txt2ImgRequest` (`smart_offload`, `smart_fallback`, `smart_cache`) into `CodexProcessingTxt2Img` so pipeline stages can make per-job decisions.
- 2025-12-31: `build_img2img_processing` now wires `distilled_cfg_scale`/`image_cfg_scale` from request metadata and propagates smart flags into `CodexProcessingImg2Img` (needed for Flux/Kontext parity with txt2img).
- 2026-01-01: `build_{txt2img,img2img}_processing` now carries `clip_skip` into `processing.metadata` so workflows can treat it as a prompt control (without prompt-tag injection).
- 2026-01-29: `build_img2img_processing` now maps mask/inpaint controls from `Img2ImgRequest` into `CodexProcessingImg2Img` (mask enforcement + blur/invert/full-res/filled-content knobs).
- 2026-01-02: Added standardized file header docstrings to engine util modules (doc-only change; part of rollout).
- 2026-01-06: `schedulers.py` now expects canonical sampler/scheduler strings (lowercase; spaces/`++` preserved); empty/unknown values raise immediately.
- 2026-02-08: `build_{txt2img,img2img}_processing` now copies `extras.er_sde` mappings before storing overrides, preserving request-local ER-SDE options without shared mutable aliasing.
- 2026-02-08: `adapters._build_refiner_config` now uses swap-pointer semantics (`switch_at_step` → `RefinerConfig.swap_at_step`) instead of refiner step-count semantics.
- 2026-02-21: `attention_backend.py` now sources defaults from runtime memory config and applies diffusers SDPA flags from effective attention policy (with explicit warning fallback when flash appears unavailable).
- 2026-02-27: `adapters.build_img2img_processing(...)` now propagates `image_mask` and `round_image_mask` alongside `mask`/`mask_round` to keep masked img2img (inpaint) semantics consistent in hires/conditioning paths.
