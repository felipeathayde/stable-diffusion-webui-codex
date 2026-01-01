# apps/backend/engines/util Overview
<!-- tags: backend, engines, util, adapters -->
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2026-01-01
Status: Active

## Purpose
- Utility modules supporting engine execution (scheduler mapping, attention backend selection, etc.).

## Notes
- Add shared utilities here instead of duplicating helpers inside specific engine packages.
- `adapters.py` builds typed `RefinerConfig` for txt2img (global) and for highres payloads; `build_txt2img_processing` now populates `processing.refiner` and `CodexHighResConfig.refiner` when extras/highres refiner is enabled.
- `build_txt2img_processing` also wires smart flags from `Txt2ImgRequest` (`smart_offload`, `smart_fallback`, `smart_cache`) into `CodexProcessingTxt2Img` so pipeline stages can make per-job decisions.
- 2025-12-31: `build_img2img_processing` now wires `distilled_cfg_scale`/`image_cfg_scale` from request metadata and propagates smart flags into `CodexProcessingImg2Img` (needed for Flux/Kontext parity with txt2img).
- 2026-01-01: `build_{txt2img,img2img}_processing` now carries `clip_skip` into `processing.metadata` so workflows can treat it as a prompt control (without prompt-tag injection).
