# apps/backend/use_cases Overview
Date: 2025-10-30
Owner: Backend Use-Case Maintainers
Last Review: 2026-01-22
Status: Active

## Purpose
- Defines high-level orchestration flows for each supported task (txt2img, img2img, txt2vid, img2vid). Each module prepares inputs, invokes the appropriate engine, and handles post-processing.

## Key Files
- `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`, `vid2vid.py` — Task-specific pipelines that bind request parameters, engines, runtimes, and services.
- `txt2img_pipeline/` — Stage-based runner used by `txt2img.py` to prepare prompts, execute base sampling, and handle HiRes passes without monolithic functions.
- `__init__.py` — Exposes helpers to orchestrator modules in `apps/backend/core`.
- Shared orchestration logic now lives in `apps/backend/runtime/workflows/`; use cases should prefer these helpers over bespoke copies of sampler/prompt setup.

## Notes
- Introduza novos use cases sempre que uma combinação de tarefa + modo precisar de orquestração própria; mantenha a lógica focalizada em preparar entradas, chamar engines e relatar progresso, delegando detalhes de modelo para `engines/` ou `runtime/`.
- Quando adicionar um novo use case, espelhe o padrão existente e registre com o orquestrador e os contratos de API.
- 2026-01-22: `txt2img.py` now includes a canonical event wrapper (`run_txt2img`) used by engines to keep mode orchestration in the use-case layer (Option A).
- 2025-12-16: `vid2vid.py` implements WAN22 video-to-video orchestration (decode input video via ffmpeg, flow-guided chunking, optional VFI, export + metadata), plus a `vid2vid_method="wan_animate"` path that runs Diffusers `WanAnimatePipeline` from preprocessed pose/face videos + reference image.
- 2026-01-02: Added standardized file header docstrings to use case modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstrings to remaining use case modules (`__init__.py`, `img2vid.py`, `txt2img.py`, `txt2vid.py`) (doc-only change; part of rollout).
