Repository Inventory and Governance — Root Directories
Date: 2025-10-25

Summary
- This document catalogs every top-level directory outside `apps/`, classifies it (Active, Compat, Vendor, Dev Tools, Data, Reference, Deprecated), explains its purpose, where it is used from the active backend, removal risk, and next steps.

Legend
- Category: Active | Compat | Vendor | Dev | Data | Reference | Deprecated
- Risk (removal): Do‑not | High | Medium | Low

Active (entry points live under `apps/`)
- apps/ — Active backend (`apps/server/backend`) and new UI (`apps/interface`).
- codex/ — Internal docs and specs (architecture, guides, roadmaps).
- .sangoi/ — Operational logs (task‑logs, handoffs, CHANGELOG).

Compat (legacy stack ainda consumido — sem novos acoplamentos)
- modules/ (Category: Compat; Risk: Do‑not)
  - Purpose: A1111 core (samplers, loaders, shared state, legacy UI glue).
  - Used by: `apps/server/run_api.py` (initialize, script_callbacks, sd_models, sd_samplers, sd_schedulers, call_queue, txt2img/img2img/txt2vid/img2vid); services (image/options/progress); engines util.
  - Governance: keep read/write; no new backend‑to‑legacy coupling beyond what exists.

- modules_forge/ (Category: Compat; Risk: High)
  - Purpose: Forge initialization and utilities (patchers, preprocessors, canvas, memory utils).
  - Used by: módulos/extensões legado (fora do backend ativo).
  - Backend: toda gestão nativa usa `apps.server.backend.codex.*` (sem `modules_forge.*`).
  - Governance: manter por ora; não criar novos usos no backend. Migração contínua para APIs Codex.

- k_diffusion/ (Category: Vendor/Compat; Risk: High)
  - Purpose: third‑party samplers used by `modules/sd_samplers_*`.
  - Used by: legacy samplers; indirectly by backend through `modules`.
  - Governance: treat as vendored dependency.

- extensions-builtin/ (Category: Compat; Risk: High)
  - Purpose: built‑in extensions (ControlNet, IP‑Adapter, LoRA, preprocessors, etc.).
  - Used by: loaded by `modules/extensions.py` (legacy flow) and referenced from modules_forge code.
  - Governance: keep; do not import directly from the new backend; integration only via legacy paths.

Frontend legacy (Gradio)
- javascript/ (Category: Compat; Risk: High)
  - Purpose: legacy JS injected by Gradio; entry `javascript/codex.ui.app.mjs` and helpers.
  - Used by: `modules/ui_gradio_extensions.py`, other `modules/ui_*.py`.

- javascript-src/ (Category: Dev; Risk: Low)
  - Purpose: TS sources and ambient shims; optional type‑checking.
  - Used by: developer tooling.

- html/ (Category: Compat; Risk: Medium)
  - Purpose: static assets (e.g., placeholders used pelo legacy UI).

Vendors and curated assets
- packages_3rdparty/ (Category: Vendor; Risk: Medium)
  - gguf/: re‑export shim for GGUF; kept for compatibility.
  - comfyui_lora_collection/, webui_lora_collection/: coleções de exemplo (não essenciais).
  - Governance: keep gguf; collections são opcionais.

Presets & external repos
- configs/ (Category: Config; Risk: Low)
  - presets/{sd15,sdxl,flux,hunyuan_video,wan_ti2v_5b} — presets por família.
  - Note: fonte de verdade para UI é `apps/interface/blocks.json` + presets do backend; alinhar em próxima migração.

- repositories/ (Category: Data; Risk: Low)
  - Purpose: repositórios auxiliares clonados (HF/local) para tooling/ensure.

Dev tools & scripts
- scripts/ (Category: Dev; Risk: Low)
  - A1111 helpers (sd_upscale.py, xyz_grid.py, postprocessing_*), utilitários dev (smoke_infer.sh, pyright‑report.sh), e inspeções WAN (wan_gguf_*).
  - Not used by backend directly; safe to keep as dev utilities.

Data, caches, artifacts
- models/ (Category: Data; Risk: Do‑not)
  - Local weights (untracked). Keep.

- cache/ (Category: Data; Risk: Low) & artifacts/ (Category: Data; Risk: Low)
  - Execution caches / build artifacts (untracked). Keep.

References only
- legacy/ (Category: Reference; Risk: Low)
  - Snapshot de código A1111/Forge para consulta. Read‑only.

- .refs/ (Category: Reference; Risk: Low)
  - ComfyUI reference tree. Read‑only.

Deprecated
- backend/ (Category: Deprecated; Risk: Low)
  - Marked as deprecated with a blocking `__init__` (raises) and README. Do not import.

- _ext_hg/ (Category: Deprecated candidate; Risk: Low)
  - Wheel/dist‑info do `huggingface_guess`. Não há import ativo pelo backend.
  - Suggestion: move to `DEPRECATED/` when convenient.

Appendix — Imports cross‑check (evidences)
- modules/ — imports from apps/server/* include initialize, sd_models, sd_samplers, sd_schedulers, call_queue, txt2img/img2img/txt2vid/img2vid; services import processing/images/options/progress.
- modules_forge/ — initialize_forge; preprocessors/control helpers referenced by extensions.
- k_diffusion/ — used transitively via modules/sd_samplers_kdiffusion.
- extensions-builtin/ — loaded dynamically by modules/extensions.py.
- javascript*/html — referenced from modules/ui_* and docs.
- packages_3rdparty/gguf — re‑export façade.

Governance
- Código ativo vive em `apps/server/backend` e `apps/interface`.
- Não importar de `legacy/`, `DEPRECATED/` ou `modules_forge.*` no backend.
- Use apenas `apps.server.backend.*` e `apps.server.backend.codex.*` no backend ativo; nenhuma façade para A1111.
- Preferir migrar presets para o backend (server‑driven) ao invés de duplicar em `configs/`.
