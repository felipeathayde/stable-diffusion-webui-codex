# apps/backend/runtime/modules Overview
<!-- tags: runtime, modules, k_prediction -->
Date: 2025-11-10
Owner: Runtime Maintainers
Last Review: 2025-12-12
Status: Active

## Purpose
- Thin compatibility wrappers expected by residual legacy tooling during the migration.

## Notes
- Audit and remove modules as soon as upstream consumers migrate to the native APIs.
- 2025-11-28: `k_prediction_from_diffusers_scheduler` now preserves scheduler `sigma_data`/`prediction_type` when constructing `Prediction`, avoiding silent scaling drift for v-pred SDXL variants.

## Invariants (KModel.apply_model)
- `c_crossattn` deve ser Tensor 3D (B,S,C); erro explícito se inválido.
- Quando `codex_config.context_dim` estiver presente:
  - Se int → `C` deve ser exatamente igual a este valor.
  - Se sequência → `C` deve pertencer ao conjunto informado.
- Se `diffusion_model.num_classes` não é `None`, `y` deve estar presente (Tensor 2D) — sem fallback.
- Se `adm_in_channels` (em `codex_config`) estiver definido (>0), `y.shape[1]` deve ser igual a este valor.

## Logging
- Logger `backend.runtime.k_model` em nível DEBUG registra shapes de `x`, `t`, `context` e `y` antes do forward do UNet.
- 2025-12-12: Added opt-in deep logs for Z Image via `CODEX_ZIMAGE_DEBUG=1` / `CODEX_ZIMAGE_DEBUG_APPLY_MODEL=1` (KModel.apply_model tensor stats + forwarded extra cond keys).
