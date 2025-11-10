# apps/backend/runtime/modules Overview
Date: 2025-11-10
Owner: Runtime Maintainers
Last Review: 2025-11-10
Status: Active

## Purpose
- Thin compatibility wrappers expected by residual legacy tooling during the migration.

## Notes
- Audit and remove modules as soon as upstream consumers migrate to the native APIs.

## Invariants (KModel.apply_model)
- `c_crossattn` deve ser Tensor 3D (B,S,C); erro explícito se inválido.
- Quando `codex_config.context_dim` estiver presente:
  - Se int → `C` deve ser exatamente igual a este valor.
  - Se sequência → `C` deve pertencer ao conjunto informado.
- Se `diffusion_model.num_classes` não é `None`, `y` deve estar presente (Tensor 2D) — sem fallback.
- Se `adm_in_channels` (em `codex_config`) estiver definido (>0), `y.shape[1]` deve ser igual a este valor.

## Logging
- Logger `backend.runtime.k_model` em nível DEBUG registra shapes de `x`, `t`, `context` e `y` antes do forward do UNet.
