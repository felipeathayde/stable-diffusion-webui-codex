# UNet (Codex) — Agent Notes
Date: 2025-11-10
Owner: Runtime Maintainers
Status: Active

## Purpose
- Implementação nativa do `UNet2DConditionModel` com `SpatialTransformer` e suporte a condicionamento via cross-attn (`context`) e opcional ADM (`y`).

## Invariantes
- `forward(x, timesteps, context, y, ...)`:
  - Se `num_classes` é `None` → `y` deve ser `None`.
  - Se `num_classes` não é `None` → `y` deve ser Tensor 2D compatível; erro explícito em caso de desencontro.
- Não há fallbacks.

## Logging
- DEBUG (`backend.runtime.unet`) pode registrar `context_dim` efetivo e shapes de entrada.

