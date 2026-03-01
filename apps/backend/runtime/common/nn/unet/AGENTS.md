# UNet (Codex) — Agent Notes
Date: 2025-11-10
Last Review: 2026-03-01
Status: Active

## Purpose
- Implementação nativa do `UNet2DConditionModel` com `SpatialTransformer` e suporte a condicionamento via cross-attn (`context`) e opcional ADM (`y`).

## Key files
- `apps/backend/runtime/common/nn/unet/model.py` — `UNet2DConditionModel` implementation.
- `apps/backend/runtime/common/nn/unet/config.py` — typed `UNetConfig`.
- `apps/backend/runtime/common/nn/unet/layers.py` — core blocks (`ResBlock`, `SpatialTransformer`, up/downsample, etc.).
- `apps/backend/runtime/common/nn/unet/utils.py` — shared helper factories (`conv_nd`, `avg_pool_nd`) and embeddings.

## Invariantes
- `forward(x, timesteps, context, y, ...)`:
  - Se `num_classes` é `None` → `y` deve ser `None`.
  - Se `num_classes` não é `None` → `y` deve ser Tensor 2D compatível; erro explícito em caso de desencontro.
- Não há fallbacks.

## Logging
- DEBUG (`backend.runtime.unet`) pode registrar `context_dim` efetivo e shapes de entrada.

## Notes
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `config.py`, and `utils.py` (doc-only change; part of rollout).
- 2026-01-18: `__init__.py` is a package marker (no re-exports); import UNet APIs from the defining modules.
- 2026-03-01: `model.py` now seeds forward-scoped global transformer-block progress counters in `transformer_options`, and `layers.py::SpatialTransformer.forward` emits block callbacks per inner transformer block using monotonic global 1-based indexing for the full UNet forward (strict validation for progress payload scaffolding keys).
