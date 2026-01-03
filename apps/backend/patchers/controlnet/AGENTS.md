# AGENT — apps/backend/patchers/controlnet
Date: 2025-10-31
Owner: Backend Runtime Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Provide Codex-native ControlNet patchers organised by model family (SD, SDXL, Chroma).
- Expose clean builders for `ControlNet`, `ControlLora`, `T2IAdapter`, and advanced request helpers without relying on legacy modules.
- Bridge ControlNet modules with the runtime graph API (`ControlNode`, `ControlRequest`, `ControlComposite`).

## Components
- `base.py` – shared lifecycle helpers for control modules (hint management, weighting context, cloning).
- `weighting.py` – advanced weighting, mask application, and tensor broadcast utilities.
- `apply.py` – user-facing `apply_controlnet_advanced` that builds graph nodes with validation.
- `ops/lora.py` – LoRA-aware operations used by ControlNet LoRA builds.
- `models/sd/` – SD/SDXL-compatible implementations (`ControlNet`, `ControlLora`, `T2IAdapter`, adapter loader).
- `models/sdxl/` – SDXL aliases / future specialisations.
- `models/chroma/` – placeholder raising `NotImplementedError` until Chroma support lands.

## Notes
- All modules operate exclusivamente em imports `apps.*`; código legado em `.refs/**` é apenas referência.
- Weight schedules and masks validate against runtime transformer options and emit descriptive failures.
- Any new model family integration must document itself here and update `.sangoi/plans/controlnet-refactor.md`.
