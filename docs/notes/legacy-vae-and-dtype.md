# Legacy WebUI – VAE & DType Handling

Date: 2025-11-02
Author: Lucas Sangoi (via Codex)

## Overview
- Summarises how the legacy A1111/Forge WebUI handles model dtypes, particularly the VAE.
- Based on inspection of `.legacy/` (backend + modules) on 2025-11-02.

## Key Points
1. **No dtype selection in the UI**
   - The legacy UI does not expose dtype selectors. Users control precision via command-line flags.
   - Flags of interest (see `.legacy/modules/cmd_args.py`):
     - `--precision {full, half, autocast}` (default: `autocast`).
     - `--no-half`, `--no-half-vae`, `--upcast-sampling`, etc.

2. **VAE dtype selection**
   - Controlled centrally by `memory_management.vae_dtype()`.
   - Flags (`.legacy/backend/args.py`):
     - `--vae-in-fp16`, `--vae-in-bf16`, `--vae-in-fp32`, plus `--vae-in-cpu`.
   - Default behaviour (`.legacy/backend/memory_management.py`):
     - `VAE_DTYPES = [torch.float32]`; function attempts to pick bf16/fp16 only if supported and no conflicting flag.
     - Logs decision at startup: `VAE dtype preferences: [...] -> ...`.
   - Additional safeguards (`.legacy/modules/shared_options.py`):
     - `auto_vae_precision` / `auto_vae_precision_bfloat16` toggle automatic fallback to fp32/bf16 when NaNs appear.

3. **“Half” parameters = FP16**
   - Wherever the code refers to “half” (e.g., modelloader), it calls `.half()` (FP16) on the model when the flag/heuristic applies.
   - Example: `.legacy/modules/modelloader.py` → `model_descriptor.model.half()` when `prefer_half=True` and the model supports it.
   - Flags like `--no-half` simply skip converting the UNet/others to FP16.

4. **UNet dtype**
   - Determined by `memory_management.unet_dtype()` with flags such as `--unet-in-fp16`, `--unet-in-bf16`, etc.
   - Defaults to FP16 when possible (prefers performance), otherwise falls back to FP32.

## Implications for Codex rebuild
- Legacy relied on CLI flags and heuristics; precision wasn’t adjustable in the UI.
- VAE default is FP32, but automatic downgrades to FP16/BF16 occur when GPUs support it or when the user requests it explicitly.
- When porting behaviour, ensure:
  - VAE defaults mirror legacy (FP32 unless heuristics/flags push FP16/BF16).
  - “Half params” toggles translate to FP16 conversions.
  - Safety toggles like auto-precision fallback remain available to avoid NaNs.

