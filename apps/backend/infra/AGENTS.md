# apps/backend/infra Overview
Date: 2025-10-28
Last Review: 2026-02-10
Status: Active

## Purpose
- Provides infrastructure utilities shared across backend modules: accelerator detection, configuration parsing, registry tooling.

## Subdirectories
- `accelerators/` — Accelerator discovery and device capability helpers.
- `config/` — Dynamic configuration loaders, CLI argument wiring, and shared config readers (e.g., `paths.json`).
- `registry/` — Shared registry utilities used by engines, services, and runtime modules.

## Notes
- Keep infrastructure logic here to avoid scattering environment/CLI handling across business logic modules.
- When adding new accelerator backends or configuration sources, update these helpers instead of embedding logic in engines.
- 2025-11-03: CLI parser exposes `--debug-conditioning`, mapping to `CODEX_DEBUG_COND` for SDXL conditioning diagnostics.
- 2025-11-03: `--pin-shared-memory` controls host pinning for offloaded models; disabled by default on Windows-heavy deployments (env alias removed; avoid settings via env).
- 2026-01-01: Added `--debug-preview-factors` to log best-fit latent→RGB preview factors for tuning `Approx cheap` live previews.
- 2025-12-06: `config/paths.py` agora garante, em best-effort, que roots relativos de modelos definidos em `apps/paths.json` (`sd15_*`, `sdxl_*`, `flux_*`, `wan22_*`) existam sob o repo root, criando diretórios ausentes apenas para entradas relativas; paths absolutos continuam dependendo de provisionamento manual.
- 2025-12-29: Repo root resolution now prefers `CODEX_ROOT` (launchers) over process CWD so configs like `apps/paths.json` and `apps/settings_values.json` stay stable across launch methods.
- 2026-01-01: Added opt-in load-time GGUF dequantization (now `--gguf-exec=dequant_upfront`) trading RAM/VRAM for speed; default remains on-the-fly (`dequant_forward`).
- 2026-01-23: Migrated GGUF execution flags to `--gguf-exec` (single canonical switch) and added `--lora-online-math` for explicit online LoRA semantics (reserved for future packed GGUF kernels).
- 2026-01-24: `config/args.py` now supports `--attention-backend` and seeds attention backend from the saved WebUI option `codex_attention_backend` when no CLI override is provided.
- 2026-01-24: Added `config/bootstrap_env.py` so backend bootstrap can publish resolved CLI/env values to env readers without mutating `os.environ`.
- 2026-01-24: Removed silent CPU fallbacks for missing device defaults; `config/args.py` now prompts in foreground TTY sessions and fails loud in non-interactive startups (requires `--core-device/--te-device/--vae-device` or persisted defaults).
- 2026-02-10: `config/args.py` strict runtime path now fails loud on unknown CLI arguments in `initialize(..., strict=True)` while keeping module-import defaults (`strict=False`) tolerant.
- 2026-01-04: Added `config/env_flags.py` as the canonical env-flag parsing helper to keep debug/feature toggle semantics consistent across runtime subsystems.
- 2026-02-05: `config/paths.py` model directory keyset now includes Anima roots (`anima_ckpt`, `anima_tenc`, `anima_vae`, `anima_loras`) so best-effort provisioning mirrors existing Flux/WAN/ZImage conventions.
- 2026-01-20: Added `--lora-apply-mode` (and `CODEX_LORA_APPLY_MODE`) as a global LoRA application switch: `merge` (default; rewrites weights once) vs `online` (apply on-the-fly during forward). Requires restarting the backend process to take effect.
- 2026-01-02: Added standardized file header docstrings to `infra/__init__.py`, `infra/accelerators/*`, `infra/config/*`, and `infra/registry/*` modules (doc-only change; part of rollout).
- 2026-02-10: Added global structural-conversion policy reader `config/weight_structural_conversion.py` (`CODEX_WEIGHT_STRUCTURAL_CONVERSION=auto|convert`) so runtime/keymap/parser seams can enforce fail-loud no-conversion behavior in `auto`.
