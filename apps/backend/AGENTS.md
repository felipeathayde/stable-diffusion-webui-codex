# apps/backend Overview
<!-- tags: backend, package, exports, lazy-imports -->
Date: 2025-10-28
Owner: Backend Steward Team
Last Review: 2026-01-18
Status: Active

## Purpose
- Implements the Codex-native backend: engine orchestration, runtime components, services, registries, and vendor assets.
- Provides a clean separation from legacy web UI code (no imports from `modules.*`, `backend.*`, or archived upstream snapshots).

## Subdirectories (Top-Level)
- `core/` — Core orchestration (engine interface, device/state management, RNG stack, request handling, parameter schemas).
- `engines/` — Model-specific engines and shared utilities (diffusion, SD, Flux, Chroma, WAN22).
- `runtime/` — Reusable runtime components (attention, adapters, text processing, memory, sampling, ops, WAN22, etc.).
- `services/` — High-level services (image/media options, progress streaming) that the API consumes.
- `use_cases/` — Task orchestrators (txt2img, img2img, txt2vid, img2vid) wiring inputs to engines and runtimes.
- `patchers/` — Runtime patching utilities (LoRA application, token merging, etc.).
- `infra/` — Cross-cutting infrastructure (accelerator detection, dynamic args/registry helpers).
- `interfaces/` — API schemas and adapters that surface backend capabilities.
- `huggingface/` — Hugging Face asset mirrors and helper code maintained for strict/offline modes (actively modified as needed).
- `gguf/` — GGUF-specific helpers and metadata.
- `video/` — Shared video tooling (e.g., interpolation helpers).
- `inventory/` — Machine-readable registries/inventories used during audits.

## Key Files
- `__init__.py` — Package facade (lazy exports for engines/runtime/text-processing; patchers/services are **not** re-exported).
- `registration.py` (under `engines/`) — Canonical engine registration entrypoint.
- `state.py`, `devices.py`, `orchestrator.py` (under `core/`) — Core primitives for backend execution.

## Notes
- Prefer adding new functionality under the structured subpackages above; avoid creating new ad-hoc directories.
- Keep Hugging Face helpers up to date—Codex builds on these mirrors rather than relying on upstream defaults.
- Import patchers/services explicitly from `apps/backend/patchers/**` and `apps/backend/services/**` (no backend-level re-export surface).
- When retiring subpackages, relocate historical context to `.sangoi/deprecated/` and update this overview.
- 2025-11-02: Inline pipeline debugging agora usa `apps.backend.runtime.diagnostics.pipeline_debug`; ligue com `set_pipeline_debug(True)` ou defina `CODEX_PIPELINE_DEBUG=1` (também disponível na BIOS) para logar `entrou/saiu` na pipeline txt2img/SDXL. A pipeline de SDXL valida `torch.cuda.is_available()` e recusa execução quando o modelo está em CPU, evitando o crash em `torch_cpu.dll`.
- 2025-11-02: Memory config reads device/dtype from persisted WebUI options (and CLI overrides); `DeviceRole` policies clamp dtype to fp32 when device=CPU, `get_torch_device()` honours the core policy, and loader fallbacks that forced TE onto CUDA were removed.
- 2025-11-02: `apps/backend/interfaces/api/run_api.py` agora invoca `apps.backend.infra.config.args.initialize(...)` com CLI/env(debug)/settings antes do bootstrap e chama `memory_management.reinitialize(...)`; se os devices não estiverem definidos, o backend faz fallback para CPU com warning acionável.
- 2025-11-03: `smart_offload` (WebUI option `codex_smart_offload` + per-job payload `smart_offload`) unloads CLIP/UNet/VAE only for their active stage, freeing VRAM between stages. txt2img/img2img pipelines honor the flag by unloading components after TE encode, UNet sampling, and VAE decode.
- 2025-11-03: Added `--debug-conditioning` (`CODEX_DEBUG_COND`) to surface SDXL conditioning diagnostics through config/launcher/TUI.
- 2025-11-03: `RuntimeMemoryConfig.swap.pin_shared_memory` now maps to `--pin-shared-memory` and gates host pinning during smart offload.
 - 2025-11-20: SDXL txt2img now recomputa cond/uncond após o parse de prompt (inclui LoRA/pesos) e antes do hires, preservando metadados width/height/crop/target. Pipeline Runner loga shapes/normas quando `--debug-conditioning`/`CODEX_DEBUG_COND` está ativo.
 - 2025-11-03: Global call tracing added. Use `--trace-debug` (or `CODEX_TRACE_DEBUG=1`) to enable a process-wide function-call trace implemented via `sys.setprofile`. Every Python function call after activation is logged at `DEBUG` by `backend.calltrace` with indentation by depth. Logging is forced to `DEBUG` on activation; disable by removing the flag/env and restarting.
- 2025-11-30: Package exports for WAN engines and text-processing are now lazy-loaded via `__getattr__`, preventing BIOS/API imports from pulling torch/Hugging Face dependencies until explicitly requested.
- 2026-01-03: Added standardized file header docstrings to `apps/backend/__init__.py` and `apps/backend/huggingface/__init__.py` (doc-only change; part of rollout).
- 2026-01-06: Engine VAE overrides (`vae_path`) now unwrap Codex VAE wrappers (`first_stage_model`) before applying state dicts; Hugging Face vendored assets rely on `.gitignore` not dropping `.txt` sidecars (e.g. `merges.txt`).

## Operator Communication Guidelines
- Assuma expertise do operador. Não ofereça instruções "de iniciante" (ex.: "valide o par de TEs" ou "verifique se o modelo é SDXL") a menos que um verificador determinístico do backend tenha acusado incompatibilidade.
- Quando houver suspeita técnica, substitua dicas vagas por validações automáticas e erros explícitos com causa e ação. Se o parser sinalizar chaves ausentes/incompatíveis acima de um limiar, falhe com mensagem objetiva em vez de sugerir troubleshooting genérico.
- Nunca silencie erros; logs devem ser precisos e acionáveis. Evite recomendações prescritivas que desviem a análise de causa raiz.
