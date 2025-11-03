# apps/backend Overview
Date: 2025-10-28
Owner: Backend Steward Team
Last Review: 2025-10-28
Status: Active

## Purpose
- Implements the Codex-native backend: engine orchestration, runtime components, services, registries, and vendor assets.
- Provides a clean separation from legacy web UI code (no imports from `modules.*`, `backend.*`, or `.legacy/`).

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
- `codex/` — Codex-specific metadata and helpers (e.g., configuration defaults).

## Key Files
- `__init__.py` — Package marker.
- `registration.py` (under `engines/`) — Canonical engine registration entrypoint.
- `state.py`, `devices.py`, `orchestrator.py` (under `core/`) — Core primitives for backend execution.

## Notes
- Prefer adding new functionality under the structured subpackages above; avoid creating new ad-hoc directories.
- Keep Hugging Face helpers up to date—Codex builds on these mirrors rather than relying on upstream defaults.
- When retiring subpackages, relocate historical context to `.sangoi/deprecated/` and update this overview.
- 2025-11-02: Inline pipeline debugging agora usa `apps.backend.runtime.pipeline_debug`; ligue com `set_pipeline_debug(True)` ou defina `CODEX_PIPELINE_DEBUG=1` (também disponível na BIOS) para logar `entrou/saiu` na pipeline txt2img/SDXL. A pipeline de SDXL valida `torch.cuda.is_available()` e recusa execução quando o modelo está em CPU, evitando o crash em `torch_cpu.dll`.
- 2025-11-02: Memory config reads unified env flags (`CODEX_DIFFUSION_*`, `CODEX_TE_*`, `CODEX_VAE_*`); `DeviceRole` policies clamp dtype to fp32 when device=CPU, `get_torch_device()` honours the core policy, and loader fallbacks that forced TE onto CUDA were removed.
- 2025-11-02: `apps/backend/interfaces/api/run_api.py` agora invoca `apps.backend.infra.config.args.initialize(...)` com CLI/env/settings antes do bootstrap e chama `memory_management.reinitialize(...)`; se os devices não estiverem definidos, o backend aborta com erro claro.

## Operator Communication Guidelines
- Assuma expertise do operador. Não ofereça instruções "de iniciante" (ex.: "valide o par de TEs" ou "verifique se o modelo é SDXL") a menos que um verificador determinístico do backend tenha acusado incompatibilidade.
- Quando houver suspeita técnica, substitua dicas vagas por validações automáticas e erros explícitos com causa e ação. Se o parser sinalizar chaves ausentes/incompatíveis acima de um limiar, falhe com mensagem objetiva em vez de sugerir troubleshooting genérico.
- Nunca silencie erros; logs devem ser precisos e acionáveis. Evite recomendações prescritivas que desviem a análise de causa raiz.
