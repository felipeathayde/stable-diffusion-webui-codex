Codex Backend — Consolidated Directives
Date: 2025-10-25

Purpose
- Provide a single, authoritative document that governs how we build and evolve the backend and related code in this repository. This supersedes scattered rules and clarifies non‑negotiables.

Language
- All new documentation and updates must be written in English. Existing Portuguese docs will be translated later.

Non‑Negotiables (Policy)
- No legacy, no bridges, no shims.
  - Do not import or depend on `modules.*`, `modules_forge.*`, `legacy/`, `DEPRECATED/`, or `backend.*` (old namespace).
  - Do not create façades or “compat” layers to legacy stacks. If the backend needs functionality that exists outside `apps/`, port it natively into `apps/server/backend/**` with clear names and ownership.
  - Do not add or revive any `wan_gguf*` packages; WAN lives under `runtime/nn/wan22.py` only.
- No “minimals”. Take the time to do it right.
  - Avoid stopgaps and throwaway patches. Implement robustly, even if it takes longer.
  - If a feature isn’t ready, prefer an explicit `NotImplementedError` over a partial/faulty behavior.
- Errors must be explicit and fail fast.
  - No silent fallbacks. If a required asset or feature is missing, raise a clear exception with actionable guidance.
  - Global exception hooks write full tracebacks to `logs/exceptions-YYYYmmdd-<pid>.log` and terminate (default `CODEX_STRICT_EXIT=1`).
- PyTorch‑first directive.
  - Prefer PyTorch (and Diffusers) primitives. Use Torch SDPA attention by default.
  - Do not implement custom CUDA kernels when PyTorch already provides the functionality.

Architecture & Structure
- Canonical backend path: `apps/server/backend/**`.
- Engines are organized by task, not monoliths:
  - `engines/diffusion/{txt2img,img2img,txt2vid,img2vid}.py`
  - Engines wire into `runtime/nn/*`, `runtime/text_processing/*`, and `runtime/sampling/*`.
- Public engine utilities:
  - `engines/util/schedulers.py` provides `SamplerKind` and mapping helpers.
  - Attention backend and accelerator selection are native and configured via `CODEX_ATTENTION_BACKEND` and `CODEX_ACCELERATOR`.
- Runtime owns core math and loaders:
  - `runtime/nn/*` — model classes (UNet/Transformer/VAE/WAN/SD3/etc.)
  - `runtime/text_processing/*` — tokenization/encoding engines.
  - `runtime/sampling/*` — sampler drivers and helpers (Euler/Euler A/DDIM; DPM++ variants to follow).
  - `runtime/memory/*` — memory management and device policies.
- Core services/utilities (native only):
  - `core/state.py` — BackendState (dataclass) for progress and current image.
  - `core/devices.py` — device helpers (`default_device`, `cpu`, `torch_gc`).
  - `core/rng.py` — ImageRNG (dataclass) for deterministic noise.
- Registries (native discovery only):
  - `registry/checkpoints.py` — diffusers repos and single‑file checkpoints.
  - `registry/vae.py` — local and vendored VAE discovery.
  - `registry/text_encoders.py` — vendored encoders and tokenizers.
  - Discovery sources are explicit: local `models/` and vendored Hugging Face trees under `apps/server/backend/huggingface/`.
  - When missing, return clear errors; never “guess” silently.
- UI & Blocks
  - The single source of truth for UI blocks is `apps/interface/blocks.json` (+ overrides in `blocks.d/`).

Coding Standards
- Use dataclasses and enums for public structures (e.g., engine options, sampler kinds, registry entries).
- Clear, descriptive names for variables and functions. Avoid abbreviations unless conventional.
- Keep modules single‑purpose; avoid deep conditional trees by engine. Prefer small composable helpers.
- No catch‑all helpers or duplicated checks. Handle errors at well‑defined boundaries.
- Logging must be verbose and actionable: include what was selected (sampler, scheduler), shapes/dtypes when relevant, and next steps on errors.

Sampling Rules
- Default attention: PyTorch SDPA (enable Flash/Math/Mem as available).
- Samplers implemented natively in `runtime/sampling/driver.py`:
  - Euler (ODE), Euler A (ancestral), DDIM (η=0). DPM++ variants are being added next.
- Use sigma‑domain updates consistently; convert x0↔ε carefully and guard divisions.
- Update `BackendState` every step for progress.

Assets & Loading
- Diffusers trees must follow standard layout: `tokenizer/`, `text_encoder[/_2]/`, `unet|transformer/`, `vae/`, `scheduler/`, with `model_index.json`.
- WAN 2.2 assets: High.gguf, Low.gguf, tokenizer (tokenizer.json/tokenizer_config.json/spiece.model), text_encoder (config + weights), and VAE.
- Strict offline tokenizer mode is allowed (`--disable-online-tokenizer`), otherwise resolve tokenizers from vendored or local caches.

API & Compatibility
- Public API responses should remain compatible when feasible (e.g., fields for `/api/models`). Internals are free to change.
- If a legacy endpoint cannot be supported natively yet, return a strict error rather than a partial legacy fallback.

Git & Workflow (Hygiene)
- Commit exactly and only the files for the current task.
- Message style: `type(scope): concise summary`.
- Safety steps (recommended): create a backup branch, rebase, stage with intent, verify no conflict markers, then push.
- No destructive commands like `git clean` in scripts.

Documentation
- Update or add docs whenever behavior or configuration surfaces change.
- Place architecture and developer docs under `codex/architecture/`.
- This document is the single source of truth for backend directives.

Do / Don’t (Checklist)
- Do
  - Build native features under `apps/server/backend/**`.
  - Raise explicit errors; crash early with a full traceback when warranted.
  - Prefer SDPA and PyTorch built‑ins; design for clarity and testability.
  - Keep engines thin; push math and IO to runtime and services.
- Don’t
  - Introduce or depend on `modules.*`, `modules_forge.*`, or any bridge/shim.
  - Add “minimals” that paper over missing functionality.
  - Create engines/video/wan/* or any `wan_gguf*` packages.
  - Import from `legacy/` or `DEPRECATED/` in active code.

Operational Defaults
- Exceptions: `CODEX_STRICT_EXIT=1` (write dump then exit) — set to `0` only for controlled debugging.
- Attention backend: `CODEX_ATTENTION_BACKEND=torch-sdpa` unless override is needed.
- Accelerator: default `none`; enable TensorRT only via explicit `CODEX_ACCELERATOR=tensorrt` and available environment.

Roadmap Snapshot (Backend‑First)
- P1: Native samplers parity (add DPM++ 2M/SDE), native registries (checkpoints/VAEs/TEs), replace any NotImplemented paths with robust implementations.
- P2: Consolidate loaders/NN under `runtime/*`, unify text/tokenizers, and isolate optional extensions.
- P3: Retire legacy UI paths as the server‑driven interface reaches feature parity.

