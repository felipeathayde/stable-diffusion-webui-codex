Repository Structure — Canonical Map
Date: 2025-10-25

Top‑level
- apps/ — application code (server/backend, interface frontend)
- codex/ — internal docs/specs
- tools/ — local developer tools and scripts (no runtime deps)
- models/ — local weights and caches (untracked)
- DEPRECATED/ — historical, must not be imported
- legacy/ — historical snapshot (read‑only reference)
- .refs/ — upstream reference trees (e.g., ComfyUI); do not import

Backend (apps/server/backend)
- core/ — engine interface, registry, requests, orchestrator
- engines/
  - diffusion/ — engines per model family; entry modules by task: txt2img.py, img2img.py, txt2vid.py, img2vid.py
  - util/ — adapters, attention backend selection, schedulers mapping
- runtime/
  - nn/ — model classes (UNet/Transformers/VAE/WAN/SD3/Flux/Chroma)
  - models/ — light loaders, state_dict helpers
  - ops/ — tensor ops; prefer PyTorch SDPA; no custom kernels unless missing in PyTorch
  - text_processing/ — tokenizers/encoders engines (CLIP/T5 etc.)
  - logging.py, memory/, utils.py — shared infra
  - exception_hook.py, policy.py — crash dumps + policy guard
- codex/ — API nativa do backend (ex.: initialization, main, options) que substitui antigas integrações "forge".
- huggingface/ — vendored minimal trees for Diffusers assets (no code)
- services/ — media/options/progress helpers

Interface (apps/interface)
- blocks.json (+ blocks.d/) — server‑driven UI schema (single source of truth)
- src/ — components/views; styles under src/styles/** (semantic, no utility dump)

Critical Boundaries
- engines import only via `apps.server.backend.*` façade
- runtime must not import legacy/ or DEPRECATED/
- No `wan_gguf*` anywhere in active code; WAN lives in runtime/nn/wan22.py
- Do not create engines/video/wan or backend.* (deprecated namespace)
