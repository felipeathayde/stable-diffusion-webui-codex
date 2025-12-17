<!-- tags: backend, huggingface, wan22, diffusers, assets -->
# apps/backend/huggingface/Wan-AI Overview
Date: 2025-12-16
Owner: Runtime Maintainers
Last Review: 2025-12-16
Status: Active

## Purpose
- Vendored, lightweight Hugging Face assets (configs + tokenizers) required to run WAN2.2 in strict/offline modes without downloading metadata at runtime.
- These directories intentionally **do not** include large model weights (`*.safetensors`); only configs, indices, and tokenizer files needed by loaders/pipelines.

## Layout
- `apps/backend/huggingface/Wan-AI/Wan2.2-T2V-A14B-Diffusers/` — 14B text-to-video (Diffusers `WanPipeline`) metadata.
- `apps/backend/huggingface/Wan-AI/Wan2.2-I2V-A14B-Diffusers/` — 14B image-to-video (Diffusers `WanImageToVideoPipeline`) metadata.
- `apps/backend/huggingface/Wan-AI/Wan2.2-TI2V-5B-Diffusers/` — 5B text+image-to-video (Diffusers `WanPipeline`) metadata.
- `apps/backend/huggingface/Wan-AI/Wan2.2-Animate-14B-Diffusers/` — Wan-Animate 14B metadata (Diffusers `WanAnimatePipeline`).

## Notes
- Keep this tree aligned with `apps/backend/huggingface/assets.py` allowlists and any WAN loaders under `apps/backend/engines/wan22/**` and `apps/backend/runtime/wan22/**`.
- If a new WAN variant introduces extra component folders (e.g. `image_encoder/`, `image_processor/`), update `ensure_repo_minimal_files()` accordingly and log the change under `.sangoi/task-logs/`.
