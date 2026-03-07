<!-- tags: backend, runtime, wan22, gguf, streaming, transformer -->
# apps/backend/runtime/families/wan22 Overview
Status: Active

## Purpose
- WAN 2.2 GGUF runtime components used by WAN engines.
- Native `WanTransformer2DModel` execution, scheduler orchestration, VAE IO, and streaming support.

## Key files
- `wan22.py` — public runtime facade used by WAN engines.
- `config.py` — run/stage config parsing and device/dtype resolution.
- `run.py` — txt2vid/img2vid orchestration and streaming entrypoints.
- `sampling.py` — stage sampling loops and block-progress wiring.
- `scheduler.py` — WAN-specific scheduler helpers.
- `text_context.py` — local-files-only tokenizer/text-encoder loading for prompt embeddings.
- `stage_loader.py` — mounts base GGUF stage weights via `using_codex_operations(..., weight_format="gguf")`.
- `stage_lora.py` — stage LoRA application for base GGUF weights.
- `model.py` — transformer architecture and GGUF keyspace resolution.
- `vae_io.py` — VAE load/encode/decode helpers.
- `streaming/` — chunked/core-streaming infrastructure.

## Expectations
- Keep runtime behavior aligned with `apps/backend/engines/wan22/`.
- Base `.gguf` artifacts are the supported root-path input; unsupported packed artifacts must fail loud.
- `text_context.py` must keep tokenizer/model loading local-files-only and strict on device/key mismatches.
- Stage and VAE placement remain owned by the memory manager.
