Model Pipelines “Bible” — Legacy vs ComfyUI vs Codex
Date: 2025-10-25

Scope
- What each model family requires to run (local assets layout).
- End‑to‑end pipeline steps per family (text → embeddings → sampler → decoder; video specifics where applicable).
- How Legacy (A1111+Forge snapshot) and ComfyUI organize these pieces, mapped to Codex naming: engine, runtime, nn, ops, schedulers.

Conventions
- Diffusers layout refers to `model_index.json` and subfolders (`tokenizer/`, `text_encoder[_2]/`, `transformer|unet/`, `vae/`, `scheduler/`).
- GGUF layout refers to stage files (`*.gguf`) and sidecar configs.
- “TE” = text encoder; “Tok” = tokenizer; “UNet/Transformer/DiT” = generative core; “VAE” = autoencoder; “Sched./Sampler” = noise schedule + stepper.

1) SD 1.x (aka SD15)
- Required assets
  - Checkpoint (one of):
    - Diffusers: model_index.json with subfolders: `tokenizer/` (CLIP), `text_encoder/` (CLIP-L), `unet/`, `vae/`, `scheduler/`.
    - Single file: `.safetensors` (A1111 format) + external VAE optional.
  - Optional: LoRA/ControlNet/Embeddings as per legacy extensions.
- Pipeline
  - Tokenize prompt (CLIP tokenizer → ids length 77) → encode (CLIP-L) → get text context (cross‑attn).
  - Initialize latent z ~ N(0, I) at H/8 × W/8 × 4.
  - Scheduler picks time steps (EPS or v‑pred) → iterate UNet with cross‑attn context → denoise.
  - Decode final latent via VAE → image.
- Where in Legacy
  - Tokenize/encode: `legacy/modules/sd_hijack_clip.py`, `sd1_clip` equivalents.
  - UNet/VAE loading: `legacy/modules/sd_models.py`, samplers under `legacy/modules/sd_samplers_*`.
  - UI orchestration: `legacy/modules/txt2img.py`, `img2img.py`.
- Where in ComfyUI
  - Model class + latent format: `comfy/supported_models.py: SD15`, `comfy/latent_formats.py`.
  - Text: `comfy/sd1_clip.py`.
  - Sampling core: `comfy/model_base.py` + `comfy/model_sampling.py`.

2) SD 2.x (SD20/21)
- Required assets
  - Diffusers with CLIP‑H text encoder (`text_encoder/`), SD2 tokenizer, `unet/`, `vae/`, `scheduler/`.
  - Inpainting variants: `in_channels=4` and ADM differences.
- Pipeline
  - CLIP‑H tokenize/encode → context 1024d; UNet similar to SD1 with different cfg and attention shapes; VAE decode.
- Where
  - Legacy: `legacy/modules/sd_models.py`, `sd_samplers_*` (v‑pred variants), some handling in `sd_vae_approx.py`.
  - ComfyUI: `supported_models.py: SD20, SD21Unclip*, SD21` + `text_encoders/sd2_clip.py`.

3) SDXL (base + refiner)
- Required assets (Diffusers)
  - Base: `tokenizer/` (CLIP), `text_encoder/` (CLIP‑L), `text_encoder_2/` (CLIP‑G), `unet/` (or `transformer/`), `vae/`, `scheduler/`.
  - Refiner: `text_encoder/` (CLIP‑G only), `unet/`, `scheduler/` (shares VAE from base).
- Pipeline
  - Dual‑encoder prompt: CLIP‑L and CLIP‑G; pack into conditioning dicts (ADM conditioning includes size/crop, aesthetic scores).
  - Sample base to intermediate image → optional refiner pass at higher sigma range.
  - VAE decode to image.
- Where
  - Legacy: options and ADM knobs in `legacy/modules/shared_options.py` (sdxl_*), CLIP packing in `sd_hijack_clip.py`.
  - ComfyUI: `supported_models.py: SDXL, SDXLRefiner`, `sdxl_clip.py` for tokenizers/models.

4) FLUX (BFL – FLUX.1 dev/schnell)
- Required assets (Diffusers‑style vendor tree)
  - `tokenizer/` (T5 tokenizer JSONs), `text_encoder/` (T5‑XXL style weights/config), `transformer/` (FluxTransformer2DModel), `vae/`, `scheduler/`.
- Pipeline
  - Tokenize with T5XXL (longer context than CLIP), encode to prompt_embeds.
  - Sample with flow‑matching style stepper (EDM/Flow variants in implementation), Transformer2D block stack.
  - Decode with VAE.
- Where
  - Legacy Forge implementation: `legacy/backend/nn/flux.py` integrates attention/rope/MLP; `backend/nn/chroma.py` builds on flux.
  - ComfyUI: `text_encoders/flux.py`, `supported_models.py: ModelSamplingFlux`, latent format SDXL or custom.

5) Chroma / Chroma Radiance (Luma)
- Required assets
  - Diffusers‑like vendor trees: `tokenizer/`, `text_encoder/`, `transformer/`, `vae/` for Chroma; Radiance variant has latent/radiance specifics.
- Pipeline
  - Similar to FLUX/SDXL with model‑specific conditioning knobs; sampling uses Simple/EDM depending on weights.
- Where
  - Legacy: `legacy/backend/nn/chroma.py` (built atop flux helpers).
  - ComfyUI: `comfy_extras/nodes_chroma_radiance.py` and `comfy/ldm/chroma_radiance/*`.

6) SVD / SV3D (Stable Video Diffusion family)
- Required assets (Diffusers)
  - `tokenizer/`/TE when applicable, `unet/` with temporal attention/resblocks (in_channels=8 for frame pairs/latent warps), `vae/`, sometimes CLIP vision weights (prefix indicates `open_clip.model.visual`).
- Pipeline (img2vid)
  - Encode input image to latent; build spatiotemporal latent volume; condition with (optional) CLIP‑V vision features; temporal UNet denoise; per‑frame VAE decode.
- Where
  - Legacy: limited direct code; often via extensions.
  - ComfyUI: `supported_models.py: SVD_img2vid, SV3D_*`; temporal flags in `unet_config` show required dims.

7) Hunyuan Video / Image (Tencent)
- Required assets
  - Tokenizer + TE: LLAMA/Qwen/ByT5 variants (`text_encoders/hunyuan_video.py` or `hunyuan_image.py`); `transformer/` UNet/DiT; `vae/`.
- Pipeline
  - Tokenize (LLAMA/Qwen/ByT5 as configured) → encode; video/image branches sample with temporal blocks for video; decode with VAE.
- Where
  - ComfyUI: `text_encoders/hunyuan_*`, `supported_models.py` entries and latent formats.

8) WAN 2.2 (video; txt2vid/img2vid)
- Required assets (Codex runtime; non‑Diffusers execution)
  - Two stage GGUFs: High and Low (`*.gguf` each).
  - Text context: tokenizer + encoder (T5/UMT5); either vendored tree under `apps/server/backend/huggingface/Wan-AI/.../tokenizer|text_encoder` or local dirs provided.
  - VAE directory or single file for encode/decode.
- Pipeline (Codex runtime)
  - Stage High: build text embeddings; infer token grid from patch embed; sample tokens with Euler‑style scheduler; decode tokens → frames (VAE).
  - Stage Low: seed from last High frame (img2vid: seed from init image); run shorter sampler; decode frames; export video with chosen options.
- Where
  - Legacy: only references; not to be used.
  - ComfyUI: `comfy_extras/nodes_wan.py` (various WAN nodes), latent formats `Wan21/Wan22`.
  - Codex: `apps/server/backend/runtime/nn/wan22.py` (single source of truth); engines in `engines/diffusion/wan22_*`.

—

Architecture Mapping (Legacy vs ComfyUI vs Codex)

- Engine (task orchestration, API boundary)
  - Legacy: implicit in `modules/txt2img.py`, `img2img.py`, scripts and shared state.
  - ComfyUI: graph executor (`execution.py`, `server.py`); model registry in `supported_models.py` acts like factory.
  - Codex: `apps/server/backend/engines/diffusion/*` (txt2img/img2img/txt2vid/img2vid per engine), registry in `engines/registration.py`.

- Runtime (shared infra)
  - Legacy: scattered across `legacy/modules/*` (opts/shared/state) and `backend/*` helpers.
  - ComfyUI: `comfy/model_base.py` (forward + sampling), `comfy/model_sampling.py`, `comfy/latent_formats.py`, `comfy/ops.py`, `comfy/conds.py`.
  - Codex: `apps/server/backend/runtime/*` (logging, memory, text_processing, models loader, ops, nn).

- NN (model definitions)
  - Legacy: `legacy/backend/nn/*.py` (Flux, Chroma); plus many A1111‑style hooks/hijacks.
  - ComfyUI: `comfy/ldm/*` for UNet/DiT/VAE variants and domain‑specific models.
  - Codex: `runtime/nn/*` (UNet/VAE/Flux/Chroma/SD3Transformer/etc.); WAN 2.2 lives in `runtime/nn/wan22.py`.

- Text encoders / Tokenizers
  - Legacy: CLIP hacks in `sd_hijack_clip.py`; options in `shared_options.py`.
  - ComfyUI: `comfy/text_encoders/*` (sd1/sd2/sd3, flux, hunyuan, wan, etc.).
  - Codex: `runtime/text_processing/*` (classic/CLIP and T5 engines) + vendored HF assets under `apps/server/backend/huggingface/...`.

- Schedulers / Samplers
  - Legacy: `modules/sd_samplers_*` (k-diffusion, DDIM/PLMS); image‑first focus.
  - ComfyUI: `comfy/model_sampling.py` contains discrete, EDM, Flux and special flows.
  - Codex: `engines/util/schedulers.py` exposes enum `SamplerKind`; engines map UI choices to schedulers.

Assets Layout Cheatsheet

- Diffusers (image)
  - Root contains `model_index.json`.
  - Subfolders present (one or more): `tokenizer/`, `text_encoder/`, `text_encoder_2/` (SDXL), `unet/` or `transformer/`, `vae/`, `scheduler/`.
  - Minimal files: `config.json` per subfolder; weights `*.safetensors` or `.bin`.

- Diffusers (video)
  - As above; UNet usually has temporal attention and `in_channels=8`; some need vision CLIP under `conditioner.embedders.*`.

- WAN 2.2 (GGUF)
  - Two stage files: `<High>.gguf`, `<Low>.gguf`.
  - Text: tokenizer dir with `tokenizer.json`+`tokenizer_config.json`+`spiece.model`; encoder dir with `config.json`+weights.
  - VAE: directory with `config.json` + (safetensors/bin) or single file via `.from_single_file`.

Cross‑reference Pointers
- Legacy reference snapshots live under `legacy/` (read‑only).
- ComfyUI reference under `.refs/ComfyUI/`:
  - `comfy/supported_models.py` (per‑family requirements)
  - `comfy/model_base.py`, `comfy/model_sampling.py`, `comfy/latent_formats.py`
  - `comfy/text_encoders/*.py`
  - WAN/Hunyuan nodes: `comfy_extras/nodes_wan.py`, `comfy_extras/nodes_hunyuan.py`

Notes / Gaps
- This document does not list every third‑party or niche model class in ComfyUI; focus is on SD1/2/XL, FLUX, Chroma, SVD/SV3D, Hunyuan, and WAN 2.2 as they map to our current scope.
- If your local Diffusers trees use different subfolder names (e.g., `transformer` vs `unet`), the loader should normalize; otherwise add aliases in the runtime models loader.

