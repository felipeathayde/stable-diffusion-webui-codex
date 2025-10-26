WAN22 Pipelines — Legacy vs ComfyUI vs Backend (Codex)

Scope
- Compare the WAN/FLUX/Chroma pipelines from Legacy (Forge/A1111) and ComfyUI (with emphasis on Wan 2.2) against the current Codex backend WAN22 GGUF runtime.
- Capture key design choices (assets, scheduling, device/memory, conditioning, error policy) and list actionable recommendations.

Sources Reviewed
- Legacy (Forge): legacy/backend/nn/flux.py, legacy/backend/nn/chroma.py, legacy/backend/huggingface/* (vendored repos).
- ComfyUI: comfy_extras/nodes_wan.py, comfy/model_base.py, comfy/supported_models.py, comfy/text_encoders/wan.py, comfy/ldm/wan/{model.py, vae2_2.py}, .refs/wan2-2workflow.json.
- Codex backend: apps/server/backend/runtime/nn/wan22.py, engines/diffusion/wan22_{14b,5b}.py, runtime/ops/operations_gguf.py, engines/diffusion/wan22_common.py.

Summary (One‑liners)
- Legacy: configs/tokenizers vendored; user supplies TE/UNet weights; robust memory manager; rich conditioning; no silent fallbacks.
- ComfyUI: two‑stage (High/Low) GGUF; TE packaged (UMT5‑XXL) + tokenizer; per‑stage LoRA; rich conditioning nodes; KSampler‑based schedulers.
- Codex (now): two‑stage GGUF; strict TE file + metadata repo; Diffusers schedulers; strict device policy; SSE progress; minimal conditioning; LoRA not yet applied in GGUF.

Assets: Configs/Tokenizer/Weights
- Legacy: decouples configs/tokenizers (vendored) from user weights; loaders instantiate from config then load state_dict.
- ComfyUI: TE config/tokenizer packaged (umt5_config_xxl.json + SPiece). User selects TE weights; VAE implementation is local.
- Codex: matches this pattern strictly; TE must be a file (wan_text_encoder_path). Config/tokenizer come from metadata_dir/text_encoder and metadata_dir/tokenizer.

Scheduling and Time Mapping
- Legacy/ComfyUI: custom normalized time mapping (SNR/time transforms) + KSampler families.
- Codex: maps Diffusers schedulers and converts their timesteps to DiT time via time_snr_shift; logs t‑map for sanity.

Latent Grid & Patch Embedding
- Legacy/ComfyUI: 3D patch embedding with stride=patch_size; VAE compresses temporal dimension (Wan22 latent length ≈ ((T−1)//4)+1).
- Codex: infers grid by probing patch_embed3d to avoid guesswork; VAE encode/decode wrapped for I2V/T2V accordingly.

I2V Input Composition (Design)
- For Image-to-Video (I2V), WAN models commonly expect 36 input channels at the UNet patch embedding, composed as:
  - 16 VAE latent channels + 4 temporal mask channels + 16 image-feature channels.
- Text-to-Video (T2V) uses 16 latent channels.
- Codex backend: when running img2vid and the GGUF expects 36 while the selected VAE yields 16, the runtime assembles the 16+4+16 layout explicitly (strictly; other combinations error out without fallback).

Device & Memory
- Legacy: central memory manager, offload/VRAM knobs; broad compatibility.
- ComfyUI: model_management orchestrates device/precision; optimized attention wrappers.
- Codex: strict device policy (CUDA unless cpu explicitly); bake GGUF once; logs CUDA mem snapshots; no residency/offload policy yet.

Conditioning & Controls
- Legacy/ComfyUI: extensive (concat latents + masks, reference latents, CLIP‑vision features, first/last frame, fun control, etc.).
- Codex: minimal for GGUF (prompt/negative, init image seeding); no control/mask/vision features yet.

Error Handling
- All three avoid silent fallbacks; Codex enforces explicit errors for missing metadata/TE/VAE/paths and for device.

Key Differences / Gaps (Codex vs Comfy/Legacy)
- Per‑stage LoRA: Comfy applies LoRA before each stage sampler; Codex doesn’t apply LoRA in GGUF yet.
- Conditioning coverage: Comfy exposes many WAN22 nodes; Codex only seeds I2V via VAE.
- Memory residency: Codex lacks a configurable dequantization residency policy to balance VRAM vs speed.
- Variant modes: Comfy exposes WAN22_T2V/I2V/Camera/Animate; Codex runtime is a single path with fewer knobs.

Actionable Recommendations (Strict, No Fallbacks)
1) Apply per‑stage LoRA in GGUF
   - Accept lora_path/lora_weight per stage and merge into dequantized weights for targeted keys.
   - Validate shapes and report explicit errors on mismatches.

2) Add essential conditioning hooks (I2V)
   - Optional extras: concat_latent_image + concat_mask (+ index) to seed Low or condition High; strict shape checks.

3) Improve usability & visibility
   - Early prepare‑phase SSE progress (done).
   - Device policy (done): CPU only if explicitly requested; else require CUDA.
   - Suppress benign SSE close errors in the client (treat normal close as success).

4) Memory knob (non‑blocking for function)
   - Add a conservative residency toggle for dequantized weights (cache some frequent matrices with explicit logs); default off.

Current Status in Codex
- Implemented: decoupled TE/config/tokenizer; strict device; SSE progress; img2vid VAE shape fix; safe tensor cloning; engine‑agnostic streaming.
- Pending for parity/ergonomics: per‑stage LoRA; richer conditioning; optional residency policy.
