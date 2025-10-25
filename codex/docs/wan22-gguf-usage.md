WAN22 (GGUF) — Strict Usage Guide

Summary
- WAN22 GGUF requires three explicit assets:
  - VAE weights file: `wan_vae_path` (models/VAE/*.safetensors or directory with VAE config+weights)
  - Text Encoder weights file: `wan_text_encoder_path` (models/text-encoder/*.safetensors)
  - Metadata repository root: `wan_metadata_dir` (apps/server/backend/huggingface/<org>/<repo>)
- The runtime loads tokenizer from `<wan_metadata_dir>/tokenizer` and text encoder config from `<wan_metadata_dir>/text_encoder`.
- No filesystem guessing. Missing/invalid assets throw explicit errors.

Minimal payload example
```
{
  "__strict_version": 1,
  "txt2vid_prompt": "a cat riding a bike",
  "txt2vid_neg_prompt": "blurry, lowres",
  "txt2vid_width": 768,
  "txt2vid_height": 432,
  "txt2vid_num_frames": 16,
  "txt2vid_fps": 24,
  "wan_high": {
    "model_dir": "<ABS>/models/wan22/Wan2.2-I2V-A14B-HighNoise-Q2_K.gguf",
    "sampler": "Euler",
    "scheduler": "Automatic",
    "steps": 12,
    "cfg_scale": 7
  },
  "wan_low": {
    "model_dir": "<ABS>/models/wan22/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf",
    "sampler": "Euler",
    "scheduler": "Automatic",
    "steps": 12,
    "cfg_scale": 7
  },
  "wan_metadata_dir": "<ABS>/apps/server/backend/huggingface/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
  "wan_text_encoder_path": "<ABS>/models/text-encoder/t5xxl-encoder.safetensors",
  "wan_vae_path": "<ABS>/models/VAE/wan-vae.safetensors",
  "wan_format": "gguf"
}
```

Strict rules (no fallbacks)
- Text encoder must be a weights file (`wan_text_encoder_path`). Directory-based TE is not supported.
- `wan_metadata_dir` is mandatory any time `wan_text_encoder_path` is provided.
- Tokenizer is loaded from `<wan_metadata_dir>/tokenizer` (or `tokenizer_2`). Provide `wan_tokenizer_dir` only for advanced overrides.
- VAE must be present: `wan_vae_path` points to a file or a directory compatible with `AutoencoderKLWan`.
- Device policy: CPU only when explicitly set (`device='cpu'`). Otherwise CUDA is required.
  - If `device` is omitted or set to `'auto'`/`'cuda'` and CUDA is unavailable, the backend raises a clear error. No silent CPU fallback.

Common errors (and fixes)
- "WAN22 GGUF: 'wan_text_encoder_path' (.safetensors file) is required" → select a TE weights file in models/text-encoder.
- "expected text encoder config under metadata repo: '<repo>/text_encoder'" → ensure you chose the correct metadata repo.
- "tokenizer metadata missing or invalid" → metadata repo must contain `tokenizer` (or `tokenizer_2`).
- "VAE path not found" → choose a valid VAE file/dir.

UI wiring
- Test tab and WAN tab expose input-lists for Metadata, Text Encoder (file), and VAE. They display names only and submit absolute paths.
- High/Low models are plain dropdowns listing `.gguf` files under models/wan22.
