WAN22 GGUF — Text Encoder Config vs. Weights (Legacy & ComfyUI)

Context
- Goal: make WAN22 GGUF use user-supplied Text Encoder (TE) weights from models/text-encoder while reading configs/tokenizers from vendored HuggingFace repos under apps/server/backend/huggingface. Avoid fallbacks; errors should be explicit and actionable.

Findings — Legacy (Forge/A1111)
- Discovery/UI
  - Scans models/VAE and models/text_encoder for files (ckpt/pt/bin/safetensors/gguf) and exposes names to the UI. Ref: legacy/modules_forge/main_entry.py (find_files_with_extensions, refresh_models).
- Decoupled weights vs. configs
  - TE weights are user-provided files. Configs/tokenizers come from vendored repos at legacy/backend/huggingface/<org>/<repo>. Ref: legacy/backend/loader.py.
  - load_huggingface_component() picks component paths (e.g., text_encoder/) from the repo layout and uses Transformers configs to build models, then loads user weights.
- Model type resolution
  - Uses huggingface_guess over the UNet state dict to pick the correct repo mapping/layout; independent from user TE path. Ref: legacy/backend/loader.py.
- Formats/quantization
  - Supports fp8/nf4/fp4/gguf via state-dict dtype detection; still reads configs from repo and loads user weights accordingly.

Findings — ComfyUI (.refs/ComfyUI)
- Discovery/UI
  - folder_paths maps text encoders to models/text_encoders (+ clip); nodes present filename lists and resolve to full paths. Ref: .refs/ComfyUI/folder_paths.py, nodes.py.
- Decoupled weights vs. configs/tokenizers
  - Tokenizers and model defs are packaged inside Comfy (e.g., comfy/text_encoders/flux.py). User selects weights by filename; configs/tokenizers are not expected in the same folder as weights.
- Error semantics
  - Missing/invalid TE is an explicit runtime error. No silent fallbacks.

Delta vs. Current Backend (before fix)
- Runtime expected a config.json adjacent to the user TE weights path, causing failures like: missing/invalid text encoder config … Should have a model_type in config.json.
- This contradicts legacy/Comfy patterns where configs/tokenizers are vendored, not co-located with user weights.

Decision & Implementation (2025-10-25)
- Strict decoupling for WAN22 GGUF:
  - TE weights: wan_text_encoder_path (file) from models/text-encoder.
  - Config/tokenizer: from wan_metadata_dir (vendored repo). Tokenizer at <repo>/tokenizer; TE config at <repo>/text_encoder.
- Runtime changes (apps/server/backend/runtime/nn/wan22.py):
  - When a TE file is provided, load AutoConfig from metadata_dir/text_encoder, instantiate UMT5/T5 encoder, and load weights from the safetensors file (strict=False). Do not read configs from the user weights directory.
  - Tokenizer resolves from metadata_dir/tokenizer (or explicit tokenizer_dir if provided). Errors are explicit.
- UI changes (apps/interface/src/views/Test.vue):
  - Sends wan_text_encoder_path (file) instead of wan_text_encoder_dir, plus wan_metadata_dir and wan_vae_path.
- API pass-through (apps/server/run_api.py):
  - Forwards wan_text_encoder_path/dir, wan_metadata_dir, wan_vae_path to request.extras.

Error Semantics (no fallbacks)
- Missing metadata_dir → “'wan_metadata_dir' is required when providing 'wan_text_encoder_path'.”
- Missing text_encoder config under metadata repo → “expected text encoder config under metadata repo: …/text_encoder”.
- Missing tokenizer under metadata repo → “tokenizer metadata missing or invalid; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.”
- Missing TE → “text encoder path missing or invalid; provide 'wan_text_encoder_path' or 'wan_text_encoder_dir'.”

Why This Matches Legacy/Comfy
- Mirrors the separation of concerns: user supplies weights; backend supplies stable configs/tokenizers from a known repo layout.
- Keeps strict, explicit errors without guessing or crawling unrelated paths.

