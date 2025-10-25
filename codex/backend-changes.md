Backend Changes (Oct 2025)
==========================

Tokenizer Loading
- Fallback to fast/auto tokenizers when `merges.txt` is absent but `tokenizer.json` exists (Transformers ≥ 4.5x).
- Silences long-seq warnings if present.

Windows Guards
- CUDA/XPU stream warmups are disabled on Windows; honor `args.cuda_stream`.

Localization
- Missing `localizations/` dir no longer breaks startup.

Extra Networks / LoRA
- Ativação nativa pendente. Não há fallback para registries do legado; quando solicitado, o backend lança erro explícito (`NotImplementedError`).

Tools Relocation
- Maintenance scripts moved to `tools/` to avoid Gradio “scripts” loader import.

No-Fallback Enforcement (2025-10-25)
- Removed implicit UNet construction fallback to CPU after CUDA OOM in `apps/server/backend/runtime/models/loader.py`. OOM now raises with a precise message (device, dtype, policy).
- Tightened WAN 2.2 Diffusers repo resolution in `apps/server/backend/engines/diffusion/wan22_common.py`: only env override (`CODEX_WAN_DIFFUSERS_REPO`) or explicit engine-key mapping. No generic guessing by variant; unresolved keys raise.
- WAN 2.2 GGUF path: engines now fetch only complementary config/tokenizer from `apps/server/backend/huggingface` and require user‑supplied weights for VAE and text encoder via request `extras` (e.g., `wan_vae_path`, `wan_text_encoder_path`). No network downloads of weights.
- Updated CLI help for `--gpu-prefer-construct` to reflect strict behavior.

Rationale: honor strict backend policy — explicit errors, no silent fallbacks; keep logs actionable.
