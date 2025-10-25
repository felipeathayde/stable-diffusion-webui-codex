Assets Checklists — Minimal Required Files
Date: 2025-10-25

SD 1.x (SD15)
- Diffusers root com `model_index.json`
- tokenizer/: tokenizer.json, tokenizer_config.json, special_tokens_map.json
- text_encoder/: config.json, pesos (safetensors/bin)
- unet/ (ou transformer/): config.json, pesos
- vae/: config.json, pesos
- scheduler/: scheduler_config.json

SD 2.x (20/21)
- Como SD15, mas com CLIP‑H em `text_encoder/` (dim=1024)

SDXL (Base)
- tokenizer/
- text_encoder/ (CLIP‑L)
- text_encoder_2/ (CLIP‑G)
- unet|transformer/, vae/, scheduler/

SDXL (Refiner)
- text_encoder/ (CLIP‑G), unet|transformer/, scheduler/

FLUX
- tokenizer/ (T5), text_encoder/ (T5), transformer/, vae/, scheduler/

Chroma / Radiance
- tokenizer/, text_encoder/, transformer/, vae/, scheduler/

SVD / SV3D
- tokenizer/ se aplicável; unet temporal (in_channels=8); vae/; opcional vision encoder

Hunyuan Video / Image
- tokenizer/ + text_encoder/ (LLAMA/Qwen/ByT5 conforme variante); transformer/; vae/

WAN 2.2 (GGUF)
- High.gguf e Low.gguf
- tokenizer/: tokenizer.json + tokenizer_config.json + spiece.model
- text_encoder/: config.json + pesos
- VAE dir ou arquivo único (AutoencoderKLWan)

Observações
- Layouts que usam `transformer/` no lugar de `unet/` são aceitos; normalize no loader
- Para `.safetensors` monolítico (legacy), usar carregadores específicos que montam objetos equivalentes

