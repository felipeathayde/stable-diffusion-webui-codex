Forge Legacy Feature Inventory vs Vue UI
Date: 2025-10-22

Scope
- Source of truth for Forge snapshot: `legacy/` (read-only).
- Compared against Vue UI: `apps/interface/src` (views, stores, components).
- Method: static code scan (no runtime). References include key files for traceability.

Legend
- ✓ Supported / exposed in UI
- ~ Partial (stub or missing critical pieces)
- ✗ Missing (not exposed in current Vue)

Summary Matrix

1) Core Generation
- Txt2Img: Forge ✓ (legacy/modules/txt2img.py), Vue ✓ (`views/Txt2Img.vue`, `stores/txt2img.ts`)
- Img2Img: Forge ✓ (legacy/modules/img2img.py), Vue ✓ (`views/Img2Img.vue`, `stores/img2img.ts`)
- Inpaint: Forge ✓ (masking + img2img-inpaint; legacy/modules/masking.py, scripts/outpainting_mk_2.py), Vue ~ (`views/Inpaint.vue` placeholder; mask editor and params missing)
- Outpaint: Forge ✓ (legacy/scripts/outpainting_mk_2.py), Vue ✗ (no dedicated UI)
- PNG Info: Forge ✓, Vue ~ (`views/PngInfo.vue` stub; no backend wiring)

2) Samplers & Schedulers
- Samplers: Forge ✓ (legacy/modules/sd_samplers*.py), Vue ✓ (selector + list in stores)
- Schedulers: Forge ✓ (legacy/modules/sd_schedulers.py), Vue ✓ (selector)

3) Hi-Res / Refiner
- Hires Fix: Forge ✓ (options in `modules_forge/main_entry.py`), Vue ~ (fields exist in payload, `stores/txt2img.ts`, but disabled in UI)
- SDXL Refiner pass: Forge ✓, Vue ~ (not wired as UI controls)

4) Upscaling & Extras
- Upscale (ESRGAN/RealESRGAN/SwinIR/ScuNET): Forge ✓ (`modules/realesrgan_model.py`, `modules/esrgan_model.py`, extensions-builtin/ScuNET, SwinIR), Vue ~ (`views/Upscale.vue` stub; no API)
- Latent upscaler: Forge ✓ (`upscaler.py` + options), Vue ~ (present only as a static option)
- Face Restoration (GFPGAN/CodeFormer): Forge ✓ (`modules/gfpgan_model.py`, `modules/codeformer_model.py`), Vue ✗ (no UI)

5) Prompt & Styles
- Prompt/Negative, CFG, Steps, Seed, Batch: Forge ✓, Vue ✓ (`GenerationSettingsCard.vue` + stores)
- Styles presets: Forge ✓ (styles manager), Vue ✓ (local styles store; not yet synced to backend)

6) Extra Networks
- LoRA: Forge ✓ (`extensions-builtin/sd_forge_lora`), Vue ✓ (LoRA modal insert)
- Textual Inversion (Embeddings): Forge ✓, Vue ✓ (TI modal insert)
- Hypernetworks: Forge ✓ (legacy/modules/hypernetworks), Vue ✗ (no UI)

7) Control & Guidance
- ControlNet (multi-units + Photopea + preprocessors): Forge ✓ (`extensions-builtin/sd_forge_controlnet`, `modules_forge/supported_controlnet.py`, `supported_preprocessor.py` + `forge_preprocessor_*`), Vue ✗ (no panel)
- IP-Adapter: Forge ✓ (`extensions-builtin/sd_forge_ipadapter`), Vue ✗
- T2I-Adapter: Forge ✓ (loaded via `load_t2i_adapter`), Vue ✗

8) Advanced Modules (extensions-builtin)
- FreeU v2: Forge ✓ (`sd_forge_freeu`), Vue ✗
- Dynamic Thresholding: Forge ✓ (`sd_forge_dynamic_thresholding`), Vue ✗
- MultiDiffusion / Tiled Diffusion: Forge ✓ (`sd_forge_multidiffusion`), Vue ✗
- Soft Inpainting / Fooocus Inpaint: Forge ✓ (`soft-inpainting`, `sd_forge_fooocus_inpaint`), Vue ✗
- StyleAlign: Forge ✓ (`sd_forge_stylealign`), Vue ✗
- SAG (Self-Attn Guidance): Forge ✓ (`sd_forge_sag`), Vue ✗
- Perturbed Attention: Forge ✓ (`sd_forge_perturbed_attention`), Vue ✗
- Kohya HR Fix: Forge ✓ (`sd_forge_kohya_hrfix`), Vue ~ (hires fields present but hidden)
- Latent Modifier (mega modifier): Forge ✓ (`sd_forge_latent_modifier`), Vue ✗

9) Memory / OOM / Performance
- Never OOM: Forge ✓ (`sd_forge_neveroom` with UNet offload + VAE tiling UI), Vue ✗ (no toggle; partial generic controls for dtype/memory in Quick Settings)
- xFormers / SDP / attention opts: Forge ✓ (backend/memory_management.py), Vue ~ (not explicitly exposed; some engine/mode toggles exist)

10) Models / VAE / Text Encoders
- Checkpoint selector: Forge ✓, Vue ✓ (QuickSettings + modal)
- VAE selector: Forge ✓, Vue ✓ (QuickSettings)
- Additional text encoders: Forge ✓, Vue ✓ (QuickSettings; multiple select)

11) Utilities & Scripts
- XYZ Plot: Forge ✓ (`scripts/xyz_grid.py`), Vue ✗
- DeepDanbooru Interrogate: Forge ✓ (`modules/deepbooru.py`), Vue ✗
- Checkpoint Merger UI: Forge ✓ (`ui_checkpoint_merger.py`), Vue ✗

Detailed Pointers (non‑exhaustive)
- ControlNet core: `legacy/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py`
- Control preprocessors: `legacy/extensions-builtin/forge_preprocessor_*` and `legacy/modules_forge/supported_preprocessor.py`
- IP-Adapter: `legacy/extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py`
- Never OOM: `legacy/extensions-builtin/sd_forge_neveroom/scripts/forge_never_oom.py`
- Upscalers: `legacy/modules/realesrgan_model.py`, `legacy/modules/esrgan_model.py`, `legacy/extensions-builtin/ScuNET`, `.../SwinIR`
- Face restore: `legacy/modules/gfpgan_model.py`, `legacy/modules/codeformer_model.py`
- Samplers/Schedulers: `legacy/modules/sd_samplers*.py`, `legacy/modules/sd_schedulers.py`

Gap Analysis (highest impact first)
- ControlNet panel (units, pre/post-processors, masks, HR-fix integration): Missing
- Upscale/Extras wired to backend (RealESRGAN/SwinIR/ScuNET + face restore): Stub only
- Inpaint canvas + mask tools + inpaint params: Partial (no editor)
- Advanced modules (IP-Adapter, FreeU, MultiDiffusion, SAG, StyleAlign, Dynamic Thresholding, Kohya HR Fix): Missing
- Never OOM & memory toggles (UNet offload, VAE tiling): Missing in UI
- Utilities (XYZ Plot, Interrogate, Checkpoint Merger): Missing

Suggested Next Steps
- Phase 1: Implement ControlNet panel with 1–3 units and Canny preprocessor; add backend wiring for preprocessors list and model cache.
- Phase 2: Wire Upscale/Extras to backend (select upscaler(s), blend, tiling); expose Face Restore toggles.
- Phase 3: Inpaint canvas (mask draw/erase, blur, grow/shrink) + inpaint options; optional Photopea link-out.
- Phase 4: Add IP-Adapter and FreeU simple toggles; extend to MultiDiffusion (tiled) config.
- Phase 5: Add Never OOM toggles; surface xFormers/attention settings; add XYZ Plot as an advanced panel.

Notes
- Analysis is static (no runtime). Where UI fields exist in payload but are disabled, status marked as partial.
- Vue shows video tabs (Txt2Vid/Img2Vid) that are outside Forge legacy scope — not considered here.
