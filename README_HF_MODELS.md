<!--
This file is the canonical `README.md` content for this Hugging Face model repository:
- https://huggingface.co/sangoi-exe/sd-webui-codex

Keep this file in sync with the Hugging Face repo `README.md`.
-->

---
tags:
  - stable-diffusion
  - stable-diffusion-xl
  - flux
  - lora
  - gguf
  - text-to-image
  - image-to-image
  - video
  - codex-webui
license: other
---

# sangoi-exe/sd-webui-codex — Official model hub for Codex WebUI

This Hugging Face repository is the **official companion model hub** for **Codex WebUI** — it exists so Codex WebUI users can download the right model files quickly, in a layout that matches the WebUI.

- **Codex WebUI (GitHub):** https://github.com/sangoi-exe/stable-diffusion-webui-codex
- **Codex WebUI license (code):** PolyForm Noncommercial License 1.0.0 (noncommercial only)
- **Codex WebUI required notice:** Copyright (c) 2025-2026 Lucas Freire Sangoi (https://github.com/sangoi-exe/stable-diffusion-webui-codex)
- **This repo is not the WebUI code.** It only stores model files to make downloads easier for users.

## What this is (and what it is not)

**This is:**
- A curated set of model files (checkpoints / LoRAs / VAEs / text encoders, depending on the folder).
- Organized to plug directly into Codex WebUI’s default `models/` layout (as configured by `apps/paths.json` in the WebUI repo).

**This is not:**
- An official upstream distribution of third-party models by their authors.
- A claim of authorship or ownership over any model here.
- Legal advice.

## Ownership, licensing, and attribution (read this)

1) **I did not create or train the models stored here.** All model names, files, and trademarks remain the property of their respective owners.
2) **Each model keeps its own license/terms.** This repository does not change those terms. Always check the original model card/license and comply with it.
3) **Codex WebUI’s license applies to the WebUI code, not the model weights.** Codex WebUI is licensed under the **PolyForm Noncommercial License 1.0.0** (noncommercial only). For commercial use, see the WebUI repository documentation.
4) **Removal / rights-holder requests:** open a **Discussion** on this Hugging Face repo (title: “Removal request”) or a **GitHub issue** on the Codex WebUI repo. Include the repo path(s) and proof of ownership/authority. (I intentionally do not publish a direct email address in this README to reduce bot scraping/spam.)
5) **No license bypass:** if an upstream model requires accepting terms (gated access), you must obtain and use it under those terms. This repo is not intended to bypass gating or restrictions.

## Getting models

Codex WebUI is the intended client for this repository.

Use Codex WebUI to discover, download, and manage models from this official hub.

This README intentionally does **not** provide command-line download instructions (Git LFS / `huggingface-cli`) to keep the supported path clear.

## How this repo integrates with Codex WebUI

Codex WebUI discovers model files by scanning configured model roots and then exposes them in the UI.

This repository is laid out to match Codex WebUI’s default model roots as defined in `apps/paths.json` (in the WebUI repo). In other words: the folder names here are part of the contract.

If you customize model roots, update `apps/paths.json` accordingly.

## Folder layout (matches Codex WebUI defaults)

These folder names are intentionally aligned with Codex WebUI’s default mapping in `apps/paths.json`. Treat the names as part of the contract (rename only if you also update the paths).

- `sd15/` — SD 1.5 checkpoints (`.safetensors`)
- `sd15-loras/` — SD 1.5 LoRAs
- `sd15-vae/` — SD 1.5 VAEs
- `sd15-tenc/` — SD 1.5 text encoders
- `sdxl/` — SDXL checkpoints (`.safetensors`)
- `sdxl-loras/` — SDXL LoRAs
- `sdxl-vae/` — SDXL VAEs
- `sdxl-tenc/` — SDXL text encoders
- `flux/` — Flux model weights (engine-dependent)
- `flux-loras/` — Flux LoRAs
- `flux-vae/` — Flux VAEs
- `flux-tenc/` — Flux text encoders
- `wan22/` — WAN 2.2 model weights (engine-dependent)
- `wan22-loras/` — WAN 2.2 LoRAs
- `wan22-vae/` — WAN 2.2 VAEs
- `wan22-tenc/` — WAN 2.2 text encoders
- `zimage/` — ZImage model weights (engine-dependent)
- `zimage-loras/` — ZImage LoRAs
- `zimage-vae/` — ZImage VAEs
- `zimage-tenc/` — ZImage text encoders

If you change the layout, update `apps/paths.json` accordingly (otherwise Codex WebUI won’t find your files).

## Integrity and safety notes

- Prefer **SafeTensors** when available. Avoid untrusted pickle-based weight formats.
- Verify downloads when possible (e.g. `sha256sum <file>`).
- Model files are large; make sure you have enough disk space and a stable connection.

## Support

For WebUI issues and usage: open an issue on the Codex WebUI GitHub repo.

For missing/corrupt files in this Hugging Face repo: open a discussion/issue here.
