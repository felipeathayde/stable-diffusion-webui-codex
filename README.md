# stable-diffusion-webui-codex

Codex WebUI is a FastAPI backend + Vue UI for running multiple diffusion engines (SD15/SDXL/Flux/ZImage/WAN, etc.) with explicit contracts and no silent fallbacks.

## Reference baseline
When we say “parity” or “expected behaviour”, the baseline is **Hugging Face Diffusers** (pipelines + schedulers).

## History / inspiration
This project started as a fork in the WebUI ecosystem and was heavily inspired by **AUTOMATIC1111** and the **Forge** fork. The current backend/UI are Codex-native and reimplemented in this repo’s style (no copy/paste ports).

## Documentation
- Install (Backend + Vue UI): `INSTALL.md`
- Subsystem map (single-file reference): `SUBSYSTEM-MAP.md`
- Changelog (project-facing): `.sangoi/CHANGELOG.md`
- Model assets selection/inventory: `.sangoi/reference/models/model-assets-selection-and-inventory.md`
- API tasks + SSE streaming: `.sangoi/reference/api/tasks-and-streaming.md`
- UI model tabs + QuickSettings flow: `.sangoi/reference/ui/model-tabs-and-quicksettings.md`

## License
- This repository is licensed under the PolyForm Noncommercial License 1.0.0.
- SPDX: `PolyForm-Noncommercial-1.0.0`.
- Required Notice: see `NOTICE` (must be preserved in redistributions).
- Commercial use is not permitted; see `COMMERCIAL.md`.
- Trademarks/branding: see `TRADEMARKS.md`.

## Source link (optional)
- The UI footer can show a “Source” link. Configure its target at build time using `VITE_SOURCE_URL`.
- Example: `VITE_SOURCE_URL=https://github.com/your-org/stable-diffusion-webui-codex`.

