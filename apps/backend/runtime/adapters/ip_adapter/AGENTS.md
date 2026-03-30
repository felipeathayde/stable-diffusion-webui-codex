# apps/backend/runtime/adapters/ip_adapter Overview
Date: 2026-03-30
Last Review: 2026-03-30
Status: Active

## Purpose
- Hosts the canonical IP-Adapter runtime seam: validated asset loading, reference-image embedding prep, slot-layout resolution, and request-scoped patch apply/restore.

## Key Files
- `assets.py` — Loads/caches the validated IP-Adapter asset bundle and owns layout-family detection.
- `layout.py` — Owns the canonical slot-to-UNet coordinate order per semantic engine.
- `preprocess.py` — Builds conditional/unconditional image tokens from the selected reference image.
- `modules.py` — Defines the projector/resampler modules and the shared attn2 replace patch implementation.
- `session.py` — Applies IP-Adapter to the active denoiser clone for one sampling pass and restores the baseline objects afterward.
- `types.py` — Typed config/asset/session carriers for the runtime seam.

## Notes
- IP-Adapter slot order is not generic UNet discovery order. Keep the checkpoint slot contract in `layout.py`; do not bind slots straight from `UnetPatcher._iter_transformer_coordinates()`.
- `session.py` may validate against the generic UNet transformer inventory, but the authoritative order for slot assignment is the IP-Adapter semantic-engine layout.
- Conditional vs unconditional branch selection belongs in `modules.py::IpAdapterCrossAttentionPatch`; request prep owns token construction only.
- Base-layout unconditional tokens must come from `encode(zero_image)` through the CLIP vision runtime before projection; do not fabricate base-layout `uncond` with `zeros_like(image_embeds)`.
- Do not reintroduce adapter-local CLIP vision key rewriting or raw `nn.Module.load_state_dict(...)` paths here; image-encoder loading must stay on the canonical CLIP vision/state-dict seams.
