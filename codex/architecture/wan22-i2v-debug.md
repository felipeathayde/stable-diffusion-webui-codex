WAN22 I2V (GGUF) — Handoff & Debug Decode

Summary
- Normal flow keeps the High→Low handoff entirely in latent space; the VAE is NOT called at handoff.
- VAE is used only twice: to encode the init image at the beginning and to decode final latents at the end.
- For debugging, set `WAN_I2V_DEBUG_HI_DECODE=1` to force a VAE decode preview of High tokens before handoff.

Execution Flow (img2vid)
- VAE encode: init image → base latents (16ch), tiled for stability. Logs: `VAE encode scale=... shift=...`.
- High stage: patch-embed3d → tokens, sample → tokens.
- Handoff (latent-space): unembed tokens → video_latents (often 36ch: mask4+img16+lat16). No VAE here.
- Low stage: re-embed latent volume → tokens, sample → tokens.
- VAE decode: unembed tokens → video_latents; slice to 16 base channels; decode frames. Logs include output shape and non‑finite sanitization warnings when applicable.

Debug Preview (pre-handoff)
- Env flag: `WAN_I2V_DEBUG_HI_DECODE=1` (truthy variants: `1,true,yes,on`).
- Effect: decodes High tokens to frames before the handoff. Output is discarded; purpose is inspection only.
- Availability: works in both streaming and non‑streaming paths after 2025‑10‑26.

Notes
- Channel order is configurable via `WAN_I2V_ORDER` (default `lat_first`, matching Comfy `xc + c_concat`).
- If `lat_first`: latents(16) come first; decode picks FIRST 16; assembling 36ch uses `[lat, mask4, img16]`.
- If `lat_last`: latents come last; decode picks LAST 16; assembling 36ch uses `[mask4, img16, lat]`.
- Logs clarify when handoff occurs and that no VAE is invoked at that point.
