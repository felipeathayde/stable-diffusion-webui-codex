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
- For I2V checkpoints com canais concatenados (C=36: mask4+img16+lat16), os latentes base do VAE ocupam os ÚLTIMOS 16 canais. O decode seleciona os últimos 16.
- Logs esclarecem quando o handoff ocorre e que não há VAE nesse ponto.
