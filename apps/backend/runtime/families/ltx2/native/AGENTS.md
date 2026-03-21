<!-- tags: backend, runtime, families, ltx2, native, transformer, vae, audio -->
# apps/backend/runtime/families/ltx2/native Overview
Date: 2026-03-12
Last Review: 2026-03-20
Status: Active

## Purpose
- Own the native LTX2 runtime implementation under `apps/**`.
- Preserve parser-produced raw component keyspaces for transformer, connectors, video VAE, audio VAE, and vocoder loading.
- Expose the exact runtime-facing symbols consumed by `apps/backend/runtime/families/ltx2/runtime.py`.

## Key Files
- `apps/backend/runtime/families/ltx2/native/__init__.py` — Public export surface for native runtime assembly.
- `apps/backend/runtime/families/ltx2/native/connectors.py` — Native connector stack plus strict raw-layout loader for direct and wrapped connector keyspaces.
- `apps/backend/runtime/families/ltx2/native/transformer.py` — Native audiovisual transformer with strict state loading and local RoPE helpers.
- `apps/backend/runtime/families/ltx2/native/video_vae.py` — Native video autoencoder with strict raw-layout loading and pipeline-facing compression/statistics attributes.
- `apps/backend/runtime/families/ltx2/native/audio_vae.py` — Native audio autoencoder with strict raw-layout loading and mel/sample-rate attributes.
- `apps/backend/runtime/families/ltx2/native/vocoder.py` — Native legacy raw vocoder loader plus the real 2.3 wrapped `VocoderWithBWE` owner for `bwe_generator.*` + `mel_stft.*` + `vocoder.*`.
- `apps/backend/runtime/families/ltx2/native/text.py` — Prompt normalization and connector-backed text embedding helpers.
- `apps/backend/runtime/families/ltx2/native/scheduler.py` — Native FlowMatchEuler scheduler implementation.
- `apps/backend/runtime/families/ltx2/native/pipelines.py` — Native `txt2vid` / `img2vid` execution loops that return frames plus optional generated audio and own generation-boundary streaming cleanup for wrapped transformers.

## Expectations
- Keep this directory native-only. Do not import official LTX2 Diffusers runtime/model/pipeline classes here.
- Raw parser keyspaces are the contract. Do not add runtime key remap, alias maps, or compatibility shims for Diffusers-renamed weights.
- Connectors and vocoder must load through `load_ltx2_connectors(...)` / `load_ltx2_vocoder(...)`; those loaders own wrapped-vs-direct raw layout detection, must accept the supported all-wrapped `connectors.*` connector surface, and must fail loud on mixed or remapped surfaces.
- The real LTX 2.3 connector surface is `video_embeddings_connector.*` + `audio_embeddings_connector.*` + `text_embedding_projection.*`. Do not rename those keys. `native/connectors.py` must load that surface directly from the stored keys, including GGUF-packed connector tensors plus the sidecar text projection weights.
- The real LTX 2.3 vocoder side asset is a nested wrapper bundle with `bwe_generator.*`, `mel_stft.*`, and `vocoder.*` groups. Do not flatten or rename those keys. `native/vocoder.py` must load that surface through the wrapped owner using the exact `vocoder` metadata carried out of the SafeTensors audio bundle; fail loud on mixed/remapped surfaces or missing wrapper config.
- Transformer, video VAE, and audio VAE must keep `from_config(...)` plus `load_strict_state_dict(...)` as the required runtime assembly contract.
- `video_vae.py` and `audio_vae.py` must preserve the attributes consumed by `pipelines.py` (`latents_mean`, `latents_std`, compression ratios, sample/mel metadata).
- `video_vae.py::decode(...)` now owns the explicit decode-timestep contract: when `config.timestep_conditioning=True`, callers must provide a batch-matched `timestep` tensor instead of relying on positional mismatch or hidden fallback glue. `native/pipelines.py` owns the current zero-timestep default for generation.
- `pipelines.py` may rely on the local scheduler/text/native module contracts, but canonical API/result/export ownership stays outside this directory in the canonical use-cases.
- `pipelines.py` directly touches `native.transformer.config`, `native.transformer.rope`, `native.transformer.audio_rope`, and `cache_context(...)`; any streamed wrapper must proxy those surfaces honestly and cleanup must live here, not in public metadata.
