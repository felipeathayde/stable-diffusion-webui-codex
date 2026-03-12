<!-- tags: backend, runtime, families, ltx2, native, transformer, vae, audio -->
# apps/backend/runtime/families/ltx2/native Overview
Date: 2026-03-12
Last Review: 2026-03-12
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
- `apps/backend/runtime/families/ltx2/native/vocoder.py` — Native vocoder plus strict raw-layout loader.
- `apps/backend/runtime/families/ltx2/native/text.py` — Prompt normalization and connector-backed text embedding helpers.
- `apps/backend/runtime/families/ltx2/native/scheduler.py` — Native FlowMatchEuler scheduler implementation.
- `apps/backend/runtime/families/ltx2/native/pipelines.py` — Native `txt2vid` / `img2vid` execution loops that return frames plus optional generated audio.

## Expectations
- Keep this directory native-only. Do not import official LTX2 Diffusers runtime/model/pipeline classes here.
- Raw parser keyspaces are the contract. Do not add runtime key remap, alias maps, or compatibility shims for Diffusers-renamed weights.
- Connectors and vocoder must load through `load_ltx2_connectors(...)` / `load_ltx2_vocoder(...)`; those loaders own wrapped-vs-direct raw layout detection, must accept the supported all-wrapped `connectors.*` connector surface, and must fail loud on mixed or remapped surfaces.
- Transformer, video VAE, and audio VAE must keep `from_config(...)` plus `load_strict_state_dict(...)` as the required runtime assembly contract.
- `video_vae.py` and `audio_vae.py` must preserve the attributes consumed by `pipelines.py` (`latents_mean`, `latents_std`, compression ratios, sample/mel metadata).
- `pipelines.py` may rely on the local scheduler/text/native module contracts, but canonical API/result/export ownership stays outside this directory in the canonical use-cases.
