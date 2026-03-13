# apps/backend/runtime/families/ltx2 Overview
<!-- tags: backend, runtime, families, ltx2, video, audio, gemma3 -->
Date: 2026-03-11
Last Review: 2026-03-12
Status: Active

## Purpose
- Host the native LTX2 runtime bring-up seam under `apps/**`.
- Freeze the parser-owned component contract (`transformer`, `connectors`, `vae`, `audio_vae`, `vocoder`) and the external Gemma3 text-encoder handoff while assembling the real backend-only runtime.

## Key Files
- `apps/backend/runtime/families/ltx2/config.py` — Immutable LTX2 component names, required text-encoder slot, and vendored metadata path resolution.
- `apps/backend/runtime/families/ltx2/model.py` — Typed bundle-planning dataclasses for components, external text encoder, vendored metadata, and the exact wrapped-vocoder config carried out of the real audio bundle metadata.
- `apps/backend/runtime/families/ltx2/text_encoder.py` — Strict one-asset Gemma3 resolution plus model/tokenizer loading from the vendored LTX2 metadata repo; GGUF loads use the text-only `Gemma3TextModel` path with a strict Gemma3 GGUF keyspace view under Codex GGUF ops, and `mmproj` projector files are rejected explicitly.
- `apps/backend/runtime/families/ltx2/loader.py` — Loader-side planner that turns parser output into a typed LTX2 bundle contract plus stable metadata.
- `apps/backend/runtime/families/ltx2/runtime.py` — Bundle rehydration helper, native component assembly/execution bridge, and the family-local `Ltx2RunResult` contract (`frames + audio_asset + metadata`).
- `apps/backend/runtime/families/ltx2/native/AGENTS.md` — Ownership note for the native LTX2 transformer/VAE/connector/vocoder/scheduler/pipeline modules.
- `apps/backend/runtime/families/ltx2/streaming/AGENTS.md` — Ownership note for the family-local transformer-core streaming wrapper/controller/config seam.
- `apps/backend/runtime/families/ltx2/vae.py` — Video-VAE bundle contract validation.
- `apps/backend/runtime/families/ltx2/audio.py` — Audio-VAE/vocoder bundle validation for the supported legacy raw and real 2.3 wrapped vocoder layouts, plus generated-audio WAV materialization into the shared `AudioExportAsset` contract.

## Notes
- This directory is not a compatibility shim. Do not import `.refs/**` from active code and do not add runtime key remap helpers.
- The current slice is still backend-only and unadvertised. Engine registration and canonical use-case execution are landed; semantic capability exposure remains out of scope.
- The external text encoder is exactly one `gemma3_12b` asset. Any ambiguous or missing override resolution must fail loud.
- Native execution is strict: no active-code LTX2 path may import LTX2-specific Diffusers runtime/model/pipeline classes. Keep model, scheduler, and pipeline execution inside `apps/backend/runtime/families/ltx2/**`.
- The external text-encoder seam stays text-only for the current backend lane. `text_encoder.py` accepts exactly one `gemma3_12b` asset, rejects `mmproj` projectors, and for GGUF assets must load through the strict Gemma3 text keyspace resolver plus the Codex-aware embedding shim under GGUF operations support.
- `runtime.py` must assemble through the local native package only: `load_ltx2_connectors(...)`, `load_ltx2_vocoder(...)`, and `load_strict_state_dict(...)` on transformer/video/audio VAEs. `load_ltx2_connectors(...)` owns optional `connectors.*` wrapper handling, must load the real 2.3 merged connector surface (`video_embeddings_connector.*` + `audio_embeddings_connector.*` + `text_embedding_projection.*`) without renaming keys, and must fail loud on mixed direct/wrapped surfaces. Do not reintroduce a generic `from_config(...)+load_state_dict(...)` path.
- The current core-streaming tranche is transformer-only and wrapper-backed. `runtime.py` may conditionally wrap only the native transformer when normalized `core_streaming_enabled` is true; `native/pipelines.py` owns generation-boundary reset/evict cleanup, and this slice must not add public result-metadata fields for streaming state.
- The real LTX 2.3 combined audio side asset is a wrapper bundle, not a flat raw vocoder. `audio.py` must validate the stored nested groups (`bwe_generator.*`, `mel_stft.*`, `vocoder.*`) without renaming keys, `loader.py` / `runtime.py` must carry the exact wrapped `vocoder` config from SafeTensors metadata through the bundle contract, and `native/vocoder.py` must load that wrapper layout directly instead of hardcoding BWE defaults.
- Gemma3 GGUF loads must be assembled under Codex GGUF operations support; do not bypass that context with raw HF model construction.
- The current LTX2 sampler/scheduler contract is fixed to the native FlowMatchEuler path. Accept only empty/default generic-route values (`uni-pc`/`simple`) or explicit `euler`/`simple`, report the actual effective scheduler in metadata, and fail loud on other overrides.
