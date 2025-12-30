# apps/backend/engines/flux Overview
Date: 2025-12-06
Owner: Engine Maintainers
Last Review: 2025-12-29
Status: Active

## Purpose
- Flux engine implementations and task wiring leveraging the Flux runtime.

## Notes
- Ensure scheduler and runtime dependencies stay in sync with `apps/backend/runtime/flux/`.
- Shared assembly helpers live in `spec.py`; extend specs there and reuse them across Flux-like engines (Flux, Chroma distilled variants) via `_build_components`.
- `assemble_flux_runtime` now accepts `engine_options` and can wrap the Flux transformer core in a `StreamedFluxCore` when `StreamingConfig` (built from engine options) decides streaming should be enabled based on VRAM state.
- Streaming is currently gated to the Flux engine (`spec.name == "flux"`) and uses devices from the memory manager (`get_torch_device` / `core_offload_device`); when disabled, the runtime is identical to the previous non-streaming path.
- `_maybe_enable_streaming_core` also unwraps an already-streamed core when streaming is disabled, so turning streaming off does not keep a stale `StreamedFluxCore` wrapper alive across reloads.
- Flux-family engines expose `EngineCapabilities` and set distilled-CFG behaviour during `load()`; keep bundle assembly side-effect free.
