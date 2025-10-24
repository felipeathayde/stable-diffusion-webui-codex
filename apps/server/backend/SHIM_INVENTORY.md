# Backend Shim Inventory (2025-10-24)

The legacy `backend.*` namespace is currently maintained as a compatibility layer over the new
`apps.server.backend.*` façade. The table below tracks the Python modules that still live under
`backend/` and simply re-export façade implementations.

## Root shims

- `backend/__init__.py`
- `backend/args.py`
- `backend/attention.py`
- `backend/logging_utils.py`
- `backend/memory_management.py`
- `backend/operations.py` (removed; import hook redirects to `apps.server.backend.runtime.ops.operations`)
- `backend/operations_bnb.py` (removed; redirects to `apps.server.backend.runtime.ops.operations_bnb`)
- `backend/operations_gguf.py` (removed; redirects to `apps.server.backend.runtime.ops.operations_gguf`)
- `backend/shared.py` (removed; redirects to `apps.server.backend.runtime.shared`)
- `backend/state_dict.py` (removed; redirects to `apps.server.backend.runtime.models.state_dict`)
- `backend/stream.py` (removed; redirects to `apps.server.backend.runtime.memory.stream`)
- `backend/torch_trace.py` (shim to `apps.server.backend.runtime.trace`)
- `backend/utils.py` (removed; redirects to `apps.server.backend.runtime.utils`)

## Runtime helpers

- `backend/sampling/condition.py`
- `backend/sampling/sampling_function.py`
- `backend/misc/checkpoint_pickle.py`
- `backend/misc/diffusers_state_dict.py`
- `backend/misc/image_resize.py`
- `backend/misc/sub_quadratic_attention.py`
- `backend/misc/tomesd.py`

## Services

- `backend/services/__init__.py`
- `backend/services/image_service.py`
- `backend/services/media_service.py`
- `backend/services/options_service.py`
- `backend/services/progress_service.py`
- `backend/services/sampler_service.py`

## Engines

- `backend/engines/__init__.py`
- `backend/engines/base.py`
- `backend/engines/flux/engine.py`
- `backend/engines/sd15/engine.py`
- `backend/engines/sd35/engine.py`
- `backend/engines/sdxl/engine.py`
- `backend/engines/video/hunyuan/engine.py`
- `backend/engines/video/svd/engine.py`
- `backend/engines/video/wan/i2v14b_engine.py`
- `backend/engines/video/wan/t2v14b_engine.py`
- `backend/engines/video/wan/ti2v5b_engine.py`
- `backend/engines/video/wan/loader.py`
- `backend/engines/video/wan/gguf_exec.py`
- `backend/engines/video/wan/gguf_incore.py`
- `backend/engines/util/__init__.py`
- `backend/engines/util/accelerator.py`
- `backend/engines/util/adapters.py`
- `backend/engines/util/attention_backend.py`
- `backend/video/interpolation/__init__.py`

## Patchers

- `backend/patcher/base.py`
- `backend/patcher/clip.py`
- `backend/patcher/clipvision.py`
- `backend/patcher/controlnet.py`
- `backend/patcher/lora.py`
- `backend/patcher/unet.py`
- `backend/patcher/vae.py`

## Core APIs

- `backend/core/__init__.py`
- `backend/core/engine_interface.py`
- `backend/core/exceptions.py`
- `backend/core/orchestrator.py`
- `backend/core/registry.py`
- `backend/core/requests.py`

## Runtime modules / NN

- `backend/diffusion_engine/base.py`
- `backend/diffusion_engine/flux.py`
- `backend/diffusion_engine/flux_config.py`
- `backend/diffusion_engine/sd15.py`
- `backend/diffusion_engine/sd20.py`
- `backend/diffusion_engine/sd35.py`
- `backend/diffusion_engine/sdxl.py`
- `backend/diffusion_engine/txt2img.py`
- `backend/huggingface/__init__.py`
- `backend/huggingface/assets.py`
- `backend/modules/k_diffusion_extra.py`
- `backend/modules/k_model.py`
- `backend/modules/k_prediction.py`
- `backend/nn/base.py`
- `backend/nn/chroma.py`
- `backend/nn/clip.py`
- `backend/nn/flux.py`
- `backend/nn/t5.py`
- `backend/nn/vae.py`
- `backend/nn/unet.py`
- `backend/nn/cnets/cldm.py`
- `backend/nn/cnets/t2i_adapter.py`
- `backend/text_processing/classic_engine.py`
- `backend/text_processing/emphasis.py`
- `backend/text_processing/parsing.py`
- `backend/text_processing/t5_engine.py`
- `backend/text_processing/textual_inversion.py`

## Next steps

- Any new code should reference `apps.server.backend.*` directly.
- Shims above remain for compatibility; mark consumers and remove them once downstream
  integrations migrate.
- Update this document as shims are collapsed or removed.
- Import redirector installed in `backend/__init__.py` to resolve `backend.*` → façade
  even when individual shim files are removed. This reduces the need to keep
  file-per-module shims while preserving compatibility for imports.
