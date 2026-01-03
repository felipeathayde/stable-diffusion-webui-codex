"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Workflow helpers shared across Codex generation tasks (image/video).
Re-exports prompt normalization, sampling plan helpers, init-image preparation, and video orchestration utilities.

Symbols (top-level; keep in sync; no ghosts):
- `apply_dimension_overrides` (function): Applies request-level dimension overrides to a processing config.
- `apply_prompt_context` (function): Applies prompt-context controls (e.g., clip_skip) to a processing config.
- `apply_sampling_overrides` (function): Applies request-level sampling overrides to a sampling plan.
- `apply_tiling_if_requested` (function): Enables tiling hooks on a model/processing config when requested.
- `build_prompt_context` (function): Parses prompts into a normalized `PromptContext` (extras/controls/loras).
- `build_sampling_plan` (function): Builds a `SamplingPlan` from processing config and request params.
- `collect_lora_selections` (function): Collects LoRA selections from UI/options into workflow-friendly descriptors.
- `ensure_sampler_and_rng` (function): Ensures sampler selection and RNG seeds/settings are prepared for a run.
- `execute_sampling` (function): Executes the configured sampling loop for a processing config.
- `finalize_tiling` (function): Restores/cleans up tiling hooks after sampling completes.
- `latents_to_pil` (function): Converts decoded tensors/latents into PIL images.
- `maybe_decode_for_hr` (function): Optionally decodes intermediate results for high-res workflows.
- `pil_to_tensor` (function): Converts PIL images into tensors suitable for downstream processing.
- `resolve_noise_settings` (function): Resolves `NoiseSettings` from request parameters and defaults.
- `run_before_sampling_hooks` (function): Executes pre-sampling hooks (scripts/controls) for a run.
- `run_post_sample_hooks` (function): Executes post-sampling hooks (scripts/postprocess) for a run.
- `run_process_scripts` (function): Runs script pipelines attached to the processing config.
- `prepare_init_bundle` (function): Encodes init images into an `InitImageBundle` for img2img-style workflows.
- `apply_engine_loras` (function): Applies selected LoRAs to an engine for video workflows.
- `assemble_video_metadata` (function): Builds a metadata dict for video outputs.
- `build_video_plan` (function): Normalizes request attributes into a `VideoPlan`.
- `build_video_result` (function): Assembles a `VideoResult` (frames + metadata) from a run.
- `configure_sampler` (function): Configures a video sampler/scheduler on a component given a `VideoPlan`.
- `export_video` (function): Exports a frame sequence to a video file when requested.
- `__all__` (constant): Explicit export list for the workflow facade.
"""

from .common import (
    apply_dimension_overrides,
    apply_prompt_context,
    apply_sampling_overrides,
    apply_tiling_if_requested,
    build_prompt_context,
    build_sampling_plan,
    collect_lora_selections,
    ensure_sampler_and_rng,
    execute_sampling,
    finalize_tiling,
    latents_to_pil,
    maybe_decode_for_hr,
    pil_to_tensor,
    resolve_noise_settings,
    run_before_sampling_hooks,
    run_post_sample_hooks,
    run_process_scripts,
)
from .image_init import prepare_init_bundle
from .video import (
    apply_engine_loras,
    assemble_video_metadata,
    build_video_plan,
    build_video_result,
    configure_sampler,
    export_video,
)

__all__ = [
    "apply_dimension_overrides",
    "apply_prompt_context",
    "apply_sampling_overrides",
    "apply_tiling_if_requested",
    "build_prompt_context",
    "build_sampling_plan",
    "collect_lora_selections",
    "ensure_sampler_and_rng",
    "execute_sampling",
    "finalize_tiling",
    "latents_to_pil",
    "maybe_decode_for_hr",
    "pil_to_tensor",
    "resolve_noise_settings",
    "run_before_sampling_hooks",
    "run_post_sample_hooks",
    "run_process_scripts",
    "prepare_init_bundle",
    "apply_engine_loras",
    "assemble_video_metadata",
    "build_video_plan",
    "build_video_result",
    "configure_sampler",
    "export_video",
]
