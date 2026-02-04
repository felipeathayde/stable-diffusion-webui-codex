"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Processing model dataclasses (txt2img/img2img) used by engine runtimes and orchestration.
Defines the stable “processing” parameter containers (hires/refiner + common fields) and helpers that normalize list-like inputs so
per-batch runs have consistent lengths.

Symbols (top-level; keep in sync; no ghosts):
- `_repeat_to_length` (function): Expands/truncates a sequence to a target length (used for per-batch list normalization).
- `RefinerConfig` (dataclass): Refiner stage configuration (enabled/steps/cfg/seed + model/vae overrides) with override serialization.
- `CodexHiresConfig` (dataclass): Hires configuration (target scale/steps/denoise + upscaler tile config) with override serialization.
- `CodexProcessingBase` (dataclass): Shared processing fields for image generation runs (prompt/negative/seed/steps/cfg/dims + hi-res/refiner).
- `CodexProcessingTxt2Img` (dataclass): Txt2img processing container (extends base with txt2img-specific fields).
- `CodexProcessingImg2Img` (dataclass): Img2img processing container (extends base with init image/mask/strength and related fields).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import logging
import math

from apps.backend.runtime.vision.upscalers.specs import TileConfig, default_tile_config, tile_config_from_payload

logger = logging.getLogger(__name__)


def _repeat_to_length(values: Sequence[Any], length: int, *, default: Any) -> List[Any]:
    if length <= 0:
        return []
    if not values:
        return [default for _ in range(length)]
    result = list(values)
    if len(result) >= length:
        return result[:length]
    if len(result) == 1:
        result = result * length
    else:
        factor = math.ceil(length / len(result))
        result = (result * factor)[:length]
    if len(result) < length:
        result.extend(default for _ in range(length - len(result)))
    return result[:length]


@dataclass
class RefinerConfig:
    """Configuration for a latent refiner stage."""

    enabled: bool = False
    steps: int = 0
    cfg: float = 7.0
    seed: int = -1
    model: Optional[str] = None
    vae: Optional[str] = None

    def as_override(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"enable": False}
        data: Dict[str, Any] = {
            "enable": True,
            "steps": int(self.steps),
            "cfg": float(self.cfg),
            "seed": int(self.seed),
        }
        if self.model:
            data["model"] = self.model
        if self.vae:
            data["vae"] = self.vae
        return data


@dataclass
class CodexHiresConfig:
    """Configuration for hires (second-pass) rendering."""

    enabled: bool = False
    scale: float = 1.0
    denoise: float = 0.0
    upscaler: Optional[str] = None
    tile: TileConfig = field(default_factory=default_tile_config)
    second_pass_steps: int = 0
    resize_x: int = 0
    resize_y: int = 0
    prompt: str = ""
    negative_prompt: str = ""
    cfg: float = 7.0
    distilled_cfg: float = 3.5
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    additional_modules: Tuple[str, ...] = field(default_factory=tuple)
    checkpoint_name: Optional[str] = None
    refiner: "RefinerConfig | None" = None

    def as_dict(self) -> Dict[str, Any]:
        result = {
            "enabled": self.enabled,
            "scale": self.scale,
            "denoise": self.denoise,
            "upscaler": self.upscaler,
            "tile": {
                "tile": int(self.tile.tile),
                "overlap": int(self.tile.overlap),
                "fallback_on_oom": bool(self.tile.fallback_on_oom),
                "min_tile": int(self.tile.min_tile),
            },
            "second_pass_steps": self.second_pass_steps,
            "resize_x": self.resize_x,
            "resize_y": self.resize_y,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "cfg": self.cfg,
            "distilled_cfg": self.distilled_cfg,
            "sampler_name": self.sampler_name,
            "scheduler": self.scheduler,
            "additional_modules": list(self.additional_modules),
            "checkpoint_name": self.checkpoint_name,
        }
        if self.refiner is not None:
            result["refiner"] = self.refiner.as_override()
        return result

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        self.enabled = bool(payload.get("enabled", self.enabled))
        self.scale = float(payload.get("scale", self.scale))
        self.denoise = float(payload.get("denoise", self.denoise))
        self.upscaler = payload.get("upscaler", self.upscaler)
        if "tile" in payload:
            self.tile = tile_config_from_payload(payload.get("tile"), context="hires.tile")
        self.second_pass_steps = int(payload.get("second_pass_steps", self.second_pass_steps))
        self.resize_x = int(payload.get("resize_x", self.resize_x))
        self.resize_y = int(payload.get("resize_y", self.resize_y))
        self.prompt = str(payload.get("prompt", self.prompt))
        self.negative_prompt = str(payload.get("negative_prompt", self.negative_prompt))
        self.cfg = float(payload.get("cfg", self.cfg))
        self.distilled_cfg = float(payload.get("distilled_cfg", self.distilled_cfg))
        self.sampler_name = payload.get("sampler_name", self.sampler_name)
        self.scheduler = payload.get("scheduler", self.scheduler)
        modules = payload.get("additional_modules")
        if modules is not None:
            if isinstance(modules, (list, tuple)):
                self.additional_modules = tuple(str(m) for m in modules)
            else:
                self.additional_modules = (str(modules),)
        if "checkpoint_name" in payload:
            self.checkpoint_name = payload.get("checkpoint_name")
        if "refiner" in payload and payload["refiner"] is not None:
            ref_payload = payload["refiner"]
            if isinstance(ref_payload, RefinerConfig):
                self.refiner = ref_payload
            elif isinstance(ref_payload, dict):
                self.refiner = RefinerConfig(
                    enabled=bool(ref_payload.get("enable", False)),
                    steps=int(ref_payload.get("steps", 0) or 0),
                    cfg=float(ref_payload.get("cfg", self.cfg)),
                    seed=int(ref_payload.get("seed", -1)),
                    model=str(ref_payload.get("model")) if ref_payload.get("model") else None,
                    vae=str(ref_payload.get("vae")) if ref_payload.get("vae") else None,
                )


@dataclass
class CodexProcessingBase:
    """Reusable description of a generation run.

    Unlike the legacy ``modules.processing`` classes this dataclass keeps state
    lightweight and free of side effects. Higher-level orchestration fills in
    runtime-only attributes (sampler, rng, etc.) explicitly.
    """

    prompt: str = ""
    negative_prompt: str = ""
    prompts: Sequence[str] = field(default_factory=list)
    negative_prompts: Sequence[str] = field(default_factory=list)
    styles: Sequence[str] = field(default_factory=list)
    width: int = 512
    height: int = 512
    steps: int = 20
    guidance_scale: float = 7.0
    distilled_guidance_scale: float = 3.5
    batch_size: int = 1
    iterations: int = 1
    seed: int = -1
    subseed: int = -1
    seeds: Sequence[int] = field(default_factory=list)
    subseeds: Sequence[int] = field(default_factory=list)
    subseed_strength: float = 0.0
    seed_resize_from_h: int = 0
    seed_resize_from_w: int = 0
    sampler_name: Optional[str] = None
    scheduler: Optional[str] = None
    user: str = "api"
    disable_extra_networks: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    extra_generation_params: Dict[str, Any] = field(default_factory=dict)
    override_settings: Dict[str, Any] = field(default_factory=dict)
    eta_noise_seed_delta: int = 0
    # Smart runtime flags (per-job effective settings; callers decide defaults).
    smart_offload: bool = False
    smart_fallback: bool = False
    smart_cache: bool = False

    # Runtime-assigned attributes (populated by orchestrator/use-cases)
    sd_model: Any = None
    sampler: Any = None
    rng: Any = None
    scripts: Any = None
    script_args: Sequence[Any] = field(default_factory=tuple)
    modified_noise: Any = None

    # Derived collections populated by ``prepare_prompt_data``
    all_prompts: List[str] = field(default_factory=list, init=False)
    all_negative_prompts: List[str] = field(default_factory=list, init=False)
    all_seeds: List[int] = field(default_factory=list, init=False)
    all_subseeds: List[int] = field(default_factory=list, init=False)
    prompts_prepared: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.prompts = list(self.prompts)
        self.negative_prompts = list(self.negative_prompts)
        self.styles = list(self.styles)
        self.seeds = list(self.seeds)
        self.subseeds = list(self.subseeds)

    @property
    def batch_total(self) -> int:
        return max(1, int(self.batch_size) * max(1, int(self.iterations)))

    @property
    def primary_prompt(self) -> str:
        if self.prompts:
            return self.prompts[0]
        return self.prompt

    @property
    def primary_negative_prompt(self) -> str:
        if self.negative_prompts:
            return self.negative_prompts[0]
        return self.negative_prompt

    def prepare_prompt_data(self) -> None:
        total = self.batch_total
        prompts = _repeat_to_length(self.prompts, total, default=self.prompt)
        negatives = _repeat_to_length(self.negative_prompts, total, default=self.negative_prompt)
        seeds = _repeat_to_length(self.seeds, total, default=self.seed)
        subseeds = _repeat_to_length(self.subseeds, total, default=self.subseed)
        self.all_prompts = prompts
        self.all_negative_prompts = negatives
        self.all_seeds = seeds
        self.all_subseeds = subseeds
        self.prompts_prepared = True

    def iteration_slice(self, iteration_index: int) -> slice:
        if iteration_index < 0:
            raise ValueError("iteration index must be non-negative")
        start = iteration_index * max(1, self.batch_size)
        end = start + max(1, self.batch_size)
        return slice(start, end)

    def get_prompts_for_iteration(self, iteration_index: int) -> Tuple[List[str], List[str]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        span = self.iteration_slice(iteration_index)
        return self.all_prompts[span], self.all_negative_prompts[span]

    def get_seeds_for_iteration(self, iteration_index: int) -> Tuple[List[int], List[int]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        span = self.iteration_slice(iteration_index)
        return self.all_seeds[span], self.all_subseeds[span]

    def iter_batches(self) -> Iterable[Tuple[int, str, str, int, int]]:
        if not self.prompts_prepared:
            self.prepare_prompt_data()
        for idx in range(self.batch_total):
            yield (
                idx,
                self.all_prompts[idx],
                self.all_negative_prompts[idx],
                self.all_seeds[idx],
                self.all_subseeds[idx],
            )

    def set_scripts(self, scripts: Any, script_args: Optional[Sequence[Any]] = None) -> None:
        self.scripts = scripts
        if script_args is not None:
            self.script_args = list(script_args)

    def update_override(self, key: str, value: Any) -> None:
        self.override_settings[str(key)] = value

    def update_extra_param(self, key: str, value: Any) -> None:
        self.extra_generation_params[str(key)] = value


@dataclass
class CodexProcessingTxt2Img(CodexProcessingBase):
    """Processing description for txt2img tasks."""

    hires: CodexHiresConfig = field(default_factory=CodexHiresConfig)
    refiner: "RefinerConfig | None" = None
    hires_refiner: "RefinerConfig | None" = None
    firstpass_image: Any = None
    latent_scale_mode: Optional[Dict[str, Any]] = None
    enable_hr: bool = False
    hr_upscale_to_x: int = 0
    hr_upscale_to_y: int = 0
    hr_second_pass_steps: int = 0
    hr_sampler_name: Optional[str] = None
    hr_scheduler: Optional[str] = None
    hr_checkpoint_name: Optional[str] = None
    hr_additional_modules: Sequence[str] = field(default_factory=list)
    hr_cfg: float = 7.0
    hr_distilled_cfg: float = 3.5
    hr_prompts: List[str] = field(default_factory=list, init=False)
    hr_negative_prompts: List[str] = field(default_factory=list, init=False)
    firstpass_use_distilled_cfg_scale: bool = False

    def enable_hires(self, *, cfg: CodexHiresConfig) -> None:
        self.hires = cfg
        self.enable_hr = cfg.enabled
        if cfg.enabled:
            self.update_extra_param("Hires Distilled CFG Scale", cfg.distilled_cfg)
            self.hr_second_pass_steps = cfg.second_pass_steps
            self.hr_upscale_to_x = cfg.resize_x or int(self.width * cfg.scale)
            self.hr_upscale_to_y = cfg.resize_y or int(self.height * cfg.scale)
            self.hr_sampler_name = cfg.sampler_name
            self.hr_scheduler = cfg.scheduler
            self.hr_checkpoint_name = cfg.checkpoint_name
            self.hr_additional_modules = cfg.additional_modules
            self.hr_cfg = cfg.cfg
            self.hr_distilled_cfg = cfg.distilled_cfg
            self.hires_refiner = cfg.refiner
        else:
            self.hr_second_pass_steps = 0
            self.hr_upscale_to_x = 0
            self.hr_upscale_to_y = 0
            self.hr_sampler_name = None
            self.hr_scheduler = None
            self.hr_checkpoint_name = None
            self.hr_additional_modules = []
            self.hires_refiner = None

    def ensure_hires_prompts(self) -> None:
        if not self.enable_hr:
            self.hr_prompts = []
            self.hr_negative_prompts = []
            return
        total = self.batch_total
        self.hr_prompts = _repeat_to_length([self.hires.prompt] if self.hires.prompt else [], total, default=self.primary_prompt)
        self.hr_negative_prompts = _repeat_to_length(
            [self.hires.negative_prompt] if self.hires.negative_prompt else [],
            total,
            default=self.primary_negative_prompt,
        )


@dataclass
class CodexProcessingImg2Img(CodexProcessingBase):
    """Processing description for img2img tasks."""

    hires: CodexHiresConfig = field(default_factory=CodexHiresConfig)
    init_image: Any = None
    init_images: Sequence[Any] = field(default_factory=list)
    denoising_strength: float = 0.75
    image_cfg_scale: Optional[float] = None
    mask: Any = None
    mask_blur: int = 4
    mask_blur_x: int = 4
    mask_blur_y: int = 4
    mask_round: bool = True
    inpainting_fill: int = 0
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 0
    inpainting_mask_invert: int = 0
    mask_enforcement: Optional[str] = None
    initial_noise_multiplier: Optional[float] = None
    latent_mask: Any = None
    resize_mode: int = 0
    round_image_mask: bool = True
    image_mask: Any = None

    def enable_hires(self, cfg: CodexHiresConfig) -> None:
        self.hires = cfg

    def has_mask(self) -> bool:
        return self.mask is not None or self.image_mask is not None

    def set_mask(self, mask: Any) -> None:
        self.mask = mask
        self.image_mask = mask
