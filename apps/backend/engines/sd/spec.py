from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Tuple

from apps.backend.infra.config.args import dynamic_args
from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.text_processing.classic_engine import ClassicTextProcessingEngine

logger = logging.getLogger("backend.engines.sd.spec")


class SDEngineConfigurationError(RuntimeError):
    """Raised when required SD engine components or configuration are missing."""


def _require_component(components: Mapping[str, object], key: str, context: str) -> object:
    try:
        component = components[key]
    except KeyError as error:
        raise SDEngineConfigurationError(f"Missing component '{key}' required for {context}.") from error
    if component is None:
        raise SDEngineConfigurationError(f"Component '{key}' is None for {context}.")
    return component


def _require_dynamic_arg(name: str) -> object:
    try:
        value = dynamic_args[name]
    except KeyError as error:
        raise SDEngineConfigurationError(f"Dynamic argument '{name}' is required for SD engines.") from error
    if value is None:
        raise SDEngineConfigurationError(f"Dynamic argument '{name}' is None.")
    return value


def _resolve_attr(obj: object, attr_path: str, *, context: str) -> object:
    current = obj
    for fragment in attr_path.split("."):
        if not hasattr(current, fragment):
            raise SDEngineConfigurationError(f"Attribute '{fragment}' missing on {context}.")
        current = getattr(current, fragment)
        if current is None:
            raise SDEngineConfigurationError(f"Attribute '{fragment}' resolved to None on {context}.")
    return current


@dataclass(frozen=True, slots=True)
class SDTextBranchSpec:
    identifier: str
    clip_attr: str
    embedding_expected_shape: int
    minimal_clip_skip: int = 1
    default_clip_skip: int = 1
    text_projection: bool = False
    return_pooled: bool = False
    final_layer_norm: bool = False
    embedding_key: str | None = None

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must be provided")
        if self.default_clip_skip < self.minimal_clip_skip:
            raise ValueError("default_clip_skip cannot be smaller than minimal_clip_skip")

    @property
    def embedding_identifier(self) -> str:
        return self.embedding_key or self.identifier


@dataclass(frozen=True, slots=True)
class SDEngineSpec:
    name: str
    clip_model_keys: Mapping[str, str]
    tokenizer_keys: Mapping[str, str]
    text_branches: Tuple[SDTextBranchSpec, ...]
    unet_key: str = "unet"
    vae_key: str = "vae"
    scheduler_key: str = "scheduler"
    embedding_dir_arg: str = "embedding_dir"
    emphasis_arg: str = "emphasis_name"

    def __post_init__(self) -> None:
        if not self.text_branches:
            raise ValueError("text_branches must not be empty")
        if set(self.clip_model_keys.keys()) != set(self.tokenizer_keys.keys()):
            raise ValueError("clip_model_keys and tokenizer_keys must have identical keys")
        branch_ids = {branch.identifier for branch in self.text_branches}
        if branch_ids != set(self.clip_model_keys.keys()):
            raise ValueError("text branch identifiers must match clip/tokenizer mapping keys")

    @property
    def branch_order(self) -> Tuple[str, ...]:
        return tuple(branch.identifier for branch in self.text_branches)

    @property
    def primary_branch(self) -> str:
        return self.text_branches[0].identifier


@dataclass(slots=True)
class SDEngineRuntime:
    clip: CLIP
    vae: VAE
    unet: UnetPatcher
    scheduler: object
    text_engines: Dict[str, ClassicTextProcessingEngine]
    branch_specs: Dict[str, SDTextBranchSpec]
    branch_order: Tuple[str, ...]

    def set_clip_skip(self, clip_skip: int) -> None:
        if not isinstance(clip_skip, int):
            raise TypeError("clip_skip must be an integer")
        for identifier, spec in self.branch_specs.items():
            if clip_skip < spec.minimal_clip_skip:
                raise ValueError(
                    f"Clip skip {clip_skip} is below minimal {spec.minimal_clip_skip} for branch '{identifier}'."
                )
            engine = self.text_engines[identifier]
            engine.clip_skip = clip_skip

    def primary_text_engine(self) -> ClassicTextProcessingEngine:
        return self.text_engines[self.branch_order[0]]

    def text_engine(self, identifier: str) -> ClassicTextProcessingEngine:
        try:
            return self.text_engines[identifier]
        except KeyError as error:
            raise KeyError(f"Unknown text branch '{identifier}'.") from error


def assemble_engine_runtime(
    spec: SDEngineSpec,
    estimated_config,
    components: Mapping[str, object],
) -> SDEngineRuntime:
    logger.debug("Assembling SD engine '%s' with branches: %s", spec.name, spec.branch_order)

    clip_model_dict: MutableMapping[str, object] = {}
    tokenizer_dict: MutableMapping[str, object] = {}

    for identifier in spec.branch_order:
        model_component_key = spec.clip_model_keys[identifier]
        tokenizer_component_key = spec.tokenizer_keys[identifier]
        clip_model_dict[identifier] = _require_component(components, model_component_key, f"{spec.name}.text_encoder")
        tokenizer_dict[identifier] = _require_component(components, tokenizer_component_key, f"{spec.name}.tokenizer")

    clip = CLIP(model_dict=clip_model_dict, tokenizer_dict=tokenizer_dict, model_config=estimated_config)
    logger.debug("CLIP patcher assembled for '%s'.", spec.name)

    vae_model = _require_component(components, spec.vae_key, f"{spec.name}.vae")
    vae = VAE(model=vae_model)
    logger.debug("VAE wrapper instantiated for '%s'.", spec.name)

    unet_model = _require_component(components, spec.unet_key, f"{spec.name}.unet")
    scheduler = _require_component(components, spec.scheduler_key, f"{spec.name}.scheduler")
    unet = UnetPatcher.from_model(model=unet_model, diffusers_scheduler=scheduler, config=estimated_config)
    logger.debug("UNet patcher created for '%s'.", spec.name)

    embedding_dir = _require_dynamic_arg(spec.embedding_dir_arg)
    emphasis_name = _require_dynamic_arg(spec.emphasis_arg)

    text_engines: Dict[str, ClassicTextProcessingEngine] = {}
    branch_specs: Dict[str, SDTextBranchSpec] = {}

    for branch in spec.text_branches:
        text_encoder = _resolve_attr(
            clip.cond_stage_model,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.cond_stage_model",
        )
        tokenizer = _resolve_attr(
            clip.tokenizer,
            branch.clip_attr,
            context=f"{spec.name}.{branch.identifier}.tokenizer",
        )

        engine = ClassicTextProcessingEngine(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            embedding_dir=embedding_dir,
            embedding_key=branch.embedding_identifier,
            embedding_expected_shape=branch.embedding_expected_shape,
            emphasis_name=emphasis_name,
            text_projection=branch.text_projection,
            minimal_clip_skip=branch.minimal_clip_skip,
            clip_skip=branch.default_clip_skip,
            return_pooled=branch.return_pooled,
            final_layer_norm=branch.final_layer_norm,
        )

        text_engines[branch.identifier] = engine
        branch_specs[branch.identifier] = branch
        logger.debug(
            "Text engine branch '%s' initialised for '%s' (clip_skip=%d).",
            branch.identifier,
            spec.name,
            branch.default_clip_skip,
        )

    return SDEngineRuntime(
        clip=clip,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        text_engines=text_engines,
        branch_specs=branch_specs,
        branch_order=spec.branch_order,
    )


__all__ = [
    "SDTextBranchSpec",
    "SDEngineSpec",
    "SDEngineRuntime",
    "SDEngineConfigurationError",
    "assemble_engine_runtime",
]
