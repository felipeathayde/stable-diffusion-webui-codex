"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic LoRA loader/applier with transactional backups.
Applies patch dictionaries onto model parameters with device/dtype management, supporting GGUF and optional bitsandbytes paths.
Fails loud when CodexPack packed weights are present (v1 packed-kernel execution does not support LoRA/DoRA yet).

Symbols (top-level; keep in sync; no ghosts):
- `get_parameter_devices` (function): Captures current parameter device mapping for later restoration.
- `set_parameter_devices` (function): Restores parameters to a previously captured device mapping.
- `CodexLoraLoader` (class): High-level loader/applier that integrates mapping, device placement, and progress reporting (tqdm).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter
from apps.backend.quantization.api import quantize_numpy
from apps.backend.quantization.tensor import CodexParameter
from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

from .lora_merge import merge_lora_to_weight
from .lora_types import LoraPatchEntry

logger = logging.getLogger("backend.patchers.lora")


def get_parameter_devices(model) -> Dict[str, torch.device]:
    return {key: p.device for key, p in model.named_parameters()}


def set_parameter_devices(model, parameter_devices: Mapping[str, torch.device]) -> None:
    for key, device in parameter_devices.items():
        parameter = utils.get_attr(model, key)
        if parameter.device != device:
            parameter = utils.tensor2parameter(parameter.to(device=device))
            utils.set_attr_raw(model, key, parameter)


class CodexLoraLoader:
    """Deterministic LoRA loader with transactional backups and structured logging."""

    def __init__(self, model):
        self.model = model
        self.backup: Dict[str, torch.Tensor] = {}
        self.online_parents: List[torch.nn.Module] = []
        self.loaded_signature = ""

    @torch.inference_mode()
    def refresh(
        self,
        lora_patches: MutableMapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
        *,
        offload_device=torch.device("cpu"),
        force_refresh: bool = False,
    ) -> None:
        if lora_patches and any(
            isinstance(param, CodexPackLinearQ4KTilepackV1Parameter) for _key, param in self.model.named_parameters()
        ):
            raise RuntimeError(
                "LoRA is not supported for CodexPack packed-kernel execution in v1. "
                "Load the base GGUF (non-codexpack) or disable LoRA."
            )

        signature = self._signature(lora_patches)
        if signature == self.loaded_signature and not force_refresh:
            logger.debug("LoRA loader refresh skipped (no changes).")
            return

        grouped = self._group_patches(lora_patches)
        memory_management.manager.signal_empty_cache = True
        parameter_devices = get_parameter_devices(self.model)

        self._restore_backups(parameter_devices)

        offline_groups = sum(1 for (_, online) in grouped.keys() if not online)
        logger.info(
            "Refreshing LoRA patches: groups=%d offline=%d online=%d",
            len(grouped),
            offline_groups,
            len(grouped) - offline_groups,
        )

        offline_total = sum(len(patches) for (key, online), patches in grouped.items() if not online and patches)
        progress = tqdm(total=offline_total, desc="lora merge", unit="patch") if offline_total else None

        try:
            for (param_key, online_mode), entries in grouped.items():
                if not entries:
                    continue
                if online_mode:
                    self._register_online(param_key, entries)
                    continue

                parent_layer, child_key, parameter = utils.get_attr_with_parent(self.model, param_key)
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(f"LoRA target {param_key} is not a torch.nn.Parameter.")
                if isinstance(parameter, CodexPackLinearQ4KTilepackV1Parameter):
                    raise RuntimeError(
                        "LoRA is not supported for CodexPack packed linear weights in v1. "
                        f"target={param_key!r}."
                    )

                if param_key not in self.backup:
                    if isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                        self.backup[param_key] = parameter.copy_with_data(parameter.data.detach().to(device=offload_device).clone())
                    else:
                        self.backup[param_key] = parameter.detach().to(device=offload_device).clone()

                bnb_layer = None
                gguf_parameter = None
                tensor = parameter

                if hasattr(parameter, "bnb_quantized"):
                    bnb_layer = parent_layer
                    tensor = self._dequantize_bnb(parameter)
                elif isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                    gguf_parameter = parameter
                    tensor = dequantize_tensor(parameter)
                else:
                    tensor = parameter.data

                try:
                    merged = merge_lora_to_weight(
                        entries,
                        tensor,
                        key=param_key,
                        computation_dtype=torch.float32,
                    )
                except RuntimeError as err:
                    if "out of memory" not in str(err).lower():
                        raise
                    logger.warning("LoRA merge OOM on %s; offloading to %s and retrying", param_key, offload_device)
                    self._offload_model(parameter_devices, offload_device)
                    memory_management.manager.soft_empty_cache()
                    merged = merge_lora_to_weight(
                        entries,
                        tensor,
                        key=param_key,
                        computation_dtype=torch.float32,
                    )

                if gguf_parameter is None:
                    merged = merged.to(dtype=parameter.dtype, device=parameter.device)

                if bnb_layer is not None:
                    bnb_layer.reload_weight(merged)
                elif gguf_parameter is not None:
                    # Re-quantize offline-merged weights back into GGUF packed storage.
                    # We do this explicitly (no implicit dtype casts): storage stays byte-packed.
                    qtype = gguf_parameter.qtype
                    if qtype is None:
                        raise RuntimeError(f"Unexpected GGUF parameter without qtype: {param_key}")

                    packed = quantize_numpy(merged.detach().cpu().numpy(), qtype)
                    restored = CodexParameter(
                        packed,
                        qtype=qtype,
                        shape=tuple(merged.shape),
                        computation_dtype=gguf_parameter.computation_dtype,
                    ).to(device=parameter.device, dtype=gguf_parameter.computation_dtype)
                    utils.set_attr_raw(self.model, param_key, restored)
                else:
                    utils.set_attr_raw(self.model, param_key, torch.nn.Parameter(merged, requires_grad=False))

                if progress is not None:
                    progress.update(len(entries))
                logger.debug("Applied %d LoRA patches to %s", len(entries), param_key)

            self.loaded_signature = signature
        finally:
            if progress is not None:
                progress.close()
            set_parameter_devices(self.model, parameter_devices)

    def _restore_backups(self, parameter_devices: Mapping[str, torch.device]) -> None:
        for module in self.online_parents:
            if hasattr(module, "codex_online_loras"):
                del module.codex_online_loras
        self.online_parents.clear()

        for key, tensor in self.backup.items():
            target_device = parameter_devices.get(key, tensor.device)
            if isinstance(tensor, CodexParameter) and tensor.qtype is not None:
                restored = tensor.to(device=target_device, dtype=tensor.computation_dtype)
                utils.set_attr_raw(self.model, key, restored)
                continue
            restored = tensor.to(device=target_device).clone()
            utils.set_attr_raw(self.model, key, torch.nn.Parameter(restored, requires_grad=False))
        self.backup.clear()

    def _register_online(self, param_key: str, entries: Sequence[LoraPatchEntry]) -> None:
        parent_layer, child_key, parameter = utils.get_attr_with_parent(self.model, param_key)
        if not hasattr(parent_layer, "codex_online_loras"):
            parent_layer.codex_online_loras = {}
        parent_layer.codex_online_loras[child_key] = list(entries)
        if parent_layer not in self.online_parents:
            self.online_parents.append(parent_layer)
        logger.debug("Registered %d online LoRA patches for %s", len(entries), param_key)

    def _group_patches(
        self,
        lora_patches: Mapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]],
    ) -> Dict[Tuple[str, bool], List[LoraPatchEntry]]:
        grouped: Dict[Tuple[str, bool], List[LoraPatchEntry]] = {}
        for (filename, strength_patch, strength_model, online_mode), param_map in lora_patches.items():
            for param_key, patches in param_map.items():
                target = grouped.setdefault((param_key, online_mode), [])
                target.extend(patches)
                logger.debug(
                    "Queued %d patches for %s (file=%s strength_patch=%.3f strength_model=%.3f online=%s)",
                    len(patches),
                    param_key,
                    filename,
                    strength_patch,
                    strength_model,
                    online_mode,
                )
        return grouped

    def _signature(self, lora_patches: Mapping[Tuple[str, float, float, bool], Dict[str, List[LoraPatchEntry]]]) -> str:
        items = []
        for key in sorted(lora_patches.keys()):
            param_map = lora_patches[key]
            for param_key in sorted(param_map.keys()):
                items.append((key, param_key, len(param_map[param_key])))
        return str(items)

    def _offload_model(self, parameter_devices: Mapping[str, torch.device], offload_device: torch.device) -> None:
        for key in parameter_devices.keys():
            parameter = utils.get_attr(self.model, key)
            if isinstance(parameter, CodexParameter) and parameter.qtype is not None:
                utils.set_attr_raw(
                    self.model,
                    key,
                    parameter.to(device=offload_device, dtype=parameter.computation_dtype),
                )
                continue
            utils.set_attr_raw(
                self.model,
                key,
                torch.nn.Parameter(parameter.to(device=offload_device).clone(), requires_grad=False),
            )

    def _dequantize_bnb(self, parameter: torch.nn.Parameter) -> torch.Tensor:
        try:
            from apps.backend.runtime.ops.operations_bnb import functional_dequantize_4bit
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("bitsandbytes support requested but not available.") from exc
        return functional_dequantize_4bit(parameter)


__all__ = [
    "CodexLoraLoader",
    "get_parameter_devices",
    "set_parameter_devices",
]
