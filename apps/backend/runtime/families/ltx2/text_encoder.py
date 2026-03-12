"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: External Gemma3 text-encoder resolution and loading for the native LTX2 seam.
Normalizes the resolved override map into exactly one external LTX2 text-encoder asset path, then materializes the
Gemma3 model + tokenizer runtime pair from the vendored LTX2 metadata repo without importing `.refs/**` or inventing
alternate runtime key layouts.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2TextEncoderRuntime` (dataclass): Loaded Gemma3 model + tokenizer pair for LTX2 execution.
- `resolve_ltx2_text_encoder_asset` (function): Resolve exactly one external Gemma3 asset from loader override paths.
- `load_ltx2_text_encoder_runtime` (function): Load the Gemma3 model + tokenizer from the resolved external asset.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.ops.operations_gguf import is_packed_gguf_artifact
from .model import Ltx2TextEncoderAsset, Ltx2VendorPaths

_ALLOWED_TEXT_ENCODER_SUFFIXES = (".gguf", ".safetensor", ".safetensors")
logger = logging.getLogger("backend.runtime.families.ltx2.text_encoder")


@dataclass(frozen=True, slots=True)
class Ltx2TextEncoderRuntime:
    model: Any
    tokenizer: Any


def _is_gguf_quantized_tensor(tensor_obj: Any) -> bool:
    if getattr(tensor_obj, "qtype", None) is not None:
        return True
    if is_packed_gguf_artifact(tensor_obj):
        raise RuntimeError(
            "LTX2 Gemma3: packed GGUF text-encoder artifacts are not supported on the root runtime path. "
            "Provide the base `.gguf` Gemma3 weights instead."
        )
    return False


def _place_gguf_non_quant_tensors(
    module: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with torch.no_grad():
        for submodule_name, submodule_obj in module.named_modules():
            for param_name, parameter in submodule_obj.named_parameters(recurse=False):
                if parameter is None or _is_gguf_quantized_tensor(parameter):
                    continue
                if getattr(parameter, "is_meta", False):
                    raise RuntimeError(
                        "LTX2 Gemma3: unresolved meta parameter after GGUF load "
                        f"(module={submodule_name or '<root>'} name={param_name})."
                    )
                target_dtype = dtype if torch.is_floating_point(parameter) else parameter.dtype
                if parameter.device != device or parameter.dtype != target_dtype:
                    parameter.data = parameter.data.to(device=device, dtype=target_dtype)
            for buffer_name, buffer in submodule_obj.named_buffers(recurse=False):
                if buffer is None or _is_gguf_quantized_tensor(buffer):
                    continue
                if getattr(buffer, "is_meta", False):
                    raise RuntimeError(
                        "LTX2 Gemma3: unresolved meta buffer after GGUF load "
                        f"(module={submodule_name or '<root>'} name={buffer_name})."
                    )
                target_dtype = dtype if torch.is_floating_point(buffer) else buffer.dtype
                if buffer.device != device or buffer.dtype != target_dtype:
                    submodule_obj._buffers[buffer_name] = buffer.to(device=device, dtype=target_dtype)


def resolve_ltx2_text_encoder_asset(
    *,
    override_paths: Mapping[str, str],
    vendor_paths: Ltx2VendorPaths,
) -> Ltx2TextEncoderAsset:
    candidates = [
        (str(alias).strip(), os.path.expanduser(str(path).strip()))
        for alias, path in override_paths.items()
        if str(alias).strip() and str(path).strip()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "LTX2 requires exactly one resolved external Gemma3 text-encoder path; "
            f"got {len(candidates)} candidate(s)."
        )

    alias, path = candidates[0]
    suffix = Path(path).suffix.lower()
    if suffix not in _ALLOWED_TEXT_ENCODER_SUFFIXES:
        raise RuntimeError(
            "LTX2 external text encoder must be one `.gguf`, `.safetensor`, or `.safetensors` file; "
            f"got: {path}"
        )
    if not os.path.isfile(path):
        raise RuntimeError(f"LTX2 external text encoder path not found: {path}")
    if not Path(vendor_paths.tokenizer_dir).is_dir():
        raise RuntimeError(f"LTX2 tokenizer directory not found: {vendor_paths.tokenizer_dir}")

    kind = "gguf" if suffix == ".gguf" else "safetensors"
    return Ltx2TextEncoderAsset(
        alias=alias,
        path=path,
        kind=kind,
        tokenizer_dir=vendor_paths.tokenizer_dir,
    )


def load_ltx2_text_encoder_runtime(
    *,
    asset: Ltx2TextEncoderAsset,
    vendor_paths: Ltx2VendorPaths,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Ltx2TextEncoderRuntime:
    try:
        from transformers import AutoConfig, AutoTokenizer, Gemma3ForConditionalGeneration
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LTX2 Gemma3 runtime requires `transformers==4.57.3` with Gemma3 support."
        ) from exc

    text_encoder_dir = Path(vendor_paths.repo_dir) / "text_encoder"
    if not text_encoder_dir.is_dir():
        raise RuntimeError(f"LTX2 text-encoder config directory not found: {text_encoder_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            asset.tokenizer_dir,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 tokenizer load failed from '{asset.tokenizer_dir}': {exc}") from exc

    try:
        config = AutoConfig.from_pretrained(str(text_encoder_dir), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 Gemma3 config load failed from '{text_encoder_dir}': {exc}") from exc

    state_dict = load_torch_file(asset.path, device=device)
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(
            "LTX2 Gemma3 asset must resolve to a state_dict mapping; "
            f"got {type(state_dict).__name__}."
        )

    if asset.kind == "safetensors":
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                str(text_encoder_dir),
                config=config,
                state_dict=state_dict,
                local_files_only=True,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LTX2 Gemma3 SafeTensors load failed from '{asset.path}': {exc}") from exc
    elif asset.kind == "gguf":
        try:
            from transformers import modeling_utils as hf_modeling_utils
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LTX2 Gemma3 GGUF load requires transformers.modeling_utils.") from exc
        try:
            from apps.backend.runtime.ops.operations import using_codex_operations
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("LTX2 Gemma3 GGUF load requires Codex GGUF operations support.") from exc

        with using_codex_operations(weight_format="gguf", manual_cast_enabled=True, device=None, dtype=torch_dtype):
            with hf_modeling_utils.no_init_weights():
                model = Gemma3ForConditionalGeneration(config)
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"LTX2 Gemma3 GGUF load failed from '{asset.path}': {exc}") from exc
            if missing or unexpected:
                raise RuntimeError(
                    "LTX2 Gemma3 GGUF strict load failed: "
                    f"missing={len(missing)} unexpected={len(unexpected)} "
                    f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
                )
            _place_gguf_non_quant_tensors(model, device=device, dtype=torch_dtype)
    else:
        raise RuntimeError(f"LTX2 Gemma3 asset kind is unsupported: {asset.kind!r}")

    model.eval()
    logger.info(
        "[ltx2] loaded Gemma3 text encoder: alias=%s kind=%s device=%s dtype=%s",
        asset.alias,
        asset.kind,
        device,
        torch_dtype,
    )
    return Ltx2TextEncoderRuntime(model=model, tokenizer=tokenizer)
