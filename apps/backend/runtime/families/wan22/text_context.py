"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF text conditioning builder (tokenizer + text encoder).
Loads tokenizer metadata and text encoder weights from local paths only, applies strict embedding-key alias normalization for GGUF T5 variants,
forces forward-only GGUF dequantization for TE loads with explicit target-device routing, and then builds prompt/negative embeddings for the WAN GGUF runtime.

Symbols (top-level; keep in sync; no ghosts):
- `WAN22_DEFAULT_MAX_SEQUENCE_LENGTH` (constant): Default token length used for WAN22 prompt embeddings (aligns with Diffusers default).
- `_prompt_clean` (function): Diffusers-style prompt cleaning (optional ftfy + HTML unescape + whitespace collapse).
- `_resolve_max_sequence_length` (function): Chooses a safe tokenizer max length, clamped to `WAN22_DEFAULT_MAX_SEQUENCE_LENGTH`.
- `get_text_context` (function): Builds text conditioning/context (prompt + negative prompt) for the WAN transformer with strict fail-loud text-encoder key validation, device-aware TE weight loading, and global GGUF dequant policy alignment.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any, Optional, Tuple

import torch

from .config import as_torch_dtype, resolve_device_name
from .diagnostics import get_logger


WAN22_DEFAULT_MAX_SEQUENCE_LENGTH = 512


def _prompt_clean(text: str) -> str:
    text = str(text or "")
    try:
        import ftfy  # type: ignore

        text = ftfy.fix_text(text)
    except ModuleNotFoundError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_max_sequence_length(tok: Any) -> int:
    raw = getattr(tok, "model_max_length", None)
    try:
        raw_int = int(raw) if raw is not None else WAN22_DEFAULT_MAX_SEQUENCE_LENGTH
    except Exception:
        raw_int = WAN22_DEFAULT_MAX_SEQUENCE_LENGTH

    # Some tokenizers expose an absurd sentinel (e.g. 1e30); clamp to WAN defaults.
    max_len = min(raw_int, WAN22_DEFAULT_MAX_SEQUENCE_LENGTH)
    if max_len <= 0:
        max_len = WAN22_DEFAULT_MAX_SEQUENCE_LENGTH
    return int(max_len)


def get_text_context(
    model_dir: str,
    prompt: str,
    negative: Optional[str],
    *,
    device: str,
    dtype: str,
    text_encoder_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    vae_dir: Optional[str] = None,  # unused (kept for compatibility with existing call sites)
    model_key: Optional[str] = None,  # unused (kept for compatibility with existing call sites)
    metadata_dir: Optional[str] = None,
    logger: Any = None,
    offload_after: bool = True,
    te_device: Optional[str] = None,
    te_impl: Optional[str] = None,
    te_kernel_required: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GGUF path: use Transformers tokenizer + encoder only; do NOT fall back to Diffusers.

    - Never downloads; never calls Diffusers. If not found, raises an explicit, actionable error.
    - Tokenizer/config are read from local folders only (metadata repo or explicit dirs).
    """

    _ = (model_dir, vae_dir, model_key)  # kept for signature compatibility
    log = get_logger(logger)

    # Normalize device strings early (call sites sometimes pass 'auto').
    device = resolve_device_name(device)
    if te_device is not None:
        te_device = resolve_device_name(te_device)

    from transformers import AutoConfig, AutoTokenizer

    try:
        from transformers import UMT5EncoderModel as _Enc
    except Exception:
        from transformers import T5EncoderModel as _Enc  # type: ignore

    # Resolve tokenizer dir: prefer explicit tokenizer_dir; else infer from metadata_dir/tokenizer*
    tk_dir = tokenizer_dir
    if (not tk_dir) and metadata_dir:
        cand = os.path.join(metadata_dir, "tokenizer")
        cand2 = os.path.join(metadata_dir, "tokenizer_2")
        if os.path.isdir(cand):
            tk_dir = cand
        elif os.path.isdir(cand2):
            tk_dir = cand2

    te_path = text_encoder_dir
    te_file: Optional[str] = None
    if te_path and os.path.isfile(te_path) and te_path.lower().endswith((".safetensors", ".gguf")):
        te_file = te_path
        te_path = os.path.dirname(te_path)
    if tk_dir and os.path.isfile(tk_dir):
        tk_dir = os.path.dirname(tk_dir)

    if not tk_dir or not os.path.isdir(tk_dir):
        raise RuntimeError(
            "WAN22 GGUF: tokenizer metadata missing or invalid; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'."
        )

    try:
        tok = AutoTokenizer.from_pretrained(tk_dir, use_fast=True, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to load tokenizer from '{tk_dir}': {exc}") from exc

    max_sequence_length = _resolve_max_sequence_length(tok)
    prompt_cleaned = _prompt_clean(prompt)
    negative_cleaned = _prompt_clean(negative or "")

    log.info(
        "[wan22.gguf] tokenizer loaded: dir=%s model_max_len=%s effective_max_len=%d",
        tk_dir,
        str(getattr(tok, "model_max_length", None)),
        int(max_sequence_length),
    )

    # Effective TE preferences (extras > env > defaults)
    te_kernel_required_eff = bool(te_kernel_required) if te_kernel_required is not None else False
    te_impl_eff = (te_impl or "hf").strip().lower()
    if te_kernel_required_eff:
        te_impl_eff = "cuda_fp8"
    te_req_eff = te_impl_eff == "cuda_fp8"

    te_dev_eff = (te_device or device or "cpu").strip().lower()
    if te_dev_eff == "gpu":
        te_dev_eff = "cuda"

    # CPU TE requires fp32 (avoid implicit casts / weird numerics)
    if te_dev_eff == "cpu" and str(dtype).lower().strip() not in {"fp32", "float32"}:
        dtype = "fp32"

    log.info(
        "[wan22.gguf] text-encoder: impl=%s required=%s device=%s",
        te_impl_eff,
        str(bool(te_req_eff)).lower(),
        te_dev_eff,
    )

    # CUDA TE kernel (FP8). Required if selected; do not fallback.
    if te_impl_eff == "cuda_fp8":
        try:
            from . import wan_te_cuda as _tecuda
        except Exception as exc:
            raise RuntimeError(f"WAN22 TE CUDA kernel required but module not importable: {exc}") from exc

        if not _tecuda.available():
            last = None
            try:
                last = _tecuda.last_error()
            except Exception:
                last = None
            if last:
                raise RuntimeError(f"WAN22 TE CUDA kernel required but not available ({last}).")
            raise RuntimeError("WAN22 TE CUDA kernel required but not available. Build wan_te_cuda.")

        inputs = tok(
            [prompt_cleaned, negative_cleaned],
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]  # [2,L]
        attn_mask = inputs.get("attention_mask", None)
        log.info("[wan22.gguf] tokenized(fp8): batch=%d seqlen=%d", int(input_ids.shape[0]), int(input_ids.shape[1]))

        if not metadata_dir:
            raise RuntimeError("WAN22 GGUF: 'wan_metadata_dir' is required for TE CUDA path (need text_encoder config).")

        if not te_file:
            raise RuntimeError("WAN22 GGUF: 'wan_text_encoder_path' (.safetensors file) is required for TE CUDA path.")
        if te_file.lower().endswith(".gguf"):
            raise RuntimeError("WAN22 GGUF: TE CUDA path requires a .safetensors weights file (GGUF is unsupported).")

        enc_dir = os.path.join(metadata_dir, "text_encoder")
        cfg_hf = AutoConfig.from_pretrained(enc_dir, local_files_only=True)
        num_heads = int(getattr(cfg_hf, "num_heads", getattr(cfg_hf, "num_attention_heads", 32)))
        d_kv = int(getattr(cfg_hf, "d_kv", getattr(cfg_hf, "hidden_size", 4096) // num_heads))

        from apps.backend.runtime.families.wan22.wan_te_encoder import encode_fp8 as _encode_fp8

        dev = torch.device(te_dev_eff if te_dev_eff.startswith("cuda") and torch.cuda.is_available() else "cpu")
        if dev.type != "cuda":
            raise RuntimeError("WAN22 TE CUDA path requested but selected device is not CUDA")

        dt = as_torch_dtype(dtype)

        def _run_one(ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
            ids = ids.to(torch.long)
            return _encode_fp8(
                te_weights_path=te_file or "",
                input_ids=ids.to(dev),
                attention_mask=(mask.to(dev) if mask is not None else None),
                device=dev,
                dtype=dt,
                num_heads=num_heads,
                d_kv=d_kv,
                log_metrics=True,
            )

        p_mask = (attn_mask[0:1] if attn_mask is not None else None)
        n_mask = (attn_mask[1:2] if attn_mask is not None else None)
        p = _run_one(input_ids[0:1], p_mask)
        n = _run_one(input_ids[1:2], n_mask)
        if p_mask is not None:
            p = p * p_mask.to(dtype=p.dtype, device=p.device).unsqueeze(-1)
        if n_mask is not None:
            n = n * n_mask.to(dtype=n.dtype, device=n.device).unsqueeze(-1)
        log.info(
            "[wan22.gguf] TE(fp8) outputs: prompt=%s negative=%s dtype=%s device=%s",
            tuple(p.shape),
            tuple(n.shape),
            str(p.dtype),
            str(p.device),
        )
        if offload_after and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return p, n

    # Strict: require a TE weights file; directory-based TE loading is not supported in WAN22 GGUF.
    if te_file is None:
        raise RuntimeError(
            "WAN22 GGUF: 'wan_text_encoder_path' (.safetensors or .gguf file) is required. Directory-based text encoders are not supported."
        )

    if not metadata_dir or not os.path.isdir(metadata_dir):
        raise RuntimeError("WAN22 GGUF: 'wan_metadata_dir' is required when providing 'wan_text_encoder_path'.")
    enc_dir = os.path.join(metadata_dir, "text_encoder")
    if not os.path.isdir(enc_dir):
        raise RuntimeError(f"WAN22 GGUF: expected text encoder config under metadata repo: '{enc_dir}'")

    try:
        cfg = AutoConfig.from_pretrained(enc_dir, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to read text encoder config from '{enc_dir}': {exc}") from exc

    use_dev_name = (te_dev_eff or device or "cpu").lower().strip()
    target_device = use_dev_name if use_dev_name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    te_is_gguf = te_file.lower().endswith(".gguf")
    if te_is_gguf:
        from transformers import modeling_utils as hf_modeling_utils
        from apps.backend.runtime.ops.operations import using_codex_operations

        with using_codex_operations(
            device=torch.device(target_device),
            dtype=as_torch_dtype(dtype),
            manual_cast_enabled=True,
            weight_format="gguf",
        ):
            with hf_modeling_utils.no_init_weights():
                enc = _Enc(cfg)
    else:
        enc = _Enc(cfg)
    try:
        if te_file.lower().endswith(".safetensors"):
            from safetensors.torch import load_file as _load_st

            sd = _load_st(te_file, device=target_device)
        else:
            from apps.backend.runtime.checkpoint.io import load_gguf_state_dict

            sd = load_gguf_state_dict(
                te_file,
                dequantize=False,
                computation_dtype=as_torch_dtype(dtype),
                device=target_device,
            )
            shared_weight = sd.get("shared.weight")
            encoder_embed_weight = sd.get("encoder.embed_tokens.weight")
            if shared_weight is not None and encoder_embed_weight is None:
                sd["encoder.embed_tokens.weight"] = shared_weight
            elif encoder_embed_weight is not None and shared_weight is None:
                sd["shared.weight"] = encoder_embed_weight
            elif shared_weight is not None and encoder_embed_weight is not None:
                shared_shape = tuple(int(dim) for dim in getattr(shared_weight, "shape", ()))
                embed_shape = tuple(int(dim) for dim in getattr(encoder_embed_weight, "shape", ()))
                if shared_shape != embed_shape:
                    raise RuntimeError(
                        "WAN22 GGUF: text encoder embedding alias shape mismatch "
                        f"(shared.weight={shared_shape} encoder.embed_tokens.weight={embed_shape})."
                    )
        missing, unexpected = enc.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "WAN22 GGUF: text encoder strict load failed: "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
            )
    except Exception as exc:
        raise RuntimeError(f"WAN22 GGUF: failed to load text encoder weights '{te_file}': {exc}") from exc

    dev = torch.device(use_dev_name if use_dev_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    if not te_is_gguf:
        try:
            enc = enc.to(device=dev, dtype=as_torch_dtype(dtype))
        except Exception:
            enc = enc.to(device=dev)

    if te_is_gguf:
        from apps.backend.quantization.codexpack_tensor import CodexPackLinearQ4KTilepackV1Parameter

        def _is_quantized_tensor(tensor_obj: Any) -> bool:
            return bool(getattr(tensor_obj, "qtype", None) is not None) or isinstance(
                tensor_obj, CodexPackLinearQ4KTilepackV1Parameter
            )

        def _place_non_quant_tensors_on_device(module_obj: torch.nn.Module) -> None:
            with torch.no_grad():
                for submodule_name, submodule_obj in module_obj.named_modules():
                    for param_name, parameter in submodule_obj.named_parameters(recurse=False):
                        if parameter is None or _is_quantized_tensor(parameter):
                            continue
                        if getattr(parameter, "is_meta", False):
                            raise RuntimeError(
                                "WAN22 GGUF: unresolved meta parameter in text encoder after load: "
                                f"module={submodule_name or '<root>'} name={param_name}"
                            )
                        if parameter.device != dev:
                            parameter.data = parameter.data.to(device=dev)
                    for buffer_name, buffer in submodule_obj.named_buffers(recurse=False):
                        if buffer is None:
                            continue
                        if _is_quantized_tensor(buffer):
                            continue
                        if getattr(buffer, "is_meta", False):
                            raise RuntimeError(
                                "WAN22 GGUF: unresolved meta buffer in text encoder after load: "
                                f"module={submodule_name or '<root>'} name={buffer_name}"
                            )
                        if buffer.device != dev:
                            submodule_obj._buffers[buffer_name] = buffer.to(device=dev)

        _place_non_quant_tensors_on_device(enc)

        requested_compute_dtype = as_torch_dtype(dtype)

        def _apply_compute_dtype(module_obj: Any) -> None:
            weight = getattr(module_obj, "weight", None)
            if weight is None or not hasattr(weight, "computation_dtype"):
                return
            try:
                setattr(weight, "computation_dtype", requested_compute_dtype)
            except Exception:
                return

        _apply_compute_dtype(getattr(enc, "shared", None))
        encoder_obj = getattr(enc, "encoder", None)
        _apply_compute_dtype(getattr(encoder_obj, "embed_tokens", None))

    def _do(clean_txt: str) -> torch.Tensor:
        inputs = tok(
            [clean_txt],
            padding="max_length",
            truncation=True,
            max_length=int(max_sequence_length),
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = enc(**inputs).last_hidden_state
            mask = inputs.get("attention_mask", None)
            if mask is not None:
                out = out * mask.to(dtype=out.dtype).unsqueeze(-1)
            return out.to(as_torch_dtype(dtype))

    p = _do(prompt_cleaned)
    n = _do(negative_cleaned)

    cfg_hidden = int(getattr(getattr(enc, "config", None), "hidden_size", p.shape[-1]))
    if int(p.shape[-1]) != cfg_hidden:
        raise RuntimeError(f"WAN22 GGUF: TE hidden_size mismatch: output={int(p.shape[-1])} config={cfg_hidden}")

    log.info(
        "[wan22.gguf] TE outputs: prompt=%s negative=%s dtype=%s device=%s",
        tuple(p.shape),
        tuple(n.shape),
        str(p.dtype),
        str(p.device),
    )

    if offload_after:
        if not te_is_gguf:
            try:
                enc.to("cpu")
            except Exception:
                pass
            if dev.type == "cuda":
                log.info("[wan22.gguf] text-encoder offloaded to CPU (smart_offload)")
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return p, n
