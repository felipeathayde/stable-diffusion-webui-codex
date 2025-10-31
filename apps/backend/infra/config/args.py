import argparse
import os
from typing import Mapping

from apps.backend.runtime.memory.config import (
    AttentionBackend,
    AttentionConfig,
    DeviceBackend,
    DeviceRole,
    MemoryBudgets,
    PrecisionFlags,
    RuntimeMemoryConfig,
    SwapConfig,
    SwapMethod,
    SwapPolicy,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--all-in-fp32", action="store_true")
    fp_group.add_argument("--all-in-fp16", action="store_true")

    fpcore_group = parser.add_mutually_exclusive_group()
    fpcore_group.add_argument("--core-in-bf16", action="store_true")
    fpcore_group.add_argument("--core-in-fp16", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e4m3fn", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e5m2", action="store_true")

    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--vae-in-fp16", action="store_true")
    fpvae_group.add_argument("--vae-in-fp32", action="store_true")
    fpvae_group.add_argument("--vae-in-bf16", action="store_true")

    parser.add_argument("--vae-in-cpu", action="store_true")

    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
    fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
    fpte_group.add_argument("--clip-in-fp16", action="store_true")
    fpte_group.add_argument("--clip-in-fp32", action="store_true")

    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--attention-split", action="store_true")
    attn_group.add_argument("--attention-quad", action="store_true")
    attn_group.add_argument("--attention-pytorch", action="store_true")

    upcast = parser.add_mutually_exclusive_group()
    upcast.add_argument("--force-upcast-attention", action="store_true")
    upcast.add_argument("--disable-attention-upcast", action="store_true")

    parser.add_argument("--disable-xformers", action="store_true")

    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)
    parser.add_argument("--disable-ipex-hijack", action="store_true")

    vram_group = parser.add_mutually_exclusive_group()
    vram_group.add_argument("--always-gpu", action="store_true")
    vram_group.add_argument("--always-high-vram", action="store_true")
    vram_group.add_argument("--always-normal-vram", action="store_true")
    vram_group.add_argument("--always-low-vram", action="store_true")
    vram_group.add_argument("--always-no-vram", action="store_true")
    vram_group.add_argument("--always-cpu", action="store_true")

    parser.add_argument("--always-offload-from-vram", action="store_true")
    parser.add_argument("--pytorch-deterministic", action="store_true")

    parser.add_argument("--cuda-malloc", action="store_true")
    parser.add_argument("--cuda-stream", action="store_true")
    parser.add_argument("--pin-shared-memory", action="store_true")

    parser.add_argument("--disable-gpu-warning", action="store_true")

    parser.add_argument("--disable-online-tokenizer", action="store_true")

    parser.add_argument(
        "--swap-policy",
        choices=[p.value for p in SwapPolicy],
        default=SwapPolicy.CPU.value,
        help="Offload policy when VRAM is insufficient.",
    )
    parser.add_argument(
        "--swap-method",
        choices=[m.value for m in SwapMethod],
        default=SwapMethod.BLOCKED.value,
        help="Data transfer mode for swap operations.",
    )
    parser.add_argument(
        "--gpu-prefer-construct",
        action="store_true",
        help="Prefer constructing models directly on GPU (no implicit fallback).",
    )

    return parser


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _apply_env_overrides(ns: argparse.Namespace, env: Mapping[str, str]) -> None:
    def _set_core_dtype(val: str | None) -> None:
        if not val:
            return
        v = val.strip().lower()
        ns.core_in_bf16 = False
        ns.core_in_fp16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
        if v in {"bf16", "bfloat16"}:
            ns.core_in_bf16 = True
        elif v in {"fp16", "half"}:
            ns.core_in_fp16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.core_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.core_in_fp8_e5m2 = True

    def _set_vae_dtype(val: str | None) -> None:
        if not val:
            return
        v = val.strip().lower()
        if v in {"bf16", "bfloat16"}:
            ns.vae_in_bf16 = True
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = False
        elif v in {"fp16", "half"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = True
            ns.vae_in_fp32 = False
        elif v in {"fp32", "float", "single"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = True

    if _truthy(env.get("CODEX_VAE_IN_CPU")):
        ns.vae_in_cpu = True

    legacy_env = env.get("CODEX_UNET_DTYPE") or env.get("WEBUI_UNET_DTYPE")
    if legacy_env:
        raise RuntimeError(
            "Detected legacy environment variable CODEX_UNET_DTYPE/WEBUI_UNET_DTYPE. "
            "Rename to CODEX_CORE_DTYPE (or WEBUI_CORE_DTYPE) and retry."
        )

    _set_core_dtype(env.get("CODEX_CORE_DTYPE") or env.get("WEBUI_CORE_DTYPE"))
    _set_vae_dtype(env.get("CODEX_VAE_DTYPE") or env.get("WEBUI_VAE_DTYPE"))

    if _truthy(env.get("CODEX_ALL_IN_FP32")):
        ns.all_in_fp32 = True

    swap_policy = (env.get("CODEX_SWAP_POLICY") or env.get("WEBUI_SWAP_POLICY") or "").lower()
    if swap_policy in {"never", "cpu", "shared"}:
        ns.swap_policy = swap_policy

    swap_method = (env.get("CODEX_SWAP_METHOD") or env.get("WEBUI_SWAP_METHOD") or "").lower()
    if swap_method in {"blocked", "async"}:
        ns.swap_method = swap_method

    if _truthy(env.get("CODEX_GPU_PREFER_CONSTRUCT")):
        ns.gpu_prefer_construct = True


def _resolve_attention_backend(ns: argparse.Namespace) -> AttentionBackend:
    if ns.attention_split:
        return AttentionBackend.SPLIT
    if ns.attention_quad:
        return AttentionBackend.QUAD
    if ns.attention_pytorch:
        return AttentionBackend.PYTORCH
    return AttentionBackend.PYTORCH


def build_runtime_memory_config(ns: argparse.Namespace) -> RuntimeMemoryConfig:
    precision = PrecisionFlags(
        all_fp32=ns.all_in_fp32,
        all_fp16=ns.all_in_fp16,
        core_fp16=ns.core_in_fp16,
        core_bf16=ns.core_in_bf16,
        core_fp8_e4m3fn=ns.core_in_fp8_e4m3fn,
        core_fp8_e5m2=ns.core_in_fp8_e5m2,
        vae_fp16=ns.vae_in_fp16,
        vae_fp32=ns.vae_in_fp32,
        vae_bf16=ns.vae_in_bf16,
        vae_in_cpu=ns.vae_in_cpu,
        clip_fp16=ns.clip_in_fp16,
        clip_fp32=ns.clip_in_fp32,
        clip_fp8_e4m3fn=ns.clip_in_fp8_e4m3fn,
        clip_fp8_e5m2=ns.clip_in_fp8_e5m2,
    )

    attention_backend = _resolve_attention_backend(ns)
    force_upcast = bool(ns.force_upcast_attention)
    if getattr(ns, "disable_attention_upcast", False):
        force_upcast = False

    attention = AttentionConfig(
        backend=attention_backend,
        enable_flash=attention_backend == AttentionBackend.PYTORCH,
        enable_mem_efficient=attention_backend == AttentionBackend.PYTORCH,
        force_upcast=force_upcast,
        allow_split_fallback=True,
        allow_quad_fallback=True,
    )

    swap = SwapConfig(
        policy=SwapPolicy(ns.swap_policy),
        method=SwapMethod(ns.swap_method),
        always_offload=ns.always_offload_from_vram,
        pin_shared_memory=ns.pin_shared_memory,
    )

    config = RuntimeMemoryConfig(
        device_backend=DeviceBackend.AUTO,
        gpu_device_id=ns.gpu_device_id,
        gpu_prefer_construct=ns.gpu_prefer_construct,
        precision=precision,
        swap=swap,
        attention=attention,
        budgets=MemoryBudgets(),
        deterministic_algorithms=ns.pytorch_deterministic,
        disable_xformers=ns.disable_xformers,
        enable_xformers_vae=not ns.disable_xformers,
    )

    if ns.always_gpu:
        config.device_backend = DeviceBackend.CUDA
    elif ns.always_cpu:
        config.device_backend = DeviceBackend.CPU
    elif ns.directml is not None:
        config.device_backend = DeviceBackend.DIRECTML
        config.allow_directml = True

    if ns.vae_in_cpu:
        config.component_policy(DeviceRole.VAE).preferred_backend = DeviceBackend.CPU

    return config


_PARSER = _build_parser()
_ARGS, _UNKNOWN = _PARSER.parse_known_args()

_DEPRECATED = [arg for arg in _UNKNOWN if arg.startswith("--unet-in-")]
if _DEPRECATED:
    raise RuntimeError(
        "Deprecated precision flag(s) detected: "
        + ", ".join(_DEPRECATED)
        + ". Use '--core-in-*' variants instead."
    )

_apply_env_overrides(_ARGS, os.environ)

args = _ARGS
memory_config = build_runtime_memory_config(args)

dynamic_args = {
    "embedding_dir": "./embeddings",
    "emphasis_name": "original",
}


__all__ = [
    "args",
    "memory_config",
    "dynamic_args",
    "build_runtime_memory_config",
]
