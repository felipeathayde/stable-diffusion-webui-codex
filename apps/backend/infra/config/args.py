import argparse
import logging
import os
from typing import Mapping

_LOG = logging.getLogger("backend.infra.config.args")

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


def _normalize_device_choice(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v or v == "auto":
        return None
    if v in {"cuda", "gpu"}:
        return "cuda"
    if v == "cpu":
        return "cpu"
    if v == "mps":
        return "mps"
    if v == "xpu":
        return "xpu"
    if v in {"directml", "dml"}:
        return "directml"
    _LOG.warning("Unsupported device option '%s'; ignoring.", value)
    return None


def _normalize_dtype_choice(value: str | None, *, allow_fp8: bool = False) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v or v == "auto":
        return None
    mapping = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "single": "fp32",
    }
    if allow_fp8:
        mapping.update(
            {
                "fp8_e4m3fn": "fp8_e4m3fn",
                "fp8-e4m3fn": "fp8_e4m3fn",
                "fp8_e4": "fp8_e4m3fn",
                "fp8_e5m2": "fp8_e5m2",
                "fp8-e5m2": "fp8_e5m2",
                "fp8_e5": "fp8_e5m2",
            }
        )
    result = mapping.get(v)
    if result is None:
        _LOG.warning("Unsupported dtype option '%s'; ignoring.", value)
    return result


def _torch_dtype_for_choice(choice: str | None) -> str | None:
    if choice is None:
        return None
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "fp8_e4m3fn": "float16",
        "fp8_e5m2": "float16",
    }
    return mapping.get(choice)


def _apply_component_device_overrides(config: RuntimeMemoryConfig, ns: argparse.Namespace) -> None:
    role_choices = (
        (DeviceRole.CORE, getattr(ns, "codex_diffusion_device", None), getattr(ns, "codex_diffusion_dtype", None)),
        (DeviceRole.TEXT_ENCODER, getattr(ns, "codex_te_device", None), getattr(ns, "codex_te_dtype", None)),
        (DeviceRole.VAE, getattr(ns, "codex_vae_device", None), getattr(ns, "codex_vae_dtype", None)),
    )
    for role, device_choice, dtype_choice in role_choices:
        policy = config.component_policy(role)
        if device_choice == "cuda":
            policy.preferred_backend = DeviceBackend.CUDA
        elif device_choice == "cpu":
            policy.preferred_backend = DeviceBackend.CPU
        elif device_choice == "mps":
            policy.preferred_backend = DeviceBackend.MPS
        elif device_choice == "xpu":
            policy.preferred_backend = DeviceBackend.XPU
        elif device_choice == "directml":
            policy.preferred_backend = DeviceBackend.DIRECTML

        forced = _torch_dtype_for_choice(dtype_choice)
        if device_choice == "cpu":
            forced = "float32"
        if forced:
            policy.forced_dtype = forced


def _apply_env_overrides(ns: argparse.Namespace, env: Mapping[str, str]) -> None:
    def _set_core_dtype(val: str | None) -> None:
        ns.core_in_fp16 = False
        ns.core_in_bf16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
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

    def _set_te_dtype(val: str | None) -> None:
        for attr in ("clip_in_fp16", "clip_in_fp32", "clip_in_fp8_e4m3fn", "clip_in_fp8_e5m2", "clip_in_bf16"):
            setattr(ns, attr, False)
        if not val:
            return
        v = val.strip().lower()
        if v in {"fp16", "half", "float16"}:
            ns.clip_in_fp16 = True
        elif v in {"fp32", "float32", "float", "single"}:
            ns.clip_in_fp32 = True
        elif v in {"bf16", "bfloat16"}:
            ns.clip_in_bf16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.clip_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.clip_in_fp8_e5m2 = True

    diffusion_device_choice = _normalize_device_choice(env.get("CODEX_DIFFUSION_DEVICE"))
    diffusion_dtype_raw = env.get("CODEX_DIFFUSION_DTYPE") or env.get("WEBUI_CORE_DTYPE")
    diffusion_dtype_choice = _normalize_dtype_choice(diffusion_dtype_raw, allow_fp8=True)
    if diffusion_device_choice == "cpu" and diffusion_dtype_choice != "fp32":
        diffusion_dtype_choice = "fp32"
    _set_core_dtype(diffusion_dtype_choice or diffusion_dtype_raw)
    ns.codex_diffusion_device = diffusion_device_choice
    ns.codex_diffusion_dtype = diffusion_dtype_choice

    vae_device_choice = _normalize_device_choice(env.get("CODEX_VAE_DEVICE"))
    vae_dtype_raw = env.get("CODEX_VAE_DTYPE") or env.get("WEBUI_VAE_DTYPE")
    vae_dtype_choice = _normalize_dtype_choice(vae_dtype_raw, allow_fp8=False)
    if vae_device_choice == "cpu":
        ns.vae_in_cpu = True
        if vae_dtype_choice != "fp32":
            vae_dtype_choice = "fp32"
    _set_vae_dtype(vae_dtype_choice or vae_dtype_raw)
    ns.codex_vae_device = vae_device_choice
    ns.codex_vae_dtype = vae_dtype_choice

    te_device_choice = _normalize_device_choice(env.get("CODEX_TE_DEVICE"))
    te_dtype_raw = env.get("CODEX_TE_DTYPE")
    te_dtype_choice = _normalize_dtype_choice(te_dtype_raw, allow_fp8=True)
    if te_device_choice == "cpu" and te_dtype_choice != "fp32":
        te_dtype_choice = "fp32"
    _set_te_dtype(te_dtype_choice or te_dtype_raw)
    ns.codex_te_device = te_device_choice
    ns.codex_te_dtype = te_dtype_choice

    if te_device_choice == "cpu" and not getattr(ns, "clip_in_fp32", False):
        ns.clip_in_fp32 = True

    if te_device_choice == "cpu" and getattr(ns, "clip_in_fp16", False):
        ns.clip_in_fp16 = False

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
        all_fp16=ns.all_in_fp16,
        core_fp16=ns.core_in_fp16,
        core_bf16=ns.core_in_bf16,
        core_fp8_e4m3fn=ns.core_in_fp8_e4m3fn,
        core_fp8_e5m2=ns.core_in_fp8_e5m2,
        vae_fp16=ns.vae_in_fp16,
        vae_fp32=ns.vae_in_fp32,
        vae_bf16=ns.vae_in_bf16,
        vae_in_cpu=ns.vae_in_cpu,
        clip_fp16=getattr(ns, "clip_in_fp16", False),
        clip_fp32=getattr(ns, "clip_in_fp32", False),
        clip_bf16=getattr(ns, "clip_in_bf16", False),
        clip_fp8_e4m3fn=getattr(ns, "clip_in_fp8_e4m3fn", False),
        clip_fp8_e5m2=getattr(ns, "clip_in_fp8_e5m2", False),
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

    _apply_component_device_overrides(config, ns)

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

# (Removed ad-hoc env-to-device override block; device/dtype resolution remains in MemoryManager)

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
