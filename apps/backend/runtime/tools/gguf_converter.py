"""GGUF Converter Tool.

Converts Safetensors model files to GGUF format with optional quantization.
This is used primarily for text encoders (e.g. Z Image Qwen3 variants).
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import torch
from safetensors import safe_open

from apps.backend.quantization.api import dequantize_numpy, quantize_numpy
from apps.backend.quantization.gguf import (
    GGML_QUANT_SIZES,
    GGMLQuantizationType,
    GGUFReader,
    GGUFWriter,
    LlamaFileType,
)
from apps.backend.quantization.gguf.quant_shapes import quant_shape_to_byte_shape

logger = logging.getLogger("backend.runtime.tools.gguf_converter")


class QuantizationType(str, Enum):
    """Supported GGUF quantization types."""

    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q5_K = "Q5_K"
    Q5_1 = "Q5_1"
    Q5_0 = "Q5_0"
    Q4_K_M = "Q4_K_M"
    Q4_K = "Q4_K"
    Q4_1 = "Q4_1"
    Q4_0 = "Q4_0"
    Q3_K = "Q3_K"
    Q2_K = "Q2_K"
    IQ4_NL = "IQ4_NL"


@dataclass(slots=True)
class ConversionConfig:
    """Configuration for GGUF conversion."""

    config_path: str  # Path to config.json or folder containing it
    safetensors_path: str  # Path to .safetensors file
    output_path: str  # Output .gguf path
    quantization: QuantizationType = QuantizationType.F16
    tensor_type_overrides: Sequence[str] = ()


@dataclass(slots=True)
class ConversionProgress:
    """Progress tracking for conversion."""

    current_step: int = 0
    total_steps: int = 0
    current_tensor: str = ""
    status: str = "idle"
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100.0


class GGUFVerificationError(Exception):
    """Raised when GGUF file verification fails."""

    pass


# Key mappings: HuggingFace → GGUF
HF_TO_GGUF_KEYS = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

def _get_layer_key_mapping(layer_idx: int) -> Dict[str, str]:
    """Get key mappings for a specific layer."""
    prefix_hf = f"model.layers.{layer_idx}"
    prefix_gguf = f"blk.{layer_idx}"
    
    return {
        f"{prefix_hf}.self_attn.q_proj.weight": f"{prefix_gguf}.attn_q.weight",
        f"{prefix_hf}.self_attn.k_proj.weight": f"{prefix_gguf}.attn_k.weight",
        f"{prefix_hf}.self_attn.v_proj.weight": f"{prefix_gguf}.attn_v.weight",
        f"{prefix_hf}.self_attn.o_proj.weight": f"{prefix_gguf}.attn_output.weight",
        f"{prefix_hf}.self_attn.q_norm.weight": f"{prefix_gguf}.attn_q_norm.weight",
        f"{prefix_hf}.self_attn.k_norm.weight": f"{prefix_gguf}.attn_k_norm.weight",
        f"{prefix_hf}.mlp.gate_proj.weight": f"{prefix_gguf}.ffn_gate.weight",
        f"{prefix_hf}.mlp.up_proj.weight": f"{prefix_gguf}.ffn_up.weight",
        f"{prefix_hf}.mlp.down_proj.weight": f"{prefix_gguf}.ffn_down.weight",
        f"{prefix_hf}.input_layernorm.weight": f"{prefix_gguf}.attn_norm.weight",
        f"{prefix_hf}.post_attention_layernorm.weight": f"{prefix_gguf}.ffn_norm.weight",
    }


def build_key_mapping(num_layers: int) -> Dict[str, str]:
    """Build complete HuggingFace → GGUF key mapping."""
    mapping = dict(HF_TO_GGUF_KEYS)
    for i in range(num_layers):
        mapping.update(_get_layer_key_mapping(i))
    return mapping


@dataclass(frozen=True, slots=True)
class _TensorPlan:
    src_name: str
    gguf_name: str
    raw_shape: tuple[int, ...]
    ggml_type: GGMLQuantizationType
    stored_shape: tuple[int, ...]
    stored_dtype: np.dtype
    stored_nbytes: int


def _resolve_config_json_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"config.json not found at: {path}")
    return path


@dataclass(frozen=True, slots=True)
class _ShardedSafetensorsIndex:
    index_path: Path
    tensor_to_shard: dict[str, Path]


def _load_sharded_safetensors_index(index_path: Path) -> _ShardedSafetensorsIndex:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index (missing weight_map): {index_path}")

    base = index_path.parent
    out: dict[str, Path] = {}
    for k, v in weight_map.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"Invalid safetensors index (non-string weight_map entry): {index_path}")
        shard = (base / v).resolve()
        if shard.suffix.lower() != ".safetensors":
            raise ValueError(f"Unsupported shard type in {index_path}: {v!r} (expected .safetensors)")
        if not shard.is_file():
            raise FileNotFoundError(f"Shard referenced by {index_path} is missing: {shard}")
        out[k] = shard

    return _ShardedSafetensorsIndex(index_path=index_path.resolve(), tensor_to_shard=out)


def _pick_safetensors_index_path(weights_dir: Path) -> Path | None:
    preferred = weights_dir / "model.safetensors.index.json"
    if preferred.is_file():
        return preferred

    candidates = sorted(weights_dir.glob("*.safetensors.index.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None

    # Common HF naming conventions; prefer the most specific.
    for fname in (
        "diffusion_pytorch_model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
    ):
        p = weights_dir / fname
        if p.is_file():
            return p

    names = ", ".join(p.name for p in candidates[:6])
    more = "" if len(candidates) <= 6 else f" (+{len(candidates) - 6} more)"
    raise ValueError(
        f"Multiple safetensors index files found under {weights_dir}: {names}{more}. "
        "Pass the desired '*.safetensors.index.json' path explicitly."
    )


class _ShardedSafetensors:
    def __init__(self, index: _ShardedSafetensorsIndex) -> None:
        self._index = index
        self._handles: dict[Path, Any] = {}
        self._stack: contextlib.ExitStack | None = None

    def __enter__(self) -> "_ShardedSafetensors":
        self._stack = contextlib.ExitStack()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            self._stack.close()
        self._stack = None
        self._handles.clear()

    def keys(self):
        return self._index.tensor_to_shard.keys()

    def _handle_for(self, shard: Path):
        if shard in self._handles:
            return self._handles[shard]
        if self._stack is None:
            raise RuntimeError("Sharded safetensors handle is not open (missing context manager).")
        handle = self._stack.enter_context(safe_open(str(shard), framework="pt", device="cpu"))
        self._handles[shard] = handle
        return handle

    def get_slice(self, name: str):
        shard = self._index.tensor_to_shard.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in sharded safetensors index: {name}")
        return self._handle_for(shard).get_slice(name)

    def get_tensor(self, name: str):
        shard = self._index.tensor_to_shard.get(name)
        if shard is None:
            raise KeyError(f"Tensor not found in sharded safetensors index: {name}")
        return self._handle_for(shard).get_tensor(name)


@contextlib.contextmanager
def _open_safetensors_source(path: str):
    """Open a safetensors source from either:
    - a single `.safetensors` file,
    - a `.safetensors.index.json` file (sharded),
    - or a directory containing either a single `.safetensors` or an index file.
    """
    p = Path(path).expanduser()
    if p.is_dir():
        index_path = _pick_safetensors_index_path(p)
        if index_path is not None:
            index = _load_sharded_safetensors_index(index_path)
            with _ShardedSafetensors(index) as source:
                yield source
            return

        candidates = sorted(p.glob("*.safetensors"))
        if len(candidates) == 1:
            with safe_open(str(candidates[0]), framework="pt", device="cpu") as source:
                yield source
            return
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files found under: {p}")
        names = ", ".join(c.name for c in candidates[:6])
        more = "" if len(candidates) <= 6 else f" (+{len(candidates) - 6} more)"
        raise ValueError(
            f"Multiple .safetensors files found under {p}: {names}{more}. "
            "Pass a single file path or the '*.safetensors.index.json' path explicitly."
        )

    # Explicit index file path.
    if p.is_file() and p.name.endswith(".safetensors.index.json"):
        index = _load_sharded_safetensors_index(p)
        with _ShardedSafetensors(index) as source:
            yield source
        return

    if p.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a .safetensors file/dir/index.json, got: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {p}")
    with safe_open(str(p), framework="pt", device="cpu") as source:
        yield source


def _requested_ggml_type(quant: QuantizationType) -> GGMLQuantizationType:
    if quant == QuantizationType.F32:
        return GGMLQuantizationType.F32
    if quant == QuantizationType.F16:
        return GGMLQuantizationType.F16
    if quant == QuantizationType.Q8_0:
        return GGMLQuantizationType.Q8_0
    if quant == QuantizationType.Q5_K_M:
        return GGMLQuantizationType.Q5_K
    if quant == QuantizationType.Q6_K:
        return GGMLQuantizationType.Q6_K
    if quant == QuantizationType.Q5_K:
        return GGMLQuantizationType.Q5_K
    if quant == QuantizationType.Q5_1:
        return GGMLQuantizationType.Q5_1
    if quant == QuantizationType.Q5_0:
        return GGMLQuantizationType.Q5_0
    if quant == QuantizationType.Q4_K_M:
        return GGMLQuantizationType.Q4_K
    if quant == QuantizationType.Q4_K:
        return GGMLQuantizationType.Q4_K
    if quant == QuantizationType.Q4_1:
        return GGMLQuantizationType.Q4_1
    if quant == QuantizationType.Q4_0:
        return GGMLQuantizationType.Q4_0
    if quant == QuantizationType.Q3_K:
        return GGMLQuantizationType.Q3_K
    if quant == QuantizationType.Q2_K:
        return GGMLQuantizationType.Q2_K
    if quant == QuantizationType.IQ4_NL:
        return GGMLQuantizationType.IQ4_NL
    raise ValueError(f"Unsupported quantization: {quant}")


def _default_tensor_type_overrides(quant: QuantizationType) -> list[tuple[str, GGMLQuantizationType]]:
    """Return built-in per-tensor overrides for mixed-precision presets.

    Patterns are applied against both the source tensor name and the GGUF tensor name.
    """
    if quant == QuantizationType.Q5_K_M:
        return [
            # Embeddings / output: keep higher precision to preserve prompt semantics.
            (r"(?:^|\.)token_embd\.weight$", GGMLQuantizationType.Q8_0),
            (r"(?:^|\.)output\.weight$", GGMLQuantizationType.Q8_0),
            (r"model\.embed_tokens\.weight$", GGMLQuantizationType.Q8_0),
            (r"lm_head\.weight$", GGMLQuantizationType.Q8_0),
            # Attention projections: bump to 6-bit.
            (r"(?:^|\.)attn_(?:q|k|v|output)\.weight$", GGMLQuantizationType.Q6_K),
            (r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$", GGMLQuantizationType.Q6_K),
        ]

    if quant == QuantizationType.Q4_K_M:
        return [
            # Embeddings / output: bump to 6-bit (still much smaller than fp16).
            (r"(?:^|\.)token_embd\.weight$", GGMLQuantizationType.Q6_K),
            (r"(?:^|\.)output\.weight$", GGMLQuantizationType.Q6_K),
            (r"model\.embed_tokens\.weight$", GGMLQuantizationType.Q6_K),
            (r"lm_head\.weight$", GGMLQuantizationType.Q6_K),
            # Attention projections: bump to 5-bit K.
            (r"(?:^|\.)attn_(?:q|k|v|output)\.weight$", GGMLQuantizationType.Q5_K),
            (r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$", GGMLQuantizationType.Q5_K),
        ]

    return []


def _compile_tensor_overrides(
    quant: QuantizationType,
    extra_overrides: Sequence[str],
) -> list[tuple[re.Pattern[str], GGMLQuantizationType]]:
    """Compile built-in + user-provided tensor quantization overrides.

    `extra_overrides` entries use a llama.cpp-like format: `<regex>=<quant>`
    where `<quant>` is any `QuantizationType` value (case-insensitive).
    """
    rules: list[tuple[re.Pattern[str], GGMLQuantizationType]] = []

    for pattern, qtype in _default_tensor_type_overrides(quant):
        rules.append((re.compile(pattern), qtype))

    for entry in extra_overrides:
        raw = str(entry or "").strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")
        pattern, qname = raw.split("=", 1)
        pattern = pattern.strip()
        qname = qname.strip()
        if not pattern or not qname:
            raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")

        try:
            q_enum = QuantizationType(qname.upper())
        except ValueError as exc:
            raise ValueError(f"Invalid quant type in override {raw!r}: {qname!r}") from exc

        rules.append((re.compile(pattern), _requested_ggml_type(q_enum)))

    return rules


def _select_tensor_ggml_type(shape: Sequence[int], requested: GGMLQuantizationType) -> GGMLQuantizationType:
    """Select the per-tensor GGML type.

    Behavior:
    - If requested is F16/F32: apply to all tensors.
    - Otherwise: keep 1D tensors in F16 and only quantize tensors whose last dim
      is divisible by the block size.
    """
    if requested in {GGMLQuantizationType.F16, GGMLQuantizationType.F32}:
        return requested

    # Common GGUF convention: keep 1D tensors in F16.
    if len(shape) <= 1:
        return GGMLQuantizationType.F16

    block_size, _ = GGML_QUANT_SIZES[requested]
    if shape[-1] % block_size != 0:
        return GGMLQuantizationType.F16

    return requested


def _plan_tensors(
    tensor_names: list[str],
    safetensors_handle: Any,
    key_mapping: Dict[str, str],
    requested: GGMLQuantizationType,
    overrides: list[tuple[re.Pattern[str], GGMLQuantizationType]],
) -> list[_TensorPlan]:
    plans: list[_TensorPlan] = []

    for src_name in tensor_names:
        sl = safetensors_handle.get_slice(src_name)
        raw_shape = tuple(int(x) for x in sl.get_shape())
        gguf_name = key_mapping.get(src_name, src_name)

        desired = requested
        for rx, qtype in overrides:
            if rx.search(src_name) or rx.search(gguf_name):
                desired = qtype
        ggml_type = _select_tensor_ggml_type(raw_shape, desired)

        if ggml_type == GGMLQuantizationType.F16:
            stored_dtype = np.dtype(np.float16)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 2)
        elif ggml_type == GGMLQuantizationType.F32:
            stored_dtype = np.dtype(np.float32)
            stored_shape = raw_shape
            stored_nbytes = int(np.prod(raw_shape, dtype=np.int64) * 4)
        else:
            stored_dtype = np.dtype(np.uint8)
            stored_shape = quant_shape_to_byte_shape(raw_shape, ggml_type)
            stored_nbytes = int(np.prod(stored_shape, dtype=np.int64))

        plans.append(
            _TensorPlan(
                src_name=src_name,
                gguf_name=gguf_name,
                raw_shape=raw_shape,
                ggml_type=ggml_type,
                stored_shape=stored_shape,
                stored_dtype=stored_dtype,
                stored_nbytes=stored_nbytes,
            )
        )

    return plans


def _add_basic_metadata(writer: GGUFWriter, arch: str, config: dict, quant: QuantizationType) -> None:
    name = str(config.get("_name_or_path") or config.get("name") or "model")
    writer.add_name(name)

    requested = _requested_ggml_type(quant)
    if requested == GGMLQuantizationType.F16:
        writer.add_file_type(int(LlamaFileType.MOSTLY_F16))
    elif requested == GGMLQuantizationType.F32:
        writer.add_file_type(int(LlamaFileType.ALL_F32))
    elif requested == GGMLQuantizationType.Q8_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q8_0))
    elif requested == GGMLQuantizationType.Q6_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q6_K))
    elif requested == GGMLQuantizationType.Q5_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_K_M))
    elif requested == GGMLQuantizationType.Q5_1:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_1))
    elif requested == GGMLQuantizationType.Q5_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_0))
    elif requested == GGMLQuantizationType.Q4_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_K_M))
    elif requested == GGMLQuantizationType.Q4_1:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_1))
    elif requested == GGMLQuantizationType.Q4_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_0))
    elif requested == GGMLQuantizationType.Q3_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q3_K_M))
    elif requested == GGMLQuantizationType.Q2_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q2_K))
    elif requested == GGMLQuantizationType.IQ4_NL:
        writer.add_file_type(int(LlamaFileType.MOSTLY_IQ4_NL))

    # Minimal arch metadata; loaders in this repo generally key off tensor names and shapes.
    writer.add_uint32(f"{arch}.context_length", int(config.get("max_position_embeddings", 4096)))
    writer.add_uint32(f"{arch}.embedding_length", int(config.get("hidden_size", 4096)))
    writer.add_uint32(f"{arch}.block_count", int(config.get("num_hidden_layers", 32)))
    writer.add_uint32(f"{arch}.attention.head_count", int(config.get("num_attention_heads", 32)))
    writer.add_uint32(f"{arch}.attention.head_count_kv", int(config.get("num_key_value_heads", 8)))
    writer.add_float32(f"{arch}.rope.freq_base", float(config.get("rope_theta", 10000.0)))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))


def convert_safetensors_to_gguf(
    config: ConversionConfig,
    progress_callback: Optional[Callable[[ConversionProgress], None]] = None,
) -> str:
    """Convert a Safetensors file to GGUF format.
    
    Args:
        config: Conversion configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to the output GGUF file
    """
    progress = ConversionProgress(status="loading_config")
    
    def update_progress():
        if progress_callback:
            progress_callback(progress)
    
    update_progress()
    
    # Load model config
    config_path = _resolve_config_json_path(config.config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    
    logger.info("Loaded config: %s", model_config.get("model_type", "unknown"))
    
    # Get architecture info
    arch = str(model_config.get("model_type") or "llama")
    num_layers = int(model_config.get("num_hidden_layers", 32))
    requested_type = _requested_ggml_type(config.quantization)
    overrides = _compile_tensor_overrides(config.quantization, config.tensor_type_overrides)
    
    # Build key mapping
    key_mapping = build_key_mapping(num_layers)
    
    # Load safetensors
    progress.status = "loading_weights"
    update_progress()
    
    logger.info("Loading safetensors: %s", config.safetensors_path)
    
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with _open_safetensors_source(config.safetensors_path) as sf:
        tensor_names = list(sf.keys())
        progress.total_steps = len(tensor_names)

        plans = _plan_tensors(tensor_names, sf, key_mapping, requested_type, overrides)

        writer = GGUFWriter(path=str(output_path), arch=arch)
        _add_basic_metadata(writer, arch, model_config, config.quantization)

        for plan in plans:
            raw_dtype = None if plan.ggml_type in {GGMLQuantizationType.F16, GGMLQuantizationType.F32} else plan.ggml_type
            writer.add_tensor_info(
                plan.gguf_name,
                tensor_shape=plan.stored_shape,
                tensor_dtype=plan.stored_dtype,
                tensor_nbytes=plan.stored_nbytes,
                raw_dtype=raw_dtype,
            )

        progress.status = "converting"
        update_progress()

        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_ti_data_to_file()

            assert writer.fout is not None
            out = writer.fout[0]
            writer.write_padding(out, out.tell())

            # Stream-write tensors in the same order used for tensor-info offsets.
            chunk_rows = 1024
            for i, plan in enumerate(plans):
                progress.current_step = i + 1
                progress.current_tensor = plan.src_name
                update_progress()

                sl = sf.get_slice(plan.src_name)
                shape = tuple(int(x) for x in sl.get_shape())
                if shape != plan.raw_shape:
                    raise RuntimeError(f"Tensor shape changed during conversion for {plan.src_name}: {shape} vs {plan.raw_shape}")

                bytes_written = 0

                if plan.ggml_type == GGMLQuantizationType.F16:
                    target_dtype = torch.float16
                    if len(shape) == 1:
                        t = sl[:].to(target_dtype).contiguous()
                        out.write(t.numpy().tobytes(order="C"))
                        bytes_written += t.numel() * 2
                    elif len(shape) == 2:
                        rows = shape[0]
                        for start in range(0, rows, chunk_rows):
                            chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                            out.write(chunk.numpy().tobytes(order="C"))
                            bytes_written += chunk.numel() * 2
                    else:
                        t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                        out.write(t.numpy().tobytes(order="C"))
                        bytes_written += t.numel() * 2

                elif plan.ggml_type == GGMLQuantizationType.F32:
                    target_dtype = torch.float32
                    if len(shape) == 1:
                        t = sl[:].to(target_dtype).contiguous()
                        out.write(t.numpy().tobytes(order="C"))
                        bytes_written += t.numel() * 4
                    elif len(shape) == 2:
                        rows = shape[0]
                        for start in range(0, rows, chunk_rows):
                            chunk = sl[start : min(rows, start + chunk_rows)].to(target_dtype).contiguous()
                            out.write(chunk.numpy().tobytes(order="C"))
                            bytes_written += chunk.numel() * 4
                    else:
                        t = sf.get_tensor(plan.src_name).to(target_dtype).contiguous()
                        out.write(t.numpy().tobytes(order="C"))
                        bytes_written += t.numel() * 4

                else:
                    if len(shape) == 1:
                        # By policy we keep 1D tensors in F16, so this would indicate a planning bug.
                        raise RuntimeError(f"Unexpected quantized 1D tensor plan for {plan.src_name}: {shape}")

                    if len(shape) == 2:
                        rows = shape[0]
                        for start in range(0, rows, chunk_rows):
                            chunk = sl[start : min(rows, start + chunk_rows)].to(torch.float32).contiguous()
                            arr = chunk.numpy()
                            try:
                                q = quantize_numpy(arr, plan.ggml_type)
                            except Exception as exc:
                                raise RuntimeError(
                                    f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                                ) from exc
                            out.write(q.tobytes(order="C"))
                            bytes_written += q.nbytes
                    else:
                        t = sf.get_tensor(plan.src_name).to(torch.float32).contiguous()
                        arr = t.numpy()
                        try:
                            q = quantize_numpy(arr, plan.ggml_type)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to quantize tensor {plan.src_name} to {plan.ggml_type.name}: {exc}"
                            ) from exc
                        out.write(q.tobytes(order="C"))
                        bytes_written += q.nbytes

                if bytes_written != plan.stored_nbytes:
                    raise RuntimeError(
                        f"Byte count mismatch for {plan.src_name}: wrote {bytes_written}, expected {plan.stored_nbytes}"
                    )
                writer.write_padding(out, plan.stored_nbytes)
        finally:
            writer.close()

    logger.info("GGUF file written: %s", output_path)
    
    # Verification step: validate the generated file
    progress.status = "verifying"
    update_progress()
    
    _verify_gguf_file(
        gguf_path=str(output_path),
        source_safetensors=config.safetensors_path,
        tensor_plans=plans,
        key_mapping=key_mapping,
    )
    
    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    
    logger.info("GGUF conversion and verification complete: %s", output_path)
    return str(output_path)


def _verify_gguf_file(
    gguf_path: str,
    source_safetensors: str,
    tensor_plans: list[_TensorPlan],
    key_mapping: Dict[str, str],
) -> None:
    """Verify the generated GGUF file against source and expected metadata.
    
    Args:
        gguf_path: Path to the generated GGUF file
        source_safetensors: Path to the source safetensors file
        expected_tensor_infos: List of expected tensor info dicts from conversion
        key_mapping: HuggingFace to GGUF key mapping used during conversion
    
    Raises:
        GGUFVerificationError: If verification fails
    """
    logger.info("Verifying GGUF file: %s", gguf_path)
    
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise GGUFVerificationError(f"GGUF file does not exist: {gguf_path}")
    
    # 1) Parse with the repo GGUF reader (validates header/KV/TI/offsets).
    reader = GGUFReader(str(gguf_path))

    expected_count = len(tensor_plans)
    if len(reader.tensors) != expected_count:
        raise GGUFVerificationError(
            f"Tensor count mismatch: GGUF has {len(reader.tensors)}, expected {expected_count}"
        )

    by_name = {t.name: t for t in reader.tensors}

    for plan in tensor_plans:
        if plan.gguf_name not in by_name:
            raise GGUFVerificationError(f"Tensor missing in GGUF: {plan.gguf_name}")
        t = by_name[plan.gguf_name]
        if t.tensor_type != plan.ggml_type:
            raise GGUFVerificationError(
                f"DTYPE mismatch for {plan.gguf_name}: GGUF has {t.tensor_type.name}, expected {plan.ggml_type.name}"
            )
        expected_shape_gguf = tuple(reversed(plan.raw_shape))
        if tuple(int(x) for x in t.shape) != expected_shape_gguf:
            raise GGUFVerificationError(
                f"Shape mismatch for {plan.gguf_name}: GGUF has {tuple(int(x) for x in t.shape)}, expected {expected_shape_gguf}"
            )
        if int(t.n_bytes) != int(plan.stored_nbytes):
            raise GGUFVerificationError(
                f"Byte size mismatch for {plan.gguf_name}: GGUF has {int(t.n_bytes)}, expected {int(plan.stored_nbytes)}"
            )

    logger.info("GGUF structure verification passed (%d tensors)", expected_count)

    def _quant_spotcheck_tol(qtype: GGMLQuantizationType) -> tuple[float, float]:
        # Spot-check tolerances are intentionally loose: goal is catching layout/packing bugs,
        # not measuring perceptual quality.
        if qtype == GGMLQuantizationType.Q2_K:
            return (1.0, 1.0)
        if qtype == GGMLQuantizationType.Q3_K:
            return (0.8, 0.8)
        return (0.6, 0.6)

    # 2) Spot-check a few tensors against source.
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    with _open_safetensors_source(source_safetensors) as source:
        for plan in tensor_plans[:3]:
            src_name = reverse_mapping.get(plan.gguf_name, plan.gguf_name)
            if src_name not in source.keys():
                continue

            gguf_tensor = by_name[plan.gguf_name]
            src_slice = source.get_slice(src_name)
            src_shape = tuple(int(x) for x in src_slice.get_shape())

            if plan.ggml_type == GGMLQuantizationType.F16:
                if len(src_shape) == 1:
                    src = src_slice[:4].to(torch.float16).flatten().numpy().tobytes(order="C")
                else:
                    src = src_slice[:1].to(torch.float16).flatten()[:4].numpy().tobytes(order="C")
                gg = gguf_tensor.data.reshape(-1).view(np.uint8)[: len(src)].tobytes(order="C")
                if gg != src:
                    raise GGUFVerificationError(f"Tensor data mismatch (F16) for {plan.gguf_name}")

            elif plan.ggml_type == GGMLQuantizationType.F32:
                if len(src_shape) == 1:
                    src = src_slice[:4].to(torch.float32).flatten().numpy().tobytes(order="C")
                else:
                    src = src_slice[:1].to(torch.float32).flatten()[:4].numpy().tobytes(order="C")
                gg = gguf_tensor.data.reshape(-1).view(np.uint8)[: len(src)].tobytes(order="C")
                if gg != src:
                    raise GGUFVerificationError(f"Tensor data mismatch (F32) for {plan.gguf_name}")

            else:
                # Quantized: dequantize first row and compare roughly.
                row_bytes = gguf_tensor.data.reshape((-1, gguf_tensor.data.shape[-1]))[0]
                out = dequantize_numpy(row_bytes, plan.ggml_type)

                # Avoid loading the full tensor: grab only the first outer slice.
                ref_chunk = src_slice[:1].reshape(-1, src_shape[-1])[0].float().numpy()
                n = min(256, ref_chunk.shape[0])
                if not np.all(np.isfinite(out[:n])):
                    raise GGUFVerificationError(f"Non-finite dequant output for {plan.gguf_name}")
                rtol, atol = _quant_spotcheck_tol(plan.ggml_type)
                if not np.allclose(out[:n], ref_chunk[:n], rtol=rtol, atol=atol):
                    raise GGUFVerificationError(f"Quantized spot-check mismatch for {plan.gguf_name}")

    logger.info("GGUF spot-check passed")


__all__ = [
    "ConversionConfig",
    "ConversionProgress", 
    "QuantizationType",
    "GGUFVerificationError",
    "convert_safetensors_to_gguf",
]
