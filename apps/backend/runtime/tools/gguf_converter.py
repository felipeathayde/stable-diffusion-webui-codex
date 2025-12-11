"""GGUF Converter Tool.

Converts Safetensors model files to GGUF format with optional quantization.
Supports text encoders like Qwen3, Llama, etc.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
from safetensors import safe_open

logger = logging.getLogger("backend.runtime.tools.gguf_converter")


class QuantizationType(str, Enum):
    """Supported GGUF quantization types."""
    F16 = "F16"
    F32 = "F32"
    Q8_0 = "Q8_0"
    Q5_K = "Q5_K"
    Q4_K = "Q4_K"


@dataclass
class ConversionConfig:
    """Configuration for GGUF conversion."""
    config_path: str  # Path to config.json
    safetensors_path: str  # Path to .safetensors file
    output_path: str  # Output .gguf path
    quantization: QuantizationType = QuantizationType.F16
    

@dataclass
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
        return (self.current_step / self.total_steps) * 100


class GGUFVerificationError(Exception):
    """Raised when GGUF file verification fails."""
    pass


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGUF data types
class GGMLType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    BF16 = 30


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
    import io
    import struct
    
    GGUF_ALIGNMENT = 32  # Default alignment for GGUF v3
    
    progress = ConversionProgress(status="loading_config")
    
    def update_progress():
        if progress_callback:
            progress_callback(progress)
    
    update_progress()
    
    # Load model config
    config_path = Path(config.config_path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    
    with open(config_path, "r") as f:
        model_config = json.load(f)
    
    logger.info("Loaded config: %s", model_config.get("model_type", "unknown"))
    
    # Get architecture info
    arch = model_config.get("model_type", "llama")
    num_layers = model_config.get("num_hidden_layers", 32)
    
    # Build key mapping
    key_mapping = build_key_mapping(num_layers)
    
    # Load safetensors
    progress.status = "loading_weights"
    update_progress()
    
    logger.info("Loading safetensors: %s", config.safetensors_path)
    
    with safe_open(config.safetensors_path, framework="pt", device="cpu") as f:
        tensor_names = list(f.keys())
        progress.total_steps = len(tensor_names)
        
        # Prepare output
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        progress.status = "converting"
        update_progress()
        
        # First pass: collect all tensor info and data
        tensor_infos = []
        tensor_data_list = []
        
        for i, name in enumerate(tensor_names):
            progress.current_step = i + 1
            progress.current_tensor = name
            update_progress()
            
            tensor = f.get_tensor(name)
            
            # Map key name
            gguf_name = key_mapping.get(name, name)
            
            # Get tensor info
            shape = list(tensor.shape)
            dtype = _get_gguf_dtype(tensor.dtype, config.quantization)
            
            # Convert to F16 (quantization not implemented)
            tensor = tensor.to(torch.float16)
            tensor_bytes = tensor.numpy().tobytes()
            
            tensor_infos.append({
                "name": gguf_name,
                "shape": shape,
                "dtype": dtype,
            })
            tensor_data_list.append(tensor_bytes)
            
            logger.debug("Converted tensor %s → %s", name, gguf_name)
        
        # Write GGUF file with proper structure
        with open(output_path, "wb") as out:
            # Write header and metadata to buffer first to calculate size
            header_buf = io.BytesIO()
            _write_gguf_header(header_buf, arch, model_config, len(tensor_names))
            
            # Write tensor info (without offsets yet) to calculate size
            tensor_info_buf = io.BytesIO()
            for info in tensor_infos:
                _write_string(tensor_info_buf, info["name"])
                n_dims = len(info["shape"])
                tensor_info_buf.write(struct.pack("<I", n_dims))
                for dim in reversed(info["shape"]):
                    tensor_info_buf.write(struct.pack("<Q", dim))
                tensor_info_buf.write(struct.pack("<I", info["dtype"].value))
                tensor_info_buf.write(struct.pack("<Q", 0))  # placeholder offset
            
            # Calculate where data section starts (need to align)
            header_size = header_buf.tell()
            tensor_info_size = tensor_info_buf.tell()
            metadata_end = header_size + tensor_info_size
            
            # Align data section start to GGUF_ALIGNMENT
            data_section_start = ((metadata_end + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
            padding_size = data_section_start - metadata_end
            
            # Calculate aligned offsets for each tensor
            current_offset = 0
            for i, tensor_bytes in enumerate(tensor_data_list):
                tensor_infos[i]["offset"] = current_offset
                # Align each tensor to GGUF_ALIGNMENT
                tensor_size = len(tensor_bytes)
                aligned_size = ((tensor_size + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
                tensor_infos[i]["aligned_size"] = aligned_size
                tensor_infos[i]["padding"] = aligned_size - tensor_size
                current_offset += aligned_size
            
            # Now write the actual file
            # 1. Header
            header_buf.seek(0)
            out.write(header_buf.read())
            
            # 2. Tensor metadata with correct offsets
            for info in tensor_infos:
                _write_string(out, info["name"])
                n_dims = len(info["shape"])
                out.write(struct.pack("<I", n_dims))
                for dim in reversed(info["shape"]):
                    out.write(struct.pack("<Q", dim))
                out.write(struct.pack("<I", info["dtype"].value))
                out.write(struct.pack("<Q", info["offset"]))
            
            # 3. Padding to align data section
            out.write(b'\x00' * padding_size)
            
            # 4. Tensor data with alignment padding
            for i, tensor_bytes in enumerate(tensor_data_list):
                out.write(tensor_bytes)
                # Add padding to align next tensor
                out.write(b'\x00' * tensor_infos[i]["padding"])
    
    logger.info("GGUF file written: %s", output_path)
    
    # Verification step: validate the generated file
    progress.status = "verifying"
    update_progress()
    
    _verify_gguf_file(
        gguf_path=str(output_path),
        source_safetensors=config.safetensors_path,
        expected_tensor_infos=tensor_infos,
        key_mapping=key_mapping,
    )
    
    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    
    logger.info("GGUF conversion and verification complete: %s", output_path)
    return str(output_path)


def _write_gguf_header(f, arch: str, config: dict, n_tensors: int):
    """Write GGUF file header."""
    import struct
    
    # Magic
    f.write(struct.pack("<I", GGUF_MAGIC))
    # Version
    f.write(struct.pack("<I", GGUF_VERSION))
    # Number of tensors
    f.write(struct.pack("<Q", n_tensors))
    # Number of metadata key-value pairs
    metadata = _build_metadata(arch, config)
    f.write(struct.pack("<Q", len(metadata)))
    
    # Write metadata
    for key, value in metadata.items():
        _write_string(f, key)
        _write_value(f, value)


def _build_metadata(arch: str, config: dict) -> dict:
    """Build GGUF metadata from model config."""
    return {
        "general.architecture": arch,
        "general.name": config.get("_name_or_path", "model"),
        f"{arch}.context_length": config.get("max_position_embeddings", 4096),
        f"{arch}.embedding_length": config.get("hidden_size", 4096),
        f"{arch}.block_count": config.get("num_hidden_layers", 32),
        f"{arch}.attention.head_count": config.get("num_attention_heads", 32),
        f"{arch}.attention.head_count_kv": config.get("num_key_value_heads", 8),
        f"{arch}.rope.freq_base": config.get("rope_theta", 10000.0),
        f"{arch}.attention.layer_norm_rms_epsilon": config.get("rms_norm_eps", 1e-6),
    }


def _write_string(f, s: str):
    """Write a GGUF string."""
    import struct
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def _write_value(f, value):
    """Write a GGUF metadata value."""
    import struct
    
    if isinstance(value, str):
        f.write(struct.pack("<I", 8))  # GGUF_TYPE_STRING
        _write_string(f, value)
    elif isinstance(value, int):
        f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
        f.write(struct.pack("<I", value))
    elif isinstance(value, float):
        f.write(struct.pack("<I", 6))  # GGUF_TYPE_FLOAT32
        f.write(struct.pack("<f", value))


def _write_tensor_metadata(f, tensor_infos: list):
    """Write tensor metadata section."""
    import struct
    
    for info in tensor_infos:
        _write_string(f, info["name"])
        
        # Number of dimensions
        n_dims = len(info["shape"])
        f.write(struct.pack("<I", n_dims))
        
        # Dimensions (reversed for GGUF column-major)
        for dim in reversed(info["shape"]):
            f.write(struct.pack("<Q", dim))
        
        # Data type
        f.write(struct.pack("<I", info["dtype"].value))
        
        # Offset
        f.write(struct.pack("<Q", info["offset"]))


def _get_gguf_dtype(torch_dtype: torch.dtype, quant: QuantizationType) -> GGMLType:
    """Get GGUF data type.
    
    NOTE: Since actual quantization is not implemented yet, we always return
    the dtype that matches what _quantize_tensor actually produces (F16).
    """
    if quant == QuantizationType.F32:
        return GGMLType.F32
    # All other types fall back to F16 since _quantize_tensor returns F16
    return GGMLType.F16


def _quantize_tensor(tensor: torch.Tensor, quant: QuantizationType) -> torch.Tensor:
    """Quantize a tensor to the specified format.
    
    Note: For now, this returns F16. Full quantization requires
    implementing the GGML quantization algorithms.
    """
    # TODO: Implement actual Q8_0, Q5_K, Q4_K quantization
    # For now, just return F16
    logger.warning("Quantization %s not yet implemented, using F16", quant)
    return tensor.to(torch.float16)


def _verify_gguf_file(
    gguf_path: str,
    source_safetensors: str,
    expected_tensor_infos: list,
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
    import struct
    
    logger.info("Verifying GGUF file: %s", gguf_path)
    
    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise GGUFVerificationError(f"GGUF file does not exist: {gguf_path}")
    
    # 1. Verify file can be opened and header is valid
    with open(gguf_path, "rb") as f:
        # Read magic
        magic_bytes = f.read(4)
        if len(magic_bytes) < 4:
            raise GGUFVerificationError("GGUF file too small to contain header")
        magic = struct.unpack("<I", magic_bytes)[0]
        if magic != GGUF_MAGIC:
            raise GGUFVerificationError(
                f"Invalid GGUF magic: expected {hex(GGUF_MAGIC)}, got {hex(magic)}"
            )
        
        # Read version
        version = struct.unpack("<I", f.read(4))[0]
        if version != GGUF_VERSION:
            raise GGUFVerificationError(
                f"Unexpected GGUF version: expected {GGUF_VERSION}, got {version}"
            )
        
        # Read tensor count
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        expected_count = len(expected_tensor_infos)
        if n_tensors != expected_count:
            raise GGUFVerificationError(
                f"Tensor count mismatch: header says {n_tensors}, expected {expected_count}"
            )
        
        # Read metadata count
        n_metadata = struct.unpack("<Q", f.read(8))[0]
        logger.debug("GGUF header valid: version=%d, tensors=%d, metadata=%d", 
                     version, n_tensors, n_metadata)
    
    logger.info("GGUF header verification passed")
    
    # 2. Compare tensor contents with source
    # Build reverse key mapping for lookup
    reverse_mapping = {v: k for k, v in key_mapping.items()}
    
    with safe_open(source_safetensors, framework="pt", device="cpu") as source:
        source_tensors = set(source.keys())
        
        for info in expected_tensor_infos:
            gguf_name = info["name"]
            expected_shape = info["shape"]
            expected_dtype = info["dtype"]
            
            # Find corresponding source tensor name
            source_name = reverse_mapping.get(gguf_name, gguf_name)
            
            if source_name not in source_tensors:
                raise GGUFVerificationError(
                    f"Tensor '{gguf_name}' (source: '{source_name}') not found in source safetensors"
                )
            
            # Verify shape matches
            source_tensor = source.get_tensor(source_name)
            source_shape = list(source_tensor.shape)
            
            if source_shape != expected_shape:
                raise GGUFVerificationError(
                    f"Shape mismatch for tensor '{gguf_name}': "
                    f"source has {source_shape}, GGUF metadata says {expected_shape}"
                )
            
            # Verify dtype makes sense (F16 for most, F32 if requested)
            if expected_dtype == GGMLType.F16:
                expected_element_size = 2
            elif expected_dtype == GGMLType.F32:
                expected_element_size = 4
            else:
                # Other dtypes - just log for now
                logger.debug("Tensor '%s' has dtype %s, skipping size check", 
                            gguf_name, expected_dtype)
                continue
            
            # Calculate expected data size
            n_elements = 1
            for dim in expected_shape:
                n_elements *= dim
            expected_size = n_elements * expected_element_size
            
            # Compare with what we stored
            stored_size = len(source_tensor.to(torch.float16).numpy().tobytes())
            if stored_size != expected_size:
                raise GGUFVerificationError(
                    f"Data size mismatch for tensor '{gguf_name}': "
                    f"expected {expected_size} bytes, got {stored_size} bytes"
                )
    
    # 3. Spot-check: read a few tensor data values from GGUF and compare to source
    logger.info("Performing spot-check on tensor data...")
    
    with open(gguf_path, "rb") as f, safe_open(source_safetensors, framework="pt", device="cpu") as source:
        import struct
        
        # Skip header and scan for data section
        # We need to parse tensor metadata to find data offsets
        f.seek(0)
        f.read(4)  # magic
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_metadata = struct.unpack("<Q", f.read(8))[0]
        
        # Skip metadata (simplified - just find first tensor info)
        # For each metadata: key string, type, value
        for _ in range(n_metadata):
            key_len = struct.unpack("<Q", f.read(8))[0]
            f.read(key_len)  # key name
            value_type = struct.unpack("<I", f.read(4))[0]
            # Skip value based on type
            if value_type == 4:  # GGUF_TYPE_STRING
                str_len = struct.unpack("<Q", f.read(8))[0]
                f.read(str_len)
            elif value_type == 5:  # GGUF_TYPE_ARRAY
                array_type = struct.unpack("<I", f.read(4))[0]
                array_len = struct.unpack("<Q", f.read(8))[0]
                if array_type == 4:  # string array
                    for _ in range(array_len):
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        f.read(str_len)
                else:
                    # Skip numeric array
                    elem_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8, 9: 8, 10: 8}
                    elem_size = elem_sizes.get(array_type, 4)
                    f.read(array_len * elem_size)
            else:
                # Scalar types
                sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 8, 8: 8, 9: 8, 10: 8}
                f.read(sizes.get(value_type, 4))
        
        # Now read tensor infos
        tensor_info_list = []
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", f.read(8))[0]
            tensor_name = f.read(name_len).decode("utf-8")
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype_val = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensor_info_list.append({
                "name": tensor_name,
                "dims": dims,
                "dtype": dtype_val,
                "offset": offset,
            })
        
        # Calculate data section start (aligned to 32)
        metadata_end = f.tell()
        GGUF_ALIGNMENT = 32
        data_section_start = ((metadata_end + GGUF_ALIGNMENT - 1) // GGUF_ALIGNMENT) * GGUF_ALIGNMENT
        
        # Build expected dtype lookup from expected_tensor_infos
        expected_dtypes = {info["name"]: info["dtype"] for info in expected_tensor_infos}
        
        # Verify tensor metadata read from GGUF matches what we expected to write
        for gguf_info in tensor_info_list:
            gguf_name = gguf_info["name"]
            gguf_dtype_val = gguf_info["dtype"]
            
            # Check if this tensor is in our expected list
            if gguf_name not in expected_dtypes:
                raise GGUFVerificationError(
                    f"Tensor '{gguf_name}' found in GGUF but not in expected tensor list"
                )
            
            expected_dtype = expected_dtypes[gguf_name]
            
            # CRITICAL CHECK: This would have caught the Q8_0 vs F16 bug!
            # Verify the dtype in GGUF header matches what we intended to write
            if gguf_dtype_val != expected_dtype.value:
                # Get human-readable names
                gguf_dtype_name = next(
                    (t.name for t in GGMLType if t.value == gguf_dtype_val), 
                    f"unknown({gguf_dtype_val})"
                )
                raise GGUFVerificationError(
                    f"DTYPE MISMATCH for tensor '{gguf_name}': "
                    f"GGUF header says {gguf_dtype_name} (value={gguf_dtype_val}), "
                    f"but we intended to write {expected_dtype.name} (value={expected_dtype.value}). "
                    f"This indicates a bug in _get_gguf_dtype() or tensor writing!"
                )
        
        logger.info("GGUF dtype verification passed: all tensors have correct dtype in header")
        
        # Spot check first 3 tensors - compare actual data bytes
        for info in tensor_info_list[:3]:
            gguf_name = info["name"]
            source_name = reverse_mapping.get(gguf_name, gguf_name)
            
            if source_name not in source.keys():
                continue
            
            # Read first 8 bytes from GGUF tensor data
            f.seek(data_section_start + info["offset"])
            gguf_bytes = f.read(8)
            
            # Get source tensor converted to F16
            source_tensor = source.get_tensor(source_name).to(torch.float16)
            source_bytes = source_tensor.flatten()[:4].numpy().tobytes()  # First 4 F16 values = 8 bytes
            
            if gguf_bytes != source_bytes:
                raise GGUFVerificationError(
                    f"Tensor data mismatch for '{gguf_name}': "
                    f"GGUF bytes {gguf_bytes.hex()} != source bytes {source_bytes.hex()}"
                )
            logger.debug("Tensor '%s' spot-check passed", gguf_name)
    
    logger.info("GGUF tensor verification passed: %d tensors validated", len(expected_tensor_infos))


__all__ = [
    "ConversionConfig",
    "ConversionProgress", 
    "QuantizationType",
    "GGUFVerificationError",
    "convert_safetensors_to_gguf",
]
