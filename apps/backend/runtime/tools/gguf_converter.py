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
    hidden_size = model_config.get("hidden_size", 4096)
    vocab_size = model_config.get("vocab_size", 32000)
    
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
        
        # Write GGUF file
        with open(output_path, "wb") as out:
            # Write header
            _write_gguf_header(out, arch, model_config, len(tensor_names))
            
            # Write tensors
            tensor_infos = []
            tensor_data_offset = 0
            
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
                
                # Quantize if needed
                if config.quantization != QuantizationType.F16:
                    tensor = _quantize_tensor(tensor, config.quantization)
                else:
                    tensor = tensor.to(torch.float16)
                
                tensor_infos.append({
                    "name": gguf_name,
                    "shape": shape,
                    "dtype": dtype,
                    "offset": tensor_data_offset,
                    "data": tensor.numpy().tobytes(),
                })
                
                tensor_data_offset += len(tensor_infos[-1]["data"])
                
                logger.debug("Converted tensor %s → %s", name, gguf_name)
            
            # Write tensor metadata
            _write_tensor_metadata(out, tensor_infos)
            
            # Write tensor data
            for info in tensor_infos:
                out.write(info["data"])
    
    progress.status = "complete"
    progress.current_step = progress.total_steps
    update_progress()
    
    logger.info("GGUF file written: %s", output_path)
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
    """Get GGUF data type."""
    if quant == QuantizationType.F32:
        return GGMLType.F32
    elif quant == QuantizationType.F16:
        return GGMLType.F16
    elif quant == QuantizationType.Q8_0:
        return GGMLType.Q8_0
    elif quant == QuantizationType.Q5_K:
        return GGMLType.Q5_K
    elif quant == QuantizationType.Q4_K:
        return GGMLType.Q4_K
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


__all__ = [
    "ConversionConfig",
    "ConversionProgress", 
    "QuantizationType",
    "convert_safetensors_to_gguf",
]
