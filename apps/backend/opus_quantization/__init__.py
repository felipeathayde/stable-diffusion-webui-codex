# OpusQuantization - GGUF Quantization Module
# Named in honor of collaborative development

# Import kernels first to register them
from . import kernels  # noqa: F401 - triggers registration

from .core import QuantType, BLOCK_SIZES, register_quant, get_quant_spec
from .tensor import OpusParameter
from .api import dequantize, bake, quantize

__all__ = [
    # Types
    "QuantType",
    "BLOCK_SIZES",
    "OpusParameter",
    # Functions
    "dequantize",
    "bake",
    "quantize",
    # Registry
    "register_quant",
    "get_quant_spec",
]

__version__ = "2.0.0"  # Major version bump: ComfyUI-GGUF port
