# OpusQuantization integration - replaces vendored Forge GGUF kernels
# Now just redirects to apps.backend.opus_quantization.gguf_ops

from apps.backend.opus_quantization.gguf_ops import (
    ParameterGGUF,
    dequantize_tensor,
    quants_mapping,
    set_cache_policy,
    clear_cache,
    _get_opus_bridge,
)

# For backward compatibility if anything imports from here
__all__ = [
    'ParameterGGUF',
    'dequantize_tensor',
    'quants_mapping',
    'set_cache_policy',
    'clear_cache',
]
