"""Qwen3-4B Text Encoder for Z Image.

Wrapper for Qwen3-4B model used as text encoder in Z Image Turbo.
Based on ComfyUI's comfy/text_encoders/z_image.py.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger("backend.runtime.zimage.text_encoder")


# Chat template for Qwen3 (matches ComfyUI z_image.py)
QWEN3_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class ZImageTextEncoder(nn.Module):
    """Text encoder for Z Image using Qwen3-4B.
    
    Wraps a Qwen3-4B model for text encoding. The model can be:
    - A safetensors checkpoint (qwen_3_4b.safetensors)
    - A GGUF quantized model (Qwen3-4B-Q8_0.gguf)
    - An FP8 quantized model (qwen3_4b_fp8_scaled.safetensors)
    
    The model uses a chat template format for text encoding.
    """
    
    def __init__(
        self,
        qwen_model: nn.Module,
        hidden_size: int = 2560,
        layer_idx: int = -2,
    ):
        """Initialize the text encoder wrapper.
        
        Args:
            qwen_model: The underlying Qwen3-4B model.
            hidden_size: Hidden dimension of the model (2560 for Qwen3-4B).
            layer_idx: Which hidden layer to use (-2 = second-to-last).
        """
        super().__init__()
        self.model = qwen_model
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self._tokenizer = None
    
    @classmethod
    def from_gguf(cls, gguf_path: str, torch_dtype: torch.dtype = torch.bfloat16) -> "ZImageTextEncoder":
        """Load text encoder from GGUF file using native Qwen3_4B.
        
        Args:
            gguf_path: Path to the GGUF file.
            torch_dtype: Target dtype for the model.
        
        Returns:
            ZImageTextEncoder instance.
        """
        logger.info("Loading Qwen3 text encoder from GGUF: %s", gguf_path)
        
        # Import native Qwen3 implementation
        from .qwen3 import Qwen3_4B, Qwen3Config, remap_gguf_keys
        
        # Load GGUF state dict using our infrastructure
        from apps.backend.opus_quantization.gguf_loader import load_gguf_state_dict
        from apps.backend.runtime.ops.operations import using_codex_operations
        
        try:
            # Load GGUF file
            logger.info("Loading GGUF state dict...")
            gguf_state_dict = load_gguf_state_dict(gguf_path)
            logger.info("Loaded %d tensors from GGUF", len(gguf_state_dict))
            
            # Remap keys from GGUF format to our model format
            logger.info("Remapping GGUF keys to native format...")
            state_dict = remap_gguf_keys(gguf_state_dict, num_layers=36)
            logger.info("Remapped to %d native keys", len(state_dict))
            
            # Use Codex operations context to enable GGUF tensor support
            # This patches torch.nn.Linear to CodexOperationsGGUF.Linear which handles ParameterGGUF
            with using_codex_operations(bnb_dtype='gguf', manual_cast_enabled=True, device=None, dtype=torch_dtype):
                # Create model with native Qwen3_4B inside the context
                config = Qwen3Config()
                model = Qwen3_4B(config, dtype=torch_dtype)
                
                # DEBUG: Verify patch application
                logger.info("DEBUG: Model embed_tokens type: %s", type(model.model.embed_tokens))
                logger.info("DEBUG: Model q_proj type: %s", type(model.model.layers[0].self_attn.q_proj))
                
                # DEBUG: Inspect state dict entry for embedding
                emb_key = "model.embed_tokens.weight"
                if emb_key in state_dict:
                    emb_tensor = state_dict[emb_key]
                    logger.info("DEBUG: State dict %s type: %s", emb_key, type(emb_tensor))
                    logger.info("DEBUG: State dict %s shape: %s", emb_key, emb_tensor.shape)
                    if hasattr(emb_tensor, 'real_shape'):
                        logger.info("DEBUG: State dict %s real_shape: %s", emb_key, emb_tensor.real_shape)

                # Load state dict - patched layers will handle GGUF tensors correctly
                try:
                    missing, unexpected = model.load_sd(state_dict)
                    if missing:
                        logger.warning("Missing keys: %s", missing[:10])
                    if unexpected:
                        logger.debug("Unexpected keys: %s", unexpected[:10])
                except RuntimeError as e:
                    logger.error("DEBUG: RuntimeError during load_sd: %s", e)
                    # Dump model shape for comparison
                    logger.error("DEBUG: Model embedding weight shape: %s", model.model.embed_tokens.weight.shape)
                    raise
                
                model = model.to(dtype=torch_dtype)
            
            param_count = sum(p.numel() for p in model.parameters())
            logger.info("GGUF text encoder loaded: %d params (%.2fB)", 
                       param_count, param_count / 1e9)
            
            return cls(model, hidden_size=2560, layer_idx=-2)
            
        except ImportError as e:
            logger.error("GGUF loader/ops not available: %s", e)
            raise ValueError(f"GGUF loader/ops not available: {e}")
        except Exception as e:
            logger.error("Failed to load GGUF text encoder: %s", e)
            raise ValueError(f"Failed to load GGUF text encoder from {gguf_path}: {e}")
    
    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor], torch_dtype: torch.dtype = torch.bfloat16) -> "ZImageTextEncoder":
        """Load text encoder from state_dict.
        
        Args:
            state_dict: State dict with model weights.
            torch_dtype: Target dtype for the model.
        
        Returns:
            ZImageTextEncoder instance.
        """
        logger.info("Loading Qwen3 text encoder from state_dict (%d keys)", len(state_dict))
        
        try:
            # Use native Qwen3_4B implementation (compatible with ComfyUI format)
            from .qwen3 import Qwen3_4B, Qwen3Config
            
            config = Qwen3Config()
            model = Qwen3_4B(config, dtype=torch_dtype)
            
            # Load weights - native implementation has compatible key format
            missing, unexpected = model.load_sd(state_dict)
            if missing:
                logger.warning("Missing keys: %s", missing[:10])
            if unexpected:
                logger.debug("Unexpected keys: %s", unexpected[:10])
            
            # Move to target dtype
            model.to(dtype=torch_dtype)
            
            encoder = cls(model, hidden_size=2560, layer_idx=-2)
            logger.info("Safetensors text encoder loaded successfully")
            return encoder
            
        except Exception as e:
            logger.error("Failed to load text encoder from state_dict: %s", e)
            raise ValueError(f"Failed to load text encoder from state_dict: {e}")
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model parameters."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None):
        """Load the Qwen tokenizer.
        
        Args:
            tokenizer_path: Path to tokenizer files. If None, uses default.
        """
        try:
            from transformers import Qwen2Tokenizer
            
            if tokenizer_path is None:
                # Try to find tokenizer in common locations
                tokenizer_path = os.path.join(
                    os.path.dirname(__file__),
                    "..", "..", "text_processing", "tokenizers", "qwen25_tokenizer"
                )
            
            if os.path.exists(tokenizer_path):
                self._tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to HuggingFace hub
                self._tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-3B")
            
            logger.info("Qwen tokenizer loaded successfully")
            
        except ImportError:
            logger.error("transformers library required for Qwen tokenizer")
            raise
    
    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
        apply_template: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with optional chat template.
        
        Args:
            texts: List of text prompts.
            max_length: Maximum token length.
            apply_template: Whether to apply the chat template.
        
        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        if self._tokenizer is None:
            self.load_tokenizer()
        
        if apply_template:
            texts = [QWEN3_TEMPLATE.format(t) for t in texts]
        
        tokens = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
        apply_template: bool = True,
    ) -> torch.Tensor:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text prompts.
            max_length: Maximum token length.
            apply_template: Whether to apply the chat template.
        
        Returns:
            Text embeddings [B, L, hidden_size].
        """
        tokens = self.tokenize(texts, max_length, apply_template)
        
        # Check if this is our native Qwen3_4B model or a HuggingFace model
        # Native Qwen3_4B returns (hidden_states, intermediate) tuple
        # HuggingFace models return an object with .hidden_states attribute
        is_native_model = hasattr(self.model, 'load_sd')  # Native Qwen3_4B has load_sd method
        
        if is_native_model:
            # Native Qwen3_4B: use intermediate_output to get second-to-last layer
            hidden_states, intermediate = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                intermediate_output=self.layer_idx,  # -2 = second-to-last layer
                final_layer_norm_intermediate=True,
            )
            # intermediate contains the second-to-last layer output (with final norm applied)
            hidden = intermediate if intermediate is not None else hidden_states
        else:
            # HuggingFace model: use output_hidden_states
            outputs = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )
            # Use second-to-last layer (like CLIP)
            hidden = outputs.hidden_states[self.layer_idx]
        
        return hidden.to(self.dtype)
    
    def forward(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(texts, max_length)


class ZImageTextProcessingEngine:
    """Text processing engine for Z Image (matches Flux pattern).
    
    This class wraps the text encoder and provides a consistent interface
    for text encoding across different model families.
    """
    
    def __init__(
        self,
        text_encoder: ZImageTextEncoder,
        emphasis_name: str = "Original",
        max_length: int = 512,
    ):
        """Initialize the text processing engine.
        
        Args:
            text_encoder: The Z Image text encoder.
            emphasis_name: Name of emphasis style (unused, for compatibility).
            max_length: Maximum token length.
        """
        self.text_encoder = text_encoder
        self.emphasis_name = emphasis_name
        self.max_length = max_length
    
    def __call__(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        return self.text_encoder.encode(texts, self.max_length)
    
    def tokenize(self, texts: List[str]) -> List[List[int]]:
        """Tokenize without encoding."""
        tokens = self.text_encoder.tokenize(texts, self.max_length)
        return tokens["input_ids"].tolist()


class Qwen3TextEncoder(ZImageTextEncoder):
    """Alias for backward compatibility."""
    pass
