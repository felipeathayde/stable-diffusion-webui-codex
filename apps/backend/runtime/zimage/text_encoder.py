"""Qwen3 Text Encoder for Z Image.

Wrapper for Qwen3 4B model used as text encoder in Z Image Turbo.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("backend.runtime.zimage.text_encoder")


# Chat template for Qwen3 (matches ComfyUI z_image.py)
QWEN3_TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Qwen3TextEncoder(nn.Module):
    """Wrapper for Qwen3 4B text encoder.
    
    Uses HuggingFace transformers Qwen2 model internally.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = 2560  # Qwen3 4B hidden size
        
        self._model = None
        self._tokenizer = None
        self._model_path = model_path
    
    def load_model(self, model_path: Optional[str] = None):
        """Load the underlying Qwen3 model."""
        path = model_path or self._model_path
        if path is None:
            raise ValueError("model_path required to load Qwen3")
        
        try:
            from transformers import Qwen2TokenizerFast, Qwen2Model
            
            logger.info("Loading Qwen3 4B from %s", path)
            self._tokenizer = Qwen2TokenizerFast.from_pretrained(path)
            self._model = Qwen2Model.from_pretrained(
                path,
                torch_dtype=self.dtype,
            ).to(self.device)
            self._model.eval()
            logger.info("Qwen3 4B loaded successfully")
            
        except ImportError:
            logger.error("transformers library required for Qwen3")
            raise
    
    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Qwen3 model not loaded. Call load_model() first.")
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Qwen3 tokenizer not loaded. Call load_model() first.")
        return self._tokenizer
    
    def tokenize(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> dict:
        """Tokenize text with chat template."""
        templated = [QWEN3_TEMPLATE.format(t) for t in texts]
        
        tokens = self.tokenizer(
            templated,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }
    
    @torch.inference_mode()
    def encode(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Encode texts to embeddings.
        
        Args:
            texts: List of text prompts.
            max_length: Maximum token length.
        
        Returns:
            Text embeddings [B, L, hidden_size].
        """
        tokens = self.tokenize(texts, max_length)
        
        outputs = self.model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            output_hidden_states=True,
        )
        
        # Use second-to-last layer (like CLIP)
        hidden = outputs.hidden_states[-2]
        
        return hidden.to(self.dtype)
    
    def forward(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Forward pass (alias for encode)."""
        return self.encode(texts, max_length)


class ZImageTextProcessingEngine:
    """Text processing engine for Z Image (matches Flux pattern)."""
    
    def __init__(
        self,
        text_encoder: Qwen3TextEncoder,
        emphasis_name: str = "Original",
        max_length: int = 512,
    ):
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
