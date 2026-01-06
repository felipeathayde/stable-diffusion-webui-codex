"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Text processing engines and helpers for backend runtime.
Re-exports text processing engines (CLIP/T5), prompt parsing/emphasis utilities, and textual inversion helpers used by engines and workflows.

Symbols (top-level; keep in sync; no ghosts):
- `ClassicTextProcessingEngine` (class): CLIP-based prompt encoder (chunking, emphasis, textual inversion integration).
- `PromptChunkFix` (namedtuple): Single embedding “fix” applied at a specific token offset (used by textual inversion).
- `last_extra_generation_params` (constant): Mutable dict carrying extra generation params (best-effort telemetry for downstream consumers).
- `T5TextProcessingEngine` (class): T5-based prompt encoder used by engines that rely on T5 text encoders.
- `emphasis` (module): Emphasis registry and implementations.
- `parsing` (module): Prompt attention parsing (`(...)`/`[...]` weights and BREAK tokens).
- `textual_inversion` (module): Textual inversion database and embedding helpers.
- `EmbeddingDatabase` (class): Textual inversion embeddings registry/database.
- `embedding_to_b64` (function): Serialize an embedding tensor to base64 for transport/storage.
- `embedding_from_b64` (function): Deserialize an embedding tensor from base64.
- `__all__` (constant): Export list for the text processing facade.
"""

from .classic_engine import ClassicTextProcessingEngine, PromptChunkFix, last_extra_generation_params
from .t5_engine import T5TextProcessingEngine
from . import emphasis, parsing, textual_inversion
from .textual_inversion import EmbeddingDatabase, embedding_to_b64, embedding_from_b64

__all__ = [
    "ClassicTextProcessingEngine",
    "EmbeddingDatabase",
    "PromptChunkFix",
    "T5TextProcessingEngine",
    "embedding_from_b64",
    "embedding_to_b64",
    "emphasis",
    "last_extra_generation_params",
    "parsing",
    "textual_inversion",
]
