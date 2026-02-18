"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima Qwen3-0.6B text encoder runtime + offline tokenizers (Qwen + T5).
Loads sha-selected Qwen3-0.6B weights through a strict Anima keymap into the native Qwen3 implementation and provides a small text-processing wrapper
that produces embeddings for conditioning. Also loads an offline T5 tokenizer used only for dual-tokenization (token ids + weights).

No runtime downloads are allowed: tokenizers must be resolved from vendored `apps/backend/huggingface/**` assets or explicit paths.

Symbols (top-level; keep in sync; no ghosts):
- `AnimaQwenTextEncoder` (class): Qwen3-0.6B encoder wrapper (native model + tokenizer).
- `AnimaQwenTextProcessingEngine` (class): Thin adapter exposing `__call__`, `tokenize`, and `tokenize_with_weights`.
- `load_anima_qwen3_06b_text_encoder` (function): Strict loader for Qwen3-0.6B weights (safetensors; sha-selected).
- `load_anima_t5_tokenizer` (function): Offline T5 tokenizer loader (used for Anima dual-tokenization ids/weights).
- `tokenize_qwen_with_weights` (function): Qwen tokenization helper with optional `return_word_ids` metadata parity.
- `tokenize_t5_with_weights` (function): Offline T5 tokenization producing ids+weights tensors for Anima conditioning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.ops.operations import using_codex_operations
from apps.backend.runtime.state_dict.keymap_anima import remap_anima_qwen3_06b_state_dict
from apps.backend.runtime.text_processing.parsing import parse_prompt_attention

logger = logging.getLogger("backend.runtime.anima.text_encoder")


def _resolve_dir_candidates(*, env_var: str, explicit: str | None, candidates: Iterable[Path]) -> list[Path]:
    repo_root = get_repo_root()
    out: list[Path] = []

    env_value = os.getenv(env_var)
    if env_value:
        out.append(Path(os.path.expanduser(env_value.strip())))
    if explicit:
        out.insert(0, Path(os.path.expanduser(str(explicit).strip())))

    out.extend(candidates)

    normalized: list[Path] = []
    for p in out:
        raw = str(p).strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        try:
            path = path.resolve()
        except Exception:
            path = path.absolute()
        normalized.append(path)
    return normalized


def _load_tokenizer_dir(*, env_var: str, explicit: str | None, candidates: Iterable[Path]) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"transformers is required to load tokenizers ({env_var}).") from exc

    tried: list[str] = []
    errors: list[str] = []
    for p in _resolve_dir_candidates(env_var=env_var, explicit=explicit, candidates=candidates):
        tried.append(str(p))
        if not p.exists() or not p.is_dir():
            continue
        try:
            tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True, use_fast=True)
            logger.info("Loaded tokenizer from %s (env=%s)", p, env_var)
            return tok
        except Exception as exc:  # noqa: BLE001 - try next candidate
            errors.append(f"{p}: {type(exc).__name__}: {exc}")

    detail = "\n".join(errors) if errors else "<no load errors captured>"
    raise RuntimeError(
        f"Failed to load an offline tokenizer for {env_var}. "
        f"Set {env_var} or vendor the tokenizer under apps/backend/huggingface. "
        f"Tried: {tried}\nErrors:\n{detail}"
    )


def _clean_prompt_text(text: str, *, emphasis_name: str = "Original") -> str:
    parsed = parse_prompt_attention(str(text or ""), emphasis_name)
    out: list[str] = []
    for seg, weight in parsed:
        if seg == "BREAK" and weight == -1:
            # We do not support multi-section conditioning for Anima v1. Treat BREAK as whitespace.
            out.append(" ")
            continue
        out.append(str(seg))
    return "".join(out)


def _default_qwen_tokenizer_candidates() -> list[Path]:
    repo_root = get_repo_root()
    return [
        repo_root / "apps" / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "qwen25_tokenizer",
        repo_root / "apps" / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image-Turbo" / "tokenizer",
        repo_root / "apps" / "backend" / "huggingface" / "Tongyi-MAI" / "Z-Image" / "tokenizer",
    ]


def _default_t5_tokenizer_candidates() -> list[Path]:
    repo_root = get_repo_root()
    # Prefer Flux T5 tokenizer mirror (tokenizer_2).
    return [
        repo_root / "apps" / "backend" / "huggingface" / "circlestone-labs" / "Anima" / "t5_tokenizer",
        repo_root / "apps" / "backend" / "huggingface" / "black-forest-labs" / "FLUX.1-dev" / "tokenizer_2",
        repo_root / "apps" / "backend" / "huggingface" / "black-forest-labs" / "FLUX.1-Kontext-dev" / "tokenizer_2",
        repo_root / "apps" / "backend" / "huggingface" / "black-forest-labs" / "FLUX.1-schnell" / "tokenizer_2",
    ]


@dataclass(frozen=True, slots=True)
class _QwenTokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


QwenWeightedToken = tuple[int, float] | tuple[int, float, int]


def _normalize_qwen_weighted_token_entry(entry: object, *, return_word_ids: bool) -> QwenWeightedToken:
    if not isinstance(entry, (tuple, list)):
        raise RuntimeError(
            "Qwen weighted token entry must be tuple/list; "
            f"got {type(entry).__name__}."
        )
    if len(entry) < 2:
        raise RuntimeError(
            "Qwen weighted token entry must contain at least (token_id, weight). "
            f"Got len={len(entry)} entry={entry!r}."
        )

    try:
        token_id = int(entry[0])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-int token_id: {entry[0]!r}.") from exc
    try:
        weight = float(entry[1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-float weight: {entry[1]!r}.") from exc

    if not return_word_ids:
        return (token_id, weight)

    if len(entry) < 3:
        raise RuntimeError(
            "Qwen weighted token entry is missing word_id while return_word_ids=True. "
            f"entry={entry!r}"
        )
    try:
        word_id = int(entry[2])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Qwen weighted token has non-int word_id: {entry[2]!r}.") from exc
    return (token_id, weight, word_id)


def _resolve_pad_token_id_for_qwen(*, tokenizer: Any, context: str) -> int:
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError(f"{context}: tokenizer missing pad_token_id.")
    try:
        pad_id_int = int(pad_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{context}: tokenizer pad_token_id must be int; got {pad_id!r}.") from exc
    token_count: int | None = None
    try:
        token_count = int(len(tokenizer))
    except Exception:
        token_count = None
    if token_count is not None and not (0 <= pad_id_int < token_count):
        raise RuntimeError(
            f"{context}: tokenizer pad_token_id out of range: "
            f"pad_token_id={pad_id_int} len={token_count}."
        )
    return pad_id_int


def tokenize_qwen_with_weights(
    *,
    tokenizer: Any,
    texts: list[str],
    emphasis_name: str = "Original",
    max_length: int = 512,
    return_word_ids: bool = False,
) -> list[list[QwenWeightedToken]]:
    """Tokenize Qwen prompts into weighted tuples, optionally preserving word-id metadata.

    Notes:
    - `word_id` is assigned per parsed emphasis segment from `parse_prompt_attention`.
    - Tokens emitted by the same parsed segment share the same `word_id`.
    """
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    out: list[list[QwenWeightedToken]] = []
    for raw in texts:
        parsed = parse_prompt_attention(str(raw or ""), emphasis_name)
        row: list[QwenWeightedToken] = []
        word_id = 1

        for seg, weight in parsed:
            if seg == "BREAK" and weight == -1:
                seg = " "
                weight = 1.0
            tokenized = tokenizer(
                [str(seg)],
                padding=False,
                truncation=False,
                add_special_tokens=False,
                verbose=False,
            )
            seg_ids = tokenized.get("input_ids")
            if not (isinstance(seg_ids, list) and seg_ids and isinstance(seg_ids[0], list)):
                raise RuntimeError("Qwen tokenizer did not return input_ids as list[list[int]].")
            for token_id in seg_ids[0]:
                raw_entry = (token_id, weight, word_id) if return_word_ids else (token_id, weight)
                row.append(
                    _normalize_qwen_weighted_token_entry(
                        raw_entry,
                        return_word_ids=return_word_ids,
                    )
                )
            if seg_ids[0]:
                word_id += 1

        if not row:
            pad_id = _resolve_pad_token_id_for_qwen(
                tokenizer=tokenizer,
                context="Anima Qwen weighted tokenization produced an empty sequence",
            )
            fallback = (pad_id, 1.0, 0) if return_word_ids else (pad_id, 1.0)
            row.append(_normalize_qwen_weighted_token_entry(fallback, return_word_ids=return_word_ids))

        if len(row) > max_length:
            raise ValueError(
                "Anima Qwen tokenization exceeded max_length=%d (len=%d). "
                "Reduce prompt length or increase CODEX_ANIMA_QWEN_MAX_LENGTH."
                % (max_length, len(row))
            )
        out.append(row)
    return out


class AnimaQwenTextEncoder(nn.Module):
    """Qwen3-0.6B text encoder (native)."""

    def __init__(self, *, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._tokenizer: Any | None = None
        self._tokenizer_path_hint: str | None = None

    def set_tokenizer_path_hint(self, tokenizer_path: str | None) -> None:
        value = str(tokenizer_path).strip() if tokenizer_path is not None else ""
        self._tokenizer_path_hint = value or None

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        hint = self._tokenizer_path_hint
        tok = _load_tokenizer_dir(
            env_var="CODEX_ANIMA_QWEN_TOKENIZER_PATH",
            explicit=hint,
            candidates=_default_qwen_tokenizer_candidates(),
        )
        self._tokenizer = tok
        return tok

    def tokenize(self, texts: list[str], *, max_length: int) -> _QwenTokenBatch:
        tok = self._require_tokenizer()
        cleaned = [_clean_prompt_text(t) for t in texts]

        # Note: keep add_special_tokens=False to match ComfyUI's SDTokenizer path (no start/end token).
        # Fail loud if the prompt exceeds max_length (no silent truncation).
        probe = tok(
            cleaned,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            verbose=False,
        )
        probe_ids = probe.get("input_ids")
        if isinstance(probe_ids, list):
            too_long = [i for i, seq in enumerate(probe_ids) if isinstance(seq, list) and len(seq) > int(max_length)]
            if too_long:
                raise ValueError(
                    "Anima Qwen tokenizer prompt is too long for max_length=%d (indices=%s). "
                    "Reduce prompt length or increase CODEX_ANIMA_QWEN_MAX_LENGTH."
                    % (int(max_length), too_long)
                )

        encoded = tok(
            cleaned,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise RuntimeError("Tokenizer did not return input_ids/attention_mask tensors.")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise RuntimeError(
                "Qwen tokenizer returned invalid tensor ranks: "
                f"input_ids.ndim={input_ids.ndim}, attention_mask.ndim={attention_mask.ndim}."
            )
        if input_ids.shape != attention_mask.shape:
            raise RuntimeError(
                "Qwen tokenizer returned mismatched tensor shapes: "
                f"input_ids.shape={tuple(input_ids.shape)} attention_mask.shape={tuple(attention_mask.shape)}."
            )
        if input_ids.shape[0] != len(cleaned):
            raise RuntimeError(
                "Qwen tokenizer returned unexpected batch size: "
                f"got={int(input_ids.shape[0])} expected={len(cleaned)}."
            )

        if input_ids.shape[1] == 0:
            pad_id = _resolve_pad_token_id_for_qwen(
                tokenizer=tok,
                context="Anima Qwen tokenizer produced an empty sequence for prompt(s)",
            )
            batch_size = int(input_ids.shape[0])
            input_ids = torch.full((batch_size, 1), fill_value=pad_id, dtype=input_ids.dtype, device=input_ids.device)
            attention_mask = torch.zeros((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
            logger.debug(
                "Qwen tokenizer produced empty sequence; synthesized masked pad token for min_length=1 "
                "(batch_size=%d, pad_token_id=%d).",
                batch_size,
                pad_id,
            )

        # Fail loud on truncation (transformers reports it via tokenizer warnings in some cases; we enforce via length).
        if input_ids.shape[1] >= int(max_length):
            # Best-effort check: if any row ends with a non-pad token while attention_mask indicates padding,
            # it likely truncated. We keep the message actionable rather than guessing.
            logger.warning(
                "Qwen tokenizer hit max_length=%d (seq_len=%d). Prompt may have been truncated.",
                int(max_length),
                int(input_ids.shape[1]),
            )
        return _QwenTokenBatch(input_ids=input_ids, attention_mask=attention_mask)

    def tokenize_with_weights(
        self,
        texts: list[str],
        *,
        max_length: int,
        emphasis_name: str = "Original",
        return_word_ids: bool = False,
    ) -> list[list[QwenWeightedToken]]:
        tok = self._require_tokenizer()
        return tokenize_qwen_with_weights(
            tokenizer=tok,
            texts=texts,
            emphasis_name=emphasis_name,
            max_length=max_length,
            return_word_ids=return_word_ids,
        )

    @torch.no_grad()
    def encode(self, texts: list[str], *, max_length: int) -> torch.Tensor:
        batch = self.tokenize(texts, max_length=max_length)
        input_ids = batch.input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = batch.attention_mask.to(device=self.device, dtype=torch.long)

        hidden, _intermediate = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError(f"Qwen3 model returned invalid hidden states: {type(hidden).__name__} shape={getattr(hidden,'shape',None)}")
        return hidden.to(dtype=self.dtype)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return torch.float32


class AnimaQwenTextProcessingEngine:
    """Thin adapter providing a consistent callable interface around `AnimaQwenTextEncoder`."""

    def __init__(self, text_encoder: AnimaQwenTextEncoder, *, max_length: int = 512) -> None:
        self.text_encoder = text_encoder
        self.max_length = int(max_length)

    def __call__(self, texts: list[str]) -> torch.Tensor:
        return self.text_encoder.encode(texts, max_length=self.max_length)

    def tokenize(self, texts: list[str]) -> list[list[int]]:
        batch = self.text_encoder.tokenize(texts, max_length=self.max_length)
        return batch.input_ids.tolist()

    def tokenize_with_weights(
        self,
        texts: list[str],
        *,
        emphasis_name: str = "Original",
        return_word_ids: bool = False,
    ) -> list[list[QwenWeightedToken]]:
        return self.text_encoder.tokenize_with_weights(
            texts,
            max_length=self.max_length,
            emphasis_name=emphasis_name,
            return_word_ids=return_word_ids,
        )


def _validate_qwen3_06b_header(*, header: Mapping[str, object], context: str) -> None:
    def _shape(key: str) -> tuple[int, ...] | None:
        meta = header.get(key)
        if isinstance(meta, dict):
            shape = meta.get("shape")
            if isinstance(shape, (list, tuple)) and all(isinstance(x, (int, float)) for x in shape):
                return tuple(int(x) for x in shape)
        return None

    embed = _shape("model.embed_tokens.weight")
    if embed is None or len(embed) != 2:
        raise RuntimeError(f"Qwen3-0.6B header missing model.embed_tokens.weight shape: {context}")
    vocab, hidden = int(embed[0]), int(embed[1])
    if hidden != 1024:
        raise RuntimeError(f"Qwen3-0.6B embed dim mismatch for {context}: got {hidden}, expected 1024.")
    if vocab <= 0:
        raise RuntimeError(f"Qwen3-0.6B vocab_size invalid for {context}: {vocab}.")

    # Spot-check a few keys so wrong checkpoints fail loudly with a useful message.
    required = (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    )
    missing = [k for k in required if k not in header]
    if missing:
        raise RuntimeError(
            "Qwen3-0.6B weights file does not look like a Qwen3 HF-style checkpoint. "
            f"Missing keys: {missing} ({context})"
        )


def load_anima_qwen3_06b_text_encoder(
    tenc_path: str,
    *,
    torch_dtype: torch.dtype,
) -> AnimaQwenTextEncoder:
    raw = str(tenc_path or "").strip()
    if not raw:
        raise ValueError("Anima Qwen3-0.6B text encoder path is required.")
    p = Path(os.path.expanduser(raw))
    try:
        p = p.resolve()
    except Exception:
        p = p.absolute()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Anima Qwen3-0.6B text encoder path not found: {p}")
    if p.suffix.lower() not in {".safetensor", ".safetensors"}:
        raise ValueError(f"Anima Qwen3-0.6B text encoder must be a .safetensors file, got: {p}")

    header = read_safetensors_header(p)
    _validate_qwen3_06b_header(header=header, context=str(p))

    from apps.backend.runtime.families.zimage.qwen3 import Qwen3_06B

    sd = load_torch_file(str(p), device="cpu")
    if not isinstance(sd, Mapping):
        raise RuntimeError(f"Anima Qwen3-0.6B loader returned non-mapping state_dict: {type(sd).__name__}")
    sd = {str(k): v for k, v in sd.items()}
    try:
        _, sd = remap_anima_qwen3_06b_state_dict(sd)
    except Exception as exc:  # noqa: BLE001 - surfaced as a load-time error with context
        raise RuntimeError(f"Anima Qwen3-0.6B key remap failed: {exc}") from exc

    with using_codex_operations(device=None, dtype=torch_dtype, manual_cast_enabled=True):
        model = Qwen3_06B(dtype=torch_dtype)
    missing, unexpected = safe_load_state_dict(model, sd, log_name="anima.qwen3_06b")
    if missing or unexpected:
        raise RuntimeError(
            "Anima Qwen3-0.6B strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )

    model.eval()
    model.to(dtype=torch_dtype)
    return AnimaQwenTextEncoder(model=model)


def load_anima_t5_tokenizer(tokenizer_path: str | None = None) -> Any:
    tok = _load_tokenizer_dir(
        env_var="CODEX_ANIMA_T5_TOKENIZER_PATH",
        explicit=tokenizer_path,
        candidates=_default_t5_tokenizer_candidates(),
    )

    token_count: int | None = None
    try:
        token_count = int(len(tok))
    except Exception:
        token_count = None
    max_tokens = 32128
    if token_count is not None and token_count > max_tokens:
        raise RuntimeError(f"Anima T5 tokenizer is too large: got len={token_count}, expected <= {max_tokens}.")

    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError("Anima T5 tokenizer missing pad_token_id.")
    try:
        pad_id = int(pad_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Anima T5 tokenizer pad_token_id must be an int; got {pad_id!r}.") from exc
    if token_count is not None and not (0 <= pad_id < token_count):
        raise RuntimeError(f"Anima T5 tokenizer pad_token_id out of range: pad_token_id={pad_id} len={token_count}.")

    return tok


def tokenize_t5_with_weights(
    *,
    tokenizer: Any,
    texts: list[str],
    emphasis_name: str = "Original",
    max_length: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts into `(input_ids, weights)` for the Anima adapter (T5 tokenizer only; no text encoder)."""
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError("T5 tokenizer missing pad_token_id")
    pad_id = int(pad_id)

    all_ids: list[list[int]] = []
    all_weights: list[list[float]] = []

    for raw in texts:
        parsed = parse_prompt_attention(str(raw or ""), emphasis_name)
        ids: list[int] = []
        weights: list[float] = []
        for seg, weight in parsed:
            if seg == "BREAK" and weight == -1:
                # See _clean_prompt_text; treat BREAK as whitespace.
                seg = " "
                weight = 1.0
            tokenized = tokenizer(
                [str(seg)],
                padding=False,
                truncation=False,
                add_special_tokens=False,
                verbose=False,
            )
            seg_ids = tokenized.get("input_ids")
            if not (isinstance(seg_ids, list) and seg_ids and isinstance(seg_ids[0], list)):
                raise RuntimeError("T5 tokenizer did not return input_ids as list[list[int]].")
            for tid in seg_ids[0]:
                ids.append(int(tid))
                weights.append(float(weight))
        if not ids:
            ids = [pad_id]
            weights = [1.0]
        if len(ids) > max_length:
            raise ValueError(
                "Anima T5 tokenization exceeded max_length=%d (len=%d). Reduce prompt length or increase CODEX_ANIMA_T5_MAX_LENGTH."
                % (max_length, len(ids))
            )
        all_ids.append(ids)
        all_weights.append(weights)

    batch = len(all_ids)
    max_len = max(len(x) for x in all_ids) if all_ids else 1

    ids_out = torch.full((batch, max_len), pad_id, dtype=torch.long)
    w_out = torch.ones((batch, max_len), dtype=torch.float32)
    for i, (ids, w) in enumerate(zip(all_ids, all_weights, strict=True)):
        ids_out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        w_out[i, : len(w)] = torch.tensor(w, dtype=torch.float32)

    return ids_out, w_out


__all__ = [
    "AnimaQwenTextEncoder",
    "AnimaQwenTextProcessingEngine",
    "load_anima_qwen3_06b_text_encoder",
    "load_anima_t5_tokenizer",
    "tokenize_qwen_with_weights",
    "tokenize_t5_with_weights",
]
