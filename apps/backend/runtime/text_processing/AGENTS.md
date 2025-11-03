# Text Processing Runtime — AGENTS Notes
Date: 2025-11-03
Owner: Runtime Maintainers
Status: Active

## Scope
Applies to `apps/backend/runtime/text_processing/*` including `classic_engine.py`.

## Design Guidelines
- No legacy imports. Keep APIs Codex‑native, clear, and typed.
- Preserve A1111/Comfy behaviour (chunking, emphasis, TI) while modernising structure.
- Avoid hard dependencies on upstream internals that drift between HF releases.

## 2025-11-03 Update — CLIP TE Forward
- Removed reliance on `embeddings.position_ids` which is not guaranteed across HF CLIP variants.
- Generate `attention_mask` and `position_ids` per batch and pass only if the text encoder’s forward supports them (signature introspection).
- Keep embedding weights (`token_embedding`, `position_embedding`) in `float32` for numerical stability independent of the text‑encoder compute dtype.
- Emit actionable logs on precision fallback; never silence errors.

### Wrapper-first calling convention
- Classic engine now calls the text encoder via the wrapper (`self.text_encoder(...)`) instead of the inner transformer. The wrapper forwards to the Codex CLIP and harmonises the return shape with HF semantics.

## Invariants
- Token chunking is stable: BOS + chunk + EOS, padding with configured PAD id.
- Emphasis/Textual Inversion application remains before the final layer norm and after hidden state selection (`clip_skip`).

## Risks
- CLIP forward signatures vary across `transformers` versions; introspection guards the call but monitor logs for unusual masks being ignored.
- If future models drop `pooler_output`, the engine must gate pooled features accordingly.

## 2025-11-03 State-Dict Normalization (CLIP)
- The loader now normalizes CLIP state dicts that come rooted at `text_model.*` by lifting them under `transformer.*` so they match `IntegratedCLIP` keys. This avoids spurious `Missing/Unexpected` when checkpoints already use the modern `text_model.*` layout but omit the outer `transformer` prefix.

## Not Implemented
- Non‑classic emphasis variants outside the configured registry will raise `NotImplementedError` when wired.
