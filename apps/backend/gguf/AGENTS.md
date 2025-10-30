# apps/backend/gguf Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Houses utilities for working with GGUF-formatted checkpoints (metadata parsing, tensor mapping, quantization helpers) used by WAN22 and other GGUF-enabled engines.

## Key Files
- `gguf_reader.py` / `gguf_writer.py` — IO helpers for reading/writing GGUF tensors and metadata.
- `metadata.py` / `constants.py` / `tensor_mapping.py` — Schema definitions and tensor routing helpers.
- `quants/` package — Quantization registry (`registry.py`), helpers (`utils.py`), and kernel families.
- `quants/kernels/base/forge_*.py` — Forge-derived BF16/Q4/Q5/Q8 math preserved under MIT license headers.
- `quants/kernels/k_family/forge_*.py` — Forge-derived Q2_K–Q6_K families, imported verbatim per kernel with MIT attribution.
- `quants/kernels/iq_family/forge_*.py` — Forge-derived IQ family (IQ1/IQ2/IQ3/IQ4) kernels split per quantization variant with MIT attribution.
- `quick_4bits_ops.py` — Shared bit-packing helpers used by kernel bake paths.
- `lazy.py` / `utility.py` — Lazy-loading helpers and shared GGUF utilities.
- `vocab.py` — Vocabulary parsing for GGUF tokenizers.

## Notes
- Keep GGUF support centralized here so runtimes/engines can depend on a single implementation.
- When updating GGUF spec support, document changes and align with WAN22 runtime expectations.
- The quantization stack is moving to `QuantKernel` registry abstractions; new kernels should register through the shared helpers rather than introducing standalone classes.
- Forge-derived low-level kernels now live in `quants/kernels/base/forge_*.py` (base types) and `quants/kernels/k_family/forge_*.py` (Q2_K–Q6_K); keep them immutable except for upstream syncs and add new math in Codex-specific wrappers.
- IQ-family kernels follow the same split: wrappers in `kerns/base/__init__.py` delegate to `quants/kernels/iq_family/forge_*.py`; keep wrappers limited to registry glue/logging and surface TODOs via explicit exceptions.
- Developer harness `tools/gguf/compare_codex_forge.py` exercises Codex vs Forge dequantization; by default it covers the K-family with IQ support gated behind explicit `--types` selection while layouts are finalized.
- `tools/gguf/smoke_quant_cpu.py` offers a CPU-only smoke test that calls both NumPy and (quando disponível) PyTorch paths. Último run (2025-10-30):
  ```
  ~/.venv/bin/python tools/gguf/smoke_quant_cpu.py
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K -> numpy == torch == Forge (max_abs/max_rel = 0)
  IQ2_S -> numpy == Forge; torch ainda não exposto (N/A)

  ~/.venv/bin/python tools/gguf/smoke_quant_cpu.py --types IQ2_XXS IQ2_XS IQ2_S IQ3_XXS IQ3_S IQ1_S IQ1_M IQ4_NL IQ4_XS
  IQ* -> numpy == Forge (max_abs/max_rel = 0); torch ainda não implementado.
  ```
