Backend Changes (Oct 2025)
==========================

Tokenizer Loading
- Fallback to fast/auto tokenizers when `merges.txt` is absent but `tokenizer.json` exists (Transformers ≥ 4.5x).
- Silences long-seq warnings if present.

Windows Guards
- CUDA/XPU stream warmups are disabled on Windows; honor `args.cuda_stream`.

Localization
- Missing `localizations/` dir no longer breaks startup.

Extra Networks / LoRA
- Ativação nativa pendente. Não há fallback para registries do legado; quando solicitado, o backend lança erro explícito (`NotImplementedError`).

Tools Relocation
- Maintenance scripts moved to `tools/` to avoid Gradio “scripts” loader import.
