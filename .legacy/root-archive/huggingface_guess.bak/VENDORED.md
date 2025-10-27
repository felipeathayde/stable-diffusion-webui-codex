This directory contains a vendored copy of `huggingface_guess`.

Upstream repository:
- URL: https://github.com/lllyasviel/huggingface_guess
- Commit: 70942022b6bcd17d941c1b4172804175758618e2 (2024-10-29)

License:
- The upstream project is licensed under GPL-3.0. Keep this notice when updating.

Files included (from upstream):
- __init__.py
- detection.py
- diffusers_convert.py
- utils.py
- latent.py
- model_list.py (trimmed header comment; core classes preserved)

Notes:
- If you need additional model classes or features, re‑sync from upstream.
- Local import sites use `import huggingface_guess` directly; the local package
  takes precedence over pip/clone paths.

