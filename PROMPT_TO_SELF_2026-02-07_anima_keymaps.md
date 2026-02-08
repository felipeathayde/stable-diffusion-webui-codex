# Prompt to self — Anima keymaps (new clean session)

```text
You are working in the repo `stable-diffusion-webui-codex`.
Continue from:
- CWD: /home/lucas/work/stable-diffusion-webui-codex
- Branch: master
- Last commit: 24b347c02f943df7fac8af1089edee2fd771e25d
- Date (UTC-3): 2026-02-07T23:00:55-03:00

Objective (1 sentence)
- Implement strict, model-aware keymaps for Anima core/VAE/text-encoder loading, using real local model files and Comfy references (especially VAE) as semantic baseline.

State
- Done:
  - Confirmed current Anima loaders are strict/fail-loud and do not use broad key remap maps:
    - core loader: `apps/backend/runtime/families/anima/loader.py`
    - VAE loader: `apps/backend/runtime/families/anima/wan_vae.py`
    - tenc loader: `apps/backend/runtime/families/anima/text_encoder.py`
  - Confirmed parser only strips `net.` for Anima core component split: `apps/backend/runtime/model_parser/families/anima.py`.
- In progress:
  - No keymap implementation started yet (this session should start from discovery + design).
- Blocked / risks:
  - Real checkpoints may contain mixed key styles/legacy wrappers.
  - VAE mapping is highest risk and must be derived from real file keys + Comfy semantics, not guesswork.

Decisions / constraints (locked)
- Do not add permissive fallback aliases; unknown/unmapped keys must fail loud.
- Use explicit keymap modules/functions; deterministic behavior only.
- Inspect local model files from Windows mount path:
  - `/mnt/c/Users/lucas/OneDrive/Documentos/stable-diffusion-webui-codex/models`
- Use `.refs/ComfyUI` only as semantic reference; never import/copy code directly into `apps/**`.
- Keep active imports inside `apps.*`; no cross-repo runtime imports.
- Tests must live under `.sangoi/dev/tests/backend`.

Follow-up (ordered)
1. Inventory all Anima candidate files in `/mnt/c/.../models` and classify by component (core/vae/tenc) + key style.
2. Build a key-style report from headers/keys (no full tensor materialization when possible).
3. Compare style report against `.refs/ComfyUI` Anima/WAN VAE key expectations.
4. Design typed keymap seams (core, VAE, tenc) with strict validation and explicit unsupported-style errors.
5. Implement keymap modules + loader integration.
6. Add regression tests (golden remap cases + fail-loud unknown/missing/unexpected).
7. Run focused validation and update `.sangoi` plan/task-log/changelog.

Next immediate step (do this first)
- Build a factual key inventory from real model files under `/mnt/c` before writing any mapper.
Commands:
CODEX_ROOT="$(git rev-parse --show-toplevel)"
find /mnt/c/Users/lucas/OneDrive/Documentos/stable-diffusion-webui-codex/models -type f \( -name '*.safetensors' -o -name '*.gguf' -o -name '*.bin' -o -name '*.pt' \) | rg -i 'anima|qwen|vae|tenc|text|encoder' | sort
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY'
from pathlib import Path
from safetensors import safe_open
root = Path('/mnt/c/Users/lucas/OneDrive/Documentos/stable-diffusion-webui-codex/models')
for p in sorted(root.rglob('*.safetensors')):
    name = str(p).lower()
    if not any(k in name for k in ('anima','qwen','vae','tenc','text_encoder')):
        continue
    try:
        with safe_open(str(p), framework='pt', device='cpu') as f:
            keys = list(f.keys())
        print(f"\n=== {p} ===")
        print(f"count={len(keys)}")
        print('\n'.join(keys[:80]))
    except Exception as e:
        print(f"\n=== {p} ===")
        print(f"ERROR: {e}")
PY

Files
- Changed files (last relevant commit(s)):
  - `apps/backend/runtime/sampling_adapters/prediction.py`
  - `apps/backend/runtime/sampling/sigma_schedules.py`
  - `apps/backend/engines/anima/spec.py`
  - `.sangoi/dev/tests/backend/test_discrete_flow_comfy_simple_schedule_parity.py`
- Focus files to open first:
  - `apps/backend/runtime/model_parser/families/anima.py` — current split/alias contract (`net.` stripping and component registration).
  - `apps/backend/runtime/families/anima/loader.py` — core strict load seam where keymap hook should be explicit.
  - `apps/backend/runtime/families/anima/wan_vae.py` — VAE strict load seam (primary target for keymap).
  - `apps/backend/runtime/families/anima/text_encoder.py` — tenc strict load seam and tokenizer constraints.
  - `apps/backend/runtime/common/vae.py` — existing VAE remap usage patterns (SDXL) to reuse architecture.
  - `apps/backend/runtime/models/key_normalization.py` — established key remap/view patterns.

Validation (what “green” looks like)
- `PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m py_compile <new/changed keymap + loader files>`
  # expected: no syntax/import errors
- `PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_*keymap*.py`
  # expected: new keymap tests pass
- `PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_offline_tokenizers.py`
  # expected: no regression in Anima tokenizer/conditioning contracts
- `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff`
  # expected: OK_HEADER_CHANGED for touched `apps/**` files
- `bash .sangoi/.tools/link-check.sh .sangoi`
  # expected: no broken links

Known traps / gotchas
- Don’t materialize giant tensors unless needed; prefer header/key scan first.
- Anima core loader expects parser-stripped keys (no `net.`); preserve this invariant.
- WAN VAE 2.2 is currently unsupported in loader; keymap work must not silently bypass this guard.
- Keep fail-loud behavior for missing/unexpected keys after remap.

References (read before coding)
- `.sangoi/howto/PROMPT_GUIDE.md`
- `.sangoi/templates/PROMPT_TO_SELF_TEMPLATE.md`
- `.sangoi/handoffs/HANDOFF_2026-01-25-sdxl-vae-keymap.md`
- `.sangoi/handoffs/HANDOFF_2026-01-25-sdxl-clip-keymap.md`
- `.sangoi/task-logs/2026-02-08-anima-comfy-simple-sigma-ladder.md`
- `.refs/ComfyUI/comfy/text_encoders/anima.py`
- `.refs/ComfyUI/comfy/ldm/anima/model.py`
- `.refs/ComfyUI/comfy/ldm/wan/vae.py`
- `.refs/ComfyUI/comfy/model_base.py`
```
