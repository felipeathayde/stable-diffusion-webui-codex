# apps/backend/video/upscaling Overview
Date: 2026-02-27
Last Review: 2026-02-27
Status: Active

## Purpose
- Own fail-loud external upscaling runners for video post-processing stages.

## Key Files
- `seedvr2_cli.py` — SeedVR2 external CLI runner (PIL frames -> lossless ffmpeg intermediate -> SeedVR2 CLI -> validated output frames).
- `__init__.py` — Package marker (no facade exports).

## Notes
- Do not import code from `.refs/**` inside `apps/**`; run SeedVR2 via subprocess only.
- Runner must fail loud on missing git/venv/pip/requirements (when required), missing ffmpeg, missing CLI entrypoint, non-zero subprocess exit, and frame count/size mismatches.
- Runtime directories:
  - Repo: `CODEX_SEEDVR2_REPO_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2/repo`)
    - Default-path bootstrap only: clone from `CODEX_SEEDVR2_REPO_URL` (default `https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git`) when missing, then always enforce `CODEX_SEEDVR2_REPO_REF` (default `4490bd1f482e026674543386bb2a4d176da245b9`) on each run (checkout; fetch+checkout fallback when ref is not local).
    - Default-path bootstrap/ref enforcement is serialized with an interprocess lock file under `CODEX_ROOT/.uv/xdg-data/seedvr2/`.
  - Isolated runtime venv: `CODEX_ROOT/.uv/xdg-data/seedvr2/venv` (requirements installed from repo `requirements.txt`, stamp-gated by requirements hash + pinned ref).
    - Venv provisioning + requirements stamp writes are serialized with an interprocess lock file under `CODEX_ROOT/.uv/xdg-data/seedvr2/`.
  - Model cache: `CODEX_SEEDVR2_MODEL_DIR` (default `CODEX_ROOT/.uv/xdg-data/seedvr2`)
  - CUDA override: `CODEX_SEEDVR2_CUDA_DEVICE` sets the explicit visible CUDA device index used by SeedVR2 and has priority over component device-derived mapping.
  - Work dir: `CODEX_ROOT/.tmp/seedvr2/<run_id>/...`
