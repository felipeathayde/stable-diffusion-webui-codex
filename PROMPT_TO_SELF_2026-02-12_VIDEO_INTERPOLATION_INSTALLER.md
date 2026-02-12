# Prompt to self — clean-session resume (video interpolation installer)

```text
You are working in the repo `stable-diffusion-webui-codex`.
Continue from:
- CWD: /home/lucas/work/stable-diffusion-webui-codex
- Branch: master
- Last commit: 17ea092ed5bc22705c5840daa04ce50b9c36d7fb
- Date (UTC-3): 2026-02-12T09:36:06-03:00

Objective (1 sentence)
- Ship Option A (out-of-box) video interpolation/runtime provisioning: installer must provision ffmpeg/ffprobe and a deterministic RIFE path, so interpolation works without manual hidden deps.

State
- Done:
  - Confirmed runtime failure seam: `video interpolation requested but RIFE is unavailable in this environment`.
  - Confirmed `apps/backend/video/interpolation/rife.py` only tries external modules (`backend_ext.rife_vfi` / `rife_vfi`), and no in-repo implementation exists.
  - Confirmed installers (`install-webui.sh`, `install-webui.bat`) do not provision ffmpeg/ffprobe or RIFE runtime/model.
  - Verified `ffmpeg-downloader` can install static binaries under repo-local `.uv/xdg-data/ffmpeg-downloader/ffmpeg`.
  - Created plan draft in `.sangoi` docs repo: `.sangoi/plans/2026-02-12-video-interpolation-deps-installer.md` (currently uncommitted there).
- In progress:
  - Implementing Option A wiring across backend + installer + deps + tests.
- Blocked / risks:
  - Need final ckpt contract for RIFE model artifact (UI defaults to `rife47.pth`, but repo currently has no model file).
  - Candidate package `ccvfi` works as adapter candidate but requires careful dependency pinning (`numpy`/`opencv`) to avoid resolver drift.
  - txt2vid accepts interpolation payload, but interpolation execution path is currently wired only in img2vid/vid2vid (contract drift to resolve explicitly).

Decisions / constraints (locked)
- User selected Option A (out-of-box provisioning).
- Keep strict fail-loud behavior (no silent fallbacks/shims).
- Do not rely on undocumented external integration module names as the only runtime path.
- Keep existing repo policy: no import/copy from `.refs/**` into active code; `.refs/**` is semantic reference only.

Follow-up (ordered)
1. Implement deterministic in-repo RIFE adapter seam (`apps/backend/video/interpolation/rife.py`) using a pinned runtime dependency path.
2. Add ffmpeg/ffprobe centralized resolution in backend video IO/export and wire installer provisioning for Linux/Windows.
3. Add/update Python deps (`pyproject.toml` + `uv.lock`) for chosen interpolation/runtime toolchain.
4. Align UI/API contract defaults/messages for interpolation (including model path expectations) and remove drift.
5. Add focused backend tests for interpolation availability/error contract + ffmpeg resolver behavior.
6. Run required validations and sync docs (`README.md`, `INSTALL.md`, `.sangoi` task-log/changelog).

Next immediate step (do this first)
- Implement runtime seam first: add explicit ffmpeg binary resolver utility + deterministic RIFE adapter contract, then wire installers.
Commands:
cd /home/lucas/work/stable-diffusion-webui-codex
rg -n "maybe_interpolate_rife|RIFEUnavailableError|ffmpeg|ffprobe|video_interpolation" apps/backend/video apps/backend/use_cases install-webui.sh install-webui.bat
sed -n '1,260p' apps/backend/video/interpolation/rife.py

Files
- Changed files (last relevant commit(s)):
  - No committed root-repo code yet for this objective (work starts from `17ea092ed5bc22705c5840daa04ce50b9c36d7fb`).
  - Draft planning artifact exists (separate `.sangoi` git repo): `.sangoi/plans/2026-02-12-video-interpolation-deps-installer.md`.
- Focus files to open first:
  - `apps/backend/video/interpolation/rife.py` — current dynamic import seam and error surface.
  - `apps/backend/video/interpolation/__init__.py` — strict fail-loud orchestration path.
  - `apps/backend/video/io/ffmpeg.py` — ffprobe/ffmpeg discovery and probe/extract contracts.
  - `apps/backend/video/export/ffmpeg_exporter.py` — export path that depends on ffmpeg binary availability.
  - `install-webui.sh` — Linux installer provisioning lane.
  - `install-webui.bat` — Windows installer provisioning lane.
  - `pyproject.toml` and `uv.lock` — dependency pinning/lock integrity.
  - `apps/interface/blocks.json` and `apps/interface/src/composables/useVideoGeneration.ts` — interpolation defaults (`rifeEnabled`, `rifeModel`, `rifeTimes`).

Validation (what “green” looks like)
- CODEX_ROOT="$PWD" PYTHONPATH="$PWD" "$PWD/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_video_api_request_steps.py
  # expected: pass; request payload contract stays valid
- CODEX_ROOT="$PWD" PYTHONPATH="$PWD" "$PWD/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_vid2vid_api_request.py .sangoi/dev/tests/backend/test_ffmpeg_exporter_prefix_safety.py
  # expected: pass; vid2vid/export contract preserved
- CODEX_ROOT="$PWD" PYTHONPATH="$PWD" "$PWD/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_wan22_14b_img2vid_guardrail.py
  # expected: pass; no WAN task exposure regressions
- python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff
  # expected: no unresolved header drift for touched `apps/**`
- bash .sangoi/.tools/link-check.sh .sangoi
  # expected: no broken markdown links

References (read before coding)
- `.sangoi/plans/2026-02-12-video-interpolation-deps-installer.md`
- `.sangoi/task-logs/2025-10-22-f6-video-interpolation-rife-integration.md`
- `.sangoi/task-logs/2025-10-25-remove-backend-fallbacks.md`
- `.sangoi/backend/runtime/video-architecture.md`
- `.sangoi/howto/PROMPT_GUIDE.md`
```
