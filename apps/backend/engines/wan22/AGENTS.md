<!-- tags: backend, engines, wan22, gguf -->

# apps/backend/engines/wan22 Overview
Date: 2025-12-06
Last Review: 2026-03-02
Status: Active

## Purpose
- WAN22 engine implementations (`txt2vid`, `img2vid`, `vid2vid`) that coordinate GGUF-backed runtime execution.

## Key Files
- `apps/backend/engines/wan22/wan22_14b.py` — `Wan2214BEngine` (canonical 14B lane for `txt2vid`/`img2vid`/`vid2vid`; GGUF-backed lazy wrapper).
- `apps/backend/engines/wan22/wan22_14b_animate.py` — `Wan22Animate14BEngine` (`vid2vid` lane; engine id `wan22_14b_animate`; GGUF-backed lazy wrapper).
- `apps/backend/engines/wan22/wan22_5b.py` — `Wan225BEngine` (GGUF-backed wrapper for 5B lane; strict file-only model validation).
- `apps/backend/engines/wan22/wan22_common.py` — shared WAN route/build/asset normalization helpers consumed by WAN22 engines.

## Current Behavior
- All active WAN22 engine lanes are GGUF-only wrappers; `load()` validates GGUF model input and stores runtime metadata without eager pipeline construction.
- `wan22_14b_animate` keeps registration key `wan22_14b_animate` and dispatches `vid2vid` through `apps/backend/use_cases/vid2vid.py` canonical orchestration (no engine-side `method` override injection).
- `wan22_14b_animate` exposes `img2vid` execution for canonical `vid2vid` chunk strategies while keeping advertised task capability scoped to `vid2vid`.
- Model-path validation fails loud on empty refs, directories, non-`.gguf` inputs, missing files, and obvious wrong-lane weight labels (5B vs 14B).
- WAN22 stage/settings behavior is payload-driven through use-cases/runtime (`apps/backend/use_cases/*` + `apps/backend/runtime/families/wan22/*`).

## Execution Paths
- GGUF: strict assets via WAN22 runtime/use-case orchestration; model materialization occurs at execution time, not engine `load()`.

## Device/Dtype Policy
- Engine-local `device` override is rejected (`load(device=...)` must be unset/`auto`); mount device comes from memory manager defaults.
