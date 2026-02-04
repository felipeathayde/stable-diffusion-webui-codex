# apps/backend/runtime/kernels/wan_t5 Overview
Date: 2025-10-28
Last Review: 2026-01-02
Status: Active

## Purpose
- WAN T5-specific CUDA kernels and build scripts enabling FP8 text encoder support.

## Notes
- Ensure build instructions match the guidance in `.sangoi/frontend/guias` and runtime documentation (requires CUDA 12.8, SM70+/SM90 configurations).
- 2026-01-02: Added standardized file header docstrings to `setup.py` (doc-only change; part of rollout).
