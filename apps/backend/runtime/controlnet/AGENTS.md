# apps/backend/runtime/controlnet Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Provide typed configuration, graph orchestration, and runtime helpers for ControlNet integration.

## Notes
- `config.py` defines `ControlRequest`, `ControlNode`, and related dataclasses; enforce validation before wiring nodes into the graph.
- `runtime.py` exposes `ControlComposite` / `build_composite`, bridging the new graph model with legacy interfaces (`get_control`, `cleanup`, etc.).
- Patchers should call `UnetPatcher.add_control_node` with a prepared `ControlNode`; sampling activates the composite via `UnetPatcher.activate_control()`.
- `preprocessors/` contains Codex-native preprocessing pipelines (edge detectors, etc.) registered through `ControlPreprocessorRegistry`.
