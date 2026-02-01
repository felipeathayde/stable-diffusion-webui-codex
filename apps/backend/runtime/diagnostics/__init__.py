"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime diagnostics and debugging helpers.
Collects optional tracing, timeline, and exception-dump utilities used to diagnose backend runtime behavior.

Symbols (top-level; keep in sync; no ghosts):
- `call_trace` (module): Global `sys.setprofile` call tracer.
- `exception_hook` (module): Sys/thread/asyncio exception dump hooks.
- `pipeline_debug` (module): Pipeline debug flag + decorator helpers.
- `profiler` (module): Global opt-in torch profiler wrapper (trace export + transfer totals).
- `timeline` (module): Timeline tracer for inference pipelines.
- `trace` (module): Lightweight torch tracing helpers.
"""

__all__ = [
    "call_trace",
    "exception_hook",
    "pipeline_debug",
    "profiler",
    "timeline",
    "trace",
]
