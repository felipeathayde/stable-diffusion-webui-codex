# AGENT — Runtime Diagnostics

Purpose: Central home for runtime diagnostics and debugging helpers (trace/timeline/pipeline debug, exception dump hooks).

Key files:
- `apps/backend/runtime/diagnostics/call_trace.py`: `sys.setprofile`-based call tracing for deep debugging.
- `apps/backend/runtime/diagnostics/exception_hook.py`: Sys/thread/asyncio exception dump hooks (writes to `logs/`).
- `apps/backend/runtime/diagnostics/pipeline_debug.py`: Pipeline debug flag + decorator helpers.
- `apps/backend/runtime/diagnostics/timeline.py`: Inference “timeline” tracer (nested stage/event tracking + render/export).
- `apps/backend/runtime/diagnostics/trace.py`: Lightweight torch tracing helpers (`torch.nn.Module.to` patch + scoped sections).

Notes:
- Diagnostics should stay lightweight and avoid importing heavy ML dependencies at import time unless strictly required.
- If a failure is expected/optional, make it explicit; do not swallow unexpected errors.

Last Review: 2026-01-18

