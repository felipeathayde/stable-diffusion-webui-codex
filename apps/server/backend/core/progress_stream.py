from __future__ import annotations

from threading import Thread
from queue import Queue, Empty
from typing import Any, Callable, Iterator


def stream_run(
    run_fn: Callable[..., Any],
    *,
    cfg: Any,
    logger: Any = None,
    result_key: str = "frames",
    poll_interval: float = 0.1,
) -> Iterator[dict]:
    """Generic streaming adapter for long-running synchronous run_* calls.

    - run_fn must accept (cfg, logger=?, on_progress=callback)
    - Emits dict events of shape {type: 'progress'|'result'|'error', ...}
    - Threaded implementation: the worker pushes progress/results into a queue; the caller yields them.
    """

    q: Queue = Queue()

    def _cb(stage: str, step: int, total: int, percent: float, **kwargs: Any) -> None:
        try:
            q.put({
                "type": "progress",
                "stage": stage,
                "step": int(step),
                "total": int(total),
                "percent": float(percent),
                **kwargs,
            })
        except Exception:
            pass

    def _worker() -> None:
        try:
            out = run_fn(cfg, logger=logger, on_progress=_cb)
            q.put({"type": "result", result_key: out})
        except Exception as ex:  # pragma: no cover
            # Always log the full stack trace to backend console, independent of UI/SSE.
            try:
                if logger is not None and hasattr(logger, "exception"):
                    logger.exception("stream_run worker crashed: %s", ex)
                else:
                    import traceback, sys
                    traceback.print_exc(file=sys.stderr)
            finally:
                # Surface error to the stream so engines can escalate
                q.put({"type": "error", "error": ex})

    t = Thread(target=_worker, name=f"stream_run-{getattr(run_fn, '__name__', 'run')}", daemon=True)
    t.start()

    # Drain queue until result/error arrives
    while True:
        try:
            ev = q.get(timeout=poll_interval)
        except Empty:  # pragma: no cover
            continue
        yield ev
        if ev.get("type") in ("result", "error"):
            break
