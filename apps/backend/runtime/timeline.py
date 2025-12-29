"""TVA-Style Timeline Tracer for Inference Pipelines.

Inspired by Loki's Sacred Timeline visualization, this module provides
visual tracking of execution flow during model inference.

Usage:
    # Enable via environment variable
    CODEX_TIMELINE=1 python launch.py

    # Or programmatically
    from apps.backend.runtime.timeline import timeline, timeline_node

    @timeline_node("sampling.step")
    def my_function():
        ...

    with timeline.capture("my_inference"):
        run_inference()
    timeline.render()
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, List, Optional, TypeVar

import torch

_log = logging.getLogger("backend.timeline")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """A single event in the execution timeline."""
    timestamp: float  # time.perf_counter()
    stage: str        # e.g., "text_encoding", "sampling", "decode"
    name: str         # e.g., "get_learned_conditioning", "step[1/8]"
    event_type: str   # "enter" or "exit"
    depth: int        # Call stack depth
    thread_id: int    # threading.get_ident()
    duration_ms: Optional[float] = None  # Set on "exit" events
    vram_mb: Optional[float] = None      # GPU memory at this point
    extra: dict = field(default_factory=dict)


@dataclass
class TimelineCapture:
    """A captured timeline session."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    events: List[TimelineEvent] = field(default_factory=list)
    peak_vram_mb: float = 0.0


# -----------------------------------------------------------------------------
# Timeline Collector (Singleton)
# -----------------------------------------------------------------------------

class TimelineCollector:
    """Collects timeline events during execution."""
    
    _instance: Optional["TimelineCollector"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "TimelineCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        self._enabled = _env_flag("CODEX_TIMELINE", False)
        self._max_events = _env_int("CODEX_TIMELINE_MAX_EVENTS", 10000)
        self._track_vram = _env_flag("CODEX_TIMELINE_VRAM", True)
        self._captures: List[TimelineCapture] = []
        self._active_capture: Optional[TimelineCapture] = None
        self._depth = threading.local()
        self._enter_times: dict[int, dict[str, float]] = {}  # thread_id -> {key: start_time}
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    def enable(self) -> None:
        """Enable timeline collection."""
        self._enabled = True
        _log.info("[timeline] Enabled")
    
    def disable(self) -> None:
        """Disable timeline collection."""
        self._enabled = False
        _log.info("[timeline] Disabled")
    
    def _get_depth(self) -> int:
        return getattr(self._depth, "value", 0)
    
    def _set_depth(self, value: int) -> None:
        self._depth.value = value
    
    def _get_vram_mb(self) -> Optional[float]:
        if not self._track_vram or not torch.cuda.is_available():
            return None
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            return None
    
    def enter(self, stage: str, name: str, **extra: Any) -> None:
        """Record entering a timeline node."""
        if not self._enabled or self._active_capture is None:
            return
        
        if len(self._active_capture.events) >= self._max_events:
            return
        
        thread_id = threading.get_ident()
        depth = self._get_depth()
        self._set_depth(depth + 1)
        
        vram = self._get_vram_mb()
        now = time.perf_counter()
        
        # Store enter time for duration calculation
        key = f"{stage}:{name}:{depth}"
        if thread_id not in self._enter_times:
            self._enter_times[thread_id] = {}
        self._enter_times[thread_id][key] = now
        
        event = TimelineEvent(
            timestamp=now,
            stage=stage,
            name=name,
            event_type="enter",
            depth=depth,
            thread_id=thread_id,
            vram_mb=vram,
            extra=extra,
        )
        self._active_capture.events.append(event)
        
        if vram and vram > self._active_capture.peak_vram_mb:
            self._active_capture.peak_vram_mb = vram
    
    def exit(self, stage: str, name: str, **extra: Any) -> None:
        """Record exiting a timeline node."""
        if not self._enabled or self._active_capture is None:
            return
        
        if len(self._active_capture.events) >= self._max_events:
            return
        
        thread_id = threading.get_ident()
        depth = max(0, self._get_depth() - 1)
        self._set_depth(depth)
        
        vram = self._get_vram_mb()
        now = time.perf_counter()
        
        # Calculate duration
        key = f"{stage}:{name}:{depth}"
        duration_ms = None
        if thread_id in self._enter_times and key in self._enter_times[thread_id]:
            start = self._enter_times[thread_id].pop(key)
            duration_ms = (now - start) * 1000
        
        event = TimelineEvent(
            timestamp=now,
            stage=stage,
            name=name,
            event_type="exit",
            depth=depth,
            thread_id=thread_id,
            duration_ms=duration_ms,
            vram_mb=vram,
            extra=extra,
        )
        self._active_capture.events.append(event)
        
        if vram and vram > self._active_capture.peak_vram_mb:
            self._active_capture.peak_vram_mb = vram
    
    @contextmanager
    def capture(self, name: str = "inference"):
        """Context manager to capture a timeline session."""
        if not self._enabled:
            yield
            return
        
        cap = TimelineCapture(name=name, start_time=time.perf_counter())
        self._active_capture = cap
        self._captures.append(cap)
        
        try:
            yield cap
        finally:
            cap.end_time = time.perf_counter()
            self._active_capture = None
    
    def get_last_capture(self) -> Optional[TimelineCapture]:
        """Get the most recent capture."""
        return self._captures[-1] if self._captures else None
    
    def clear(self) -> None:
        """Clear all captures."""
        self._captures.clear()
        self._enter_times.clear()


# Global singleton
timeline = TimelineCollector()


# -----------------------------------------------------------------------------
# Decorator
# -----------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def timeline_node(stage: str, name: str = "") -> Callable[[F], F]:
    """Decorator to mark a function as a timeline checkpoint.
    
    Args:
        stage: The pipeline stage (e.g., "text_encoding", "sampling", "decode")
        name: Optional name override. Defaults to function name.
    
    Example:
        @timeline_node("sampling", "euler_step")
        def sample_euler(...):
            ...
    """
    def decorator(func: F) -> F:
        node_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not timeline.enabled:
                return func(*args, **kwargs)
            
            timeline.enter(stage, node_name)
            try:
                return func(*args, **kwargs)
            finally:
                timeline.exit(stage, node_name)
        
        return wrapper  # type: ignore[return-value]
    
    return decorator


# -----------------------------------------------------------------------------
# Visual Renderer
# -----------------------------------------------------------------------------

# Box drawing characters
_BOX = {
    "top_left": "╭",
    "top_right": "╮", 
    "bottom_left": "╰",
    "bottom_right": "╯",
    "horizontal": "─",
    "vertical": "│",
    "branch": "├",
    "corner": "└",
    "stage_start": "◆",
    "stage_end": "◇",
}


def render_timeline(capture: Optional[TimelineCapture] = None, *, use_color: bool = True) -> str:
    """Render a timeline capture as ASCII art.
    
    Args:
        capture: The capture to render. Defaults to last capture.
        use_color: Whether to use ANSI colors.
    
    Returns:
        Rendered timeline string.
    """
    if capture is None:
        capture = timeline.get_last_capture()
    
    if capture is None:
        return "[timeline] No capture available"
    
    if not capture.events:
        return f"[timeline] {capture.name}: No events captured"
    
    lines: List[str] = []
    
    # Header
    width = 70
    lines.append("═" * width)
    title = f"{capture.name} Timeline"
    lines.append(title.center(width))
    lines.append("═" * width)
    lines.append("")
    
    # Process events by stage
    current_stage: Optional[str] = None
    stage_events: List[TimelineEvent] = []
    
    def _render_stage(stage: str, events: List[TimelineEvent]) -> None:
        if not events:
            return
        
        # Stage header
        lines.append(f"{_BOX['stage_start']} {stage} " + _BOX["horizontal"] * (width - len(stage) - 5) + _BOX["top_right"])
        
        # Events within stage
        for evt in events:
            if evt.event_type != "exit":
                continue  # Only show exit events (with duration)
            
            indent = _BOX["vertical"] + "  " + "  " * evt.depth
            duration = f"{evt.duration_ms:.1f}ms" if evt.duration_ms else ""
            vram = f"[{evt.vram_mb:.0f}MB]" if evt.vram_mb else ""
            
            name_part = f"{_BOX['corner'] if evt.depth > 0 else ''}{evt.name}"
            right_part = f"{duration} {vram}".strip()
            
            # Calculate padding
            pad = width - len(indent) - len(name_part) - len(right_part) - 3
            line = f"{indent}{name_part}{' ' * max(1, pad)}{right_part} {_BOX['vertical']}"
            lines.append(line)
        
        # Stage footer
        lines.append(f"{_BOX['stage_end']} " + _BOX["horizontal"] * (width - 4) + _BOX["bottom_right"])
        lines.append("")
    
    # Group events by stage
    for evt in capture.events:
        if evt.stage != current_stage:
            if current_stage is not None:
                _render_stage(current_stage, stage_events)
                stage_events = []
            current_stage = evt.stage
        stage_events.append(evt)
    
    # Render last stage
    if current_stage and stage_events:
        _render_stage(current_stage, stage_events)
    
    # Footer
    total_ms = (capture.end_time or time.perf_counter()) - capture.start_time
    total_ms *= 1000
    footer = f"Total: {total_ms:.0f}ms"
    if capture.peak_vram_mb > 0:
        footer += f" | Peak VRAM: {capture.peak_vram_mb:.1f} MB"
    
    lines.append("═" * width)
    lines.append(footer.center(width))
    lines.append("═" * width)
    
    return "\n".join(lines)


def print_timeline(capture: Optional[TimelineCapture] = None) -> None:
    """Print timeline to console."""
    print(render_timeline(capture))


# -----------------------------------------------------------------------------
# JSON Export (for Chrome Trace format)
# -----------------------------------------------------------------------------

def export_chrome_trace(capture: Optional[TimelineCapture] = None) -> dict:
    """Export timeline to Chrome Trace Event Format.
    
    View at chrome://tracing or https://ui.perfetto.dev/
    """
    if capture is None:
        capture = timeline.get_last_capture()
    
    if capture is None:
        return {"traceEvents": []}
    
    events = []
    base_time = capture.start_time
    
    for evt in capture.events:
        chrome_evt = {
            "name": evt.name,
            "cat": evt.stage,
            "ph": "B" if evt.event_type == "enter" else "E",
            "ts": (evt.timestamp - base_time) * 1_000_000,  # microseconds
            "pid": 1,
            "tid": evt.thread_id,
        }
        if evt.extra:
            chrome_evt["args"] = evt.extra
        events.append(chrome_evt)
    
    return {"traceEvents": events}


# -----------------------------------------------------------------------------
# Module Init
# -----------------------------------------------------------------------------

def enable_from_env() -> None:
    """Enable timeline if CODEX_TIMELINE=1."""
    if _env_flag("CODEX_TIMELINE", False):
        timeline.enable()


def get_logs_dir() -> "Path":
    """Get the logs/timeline directory, creating if needed."""
    from pathlib import Path
    from apps.backend.infra.config.repo_root import get_repo_root

    base = get_repo_root()
    
    timeline_dir = base / "logs" / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    return timeline_dir


def save_to_logs(capture: Optional[TimelineCapture] = None) -> Optional[str]:
    """Save timeline capture to logs/timeline/ folder.
    
    Saves both ASCII render (.txt) and Chrome Trace format (.json).
    
    Returns:
        Path to the saved ASCII file, or None if no capture.
    """
    import json
    from pathlib import Path
    import datetime
    
    if capture is None:
        capture = timeline.get_last_capture()
    
    if capture is None or not capture.events:
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in capture.name)
    
    logs_dir = get_logs_dir()
    
    # Save ASCII render
    txt_path = logs_dir / f"{timestamp}-{safe_name}.txt"
    txt_content = render_timeline(capture)
    txt_path.write_text(txt_content, encoding="utf-8")
    
    # Save Chrome Trace JSON
    json_path = logs_dir / f"{timestamp}-{safe_name}.json"
    json_content = export_chrome_trace(capture)
    json_path.write_text(json.dumps(json_content, indent=2), encoding="utf-8")
    
    _log.info(f"[timeline] Saved trace to: {txt_path}")
    _log.info(f"[timeline] Chrome trace: {json_path}")
    _log.info(f"[timeline] View at: https://ui.perfetto.dev/ (drag & drop the JSON)")
    
    return str(txt_path)


def auto_save_and_print(capture: Optional[TimelineCapture] = None) -> Optional[str]:
    """Print timeline and save to logs. Returns path to saved file.
    
    Call this at the end of inference to show results to user.
    """
    if not timeline.enabled:
        return None
    
    if capture is None:
        capture = timeline.get_last_capture()
    
    if capture is None or not capture.events:
        return None
    
    # Print to console
    print_timeline(capture)
    
    # Save to logs
    saved_path = save_to_logs(capture)
    
    return saved_path


__all__ = [
    "timeline",
    "timeline_node",
    "TimelineEvent",
    "TimelineCapture",
    "TimelineCollector",
    "render_timeline",
    "print_timeline",
    "export_chrome_trace",
    "enable_from_env",
    "save_to_logs",
    "auto_save_and_print",
    "get_logs_dir",
]
