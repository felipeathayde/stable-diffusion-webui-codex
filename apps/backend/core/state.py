from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional


@dataclass
class BackendState:
    """Lightweight, explicit backend state used for progress reporting.

    This replaces any dependence on legacy `modules.shared.state`.
    """

    job_count: int = 0
    job_no: int = 0
    sampling_steps: int = 0
    sampling_step: int = 0
    time_start: float = 0.0
    textinfo: str = ""
    current_image: Optional[Any] = None

    _lock: threading.Lock = threading.Lock()

    def start(self, job_count: int, sampling_steps: int) -> None:
        with self._lock:
            self.job_count = int(job_count)
            self.job_no = 0
            self.sampling_steps = int(sampling_steps)
            self.sampling_step = 0
            self.time_start = time.time()
            self.textinfo = ""
            self.current_image = None

    def set_current_image(self, image: Optional[Any] = None) -> None:
        with self._lock:
            if image is not None:
                self.current_image = image

    def tick(self, *, job_no: Optional[int] = None, sampling_step: Optional[int] = None) -> None:
        with self._lock:
            if job_no is not None:
                self.job_no = int(job_no)
            if sampling_step is not None:
                self.sampling_step = int(sampling_step)

    def dict(self) -> dict[str, Any]:
        # drop lock and non‑serializable fields
        d = asdict(self)
        d.pop("_lock", None)
        return d


# Global singleton used by services
state = BackendState()

__all__ = ["BackendState", "state"]

