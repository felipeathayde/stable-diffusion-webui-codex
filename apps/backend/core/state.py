from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional
import datetime


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
    current_latent: Optional[Any] = None
    id_live_preview: int = 0
    current_image_sampling_step: int = 0
    job: str = ""
    job_timestamp: str = ""
    processing_has_refined_job_count: bool = False
    skipped: bool = False
    interrupted: bool = False
    stopping_generation: bool = False

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
            self.current_latent = None
            self.id_live_preview = 0
            self.job = ""
            self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.processing_has_refined_job_count = False
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False

    def begin(self, job: str = "(unknown)") -> None:
        with self._lock:
            self.job = job
            self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.current_latent = None
            self.current_image = None
            self.current_image_sampling_step = 0
            self.id_live_preview = 0
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False
            self.time_start = time.time()

    def end(self) -> None:
        with self._lock:
            self.job = ""
            self.job_no = 0
            self.job_count = 0
            self.sampling_step = 0
            self.sampling_steps = 0
            self.textinfo = ""
            self.current_image = None
            self.current_latent = None
            self.processing_has_refined_job_count = False

    def next_job(self) -> None:
        with self._lock:
            if self.job_count > 0:
                self.job_no = min(self.job_no + 1, self.job_count)
            self.sampling_step = 0
            self.current_image_sampling_step = 0

    def set_current_image(self, image: Optional[Any] = None, *, sampling_step: Optional[int] = None) -> None:
        with self._lock:
            if image is not None:
                self.current_image = image
                self.id_live_preview += 1
            if sampling_step is not None:
                self.current_image_sampling_step = int(sampling_step)

    def set_current_latent(self, latent: Optional[Any]) -> None:
        with self._lock:
            self.current_latent = latent

    def update_sampling(self, *, step: Optional[int] = None, total: Optional[int] = None) -> None:
        with self._lock:
            if step is not None:
                self.sampling_step = int(step)
            if total is not None:
                self.sampling_steps = int(total)

    def tick(self, *, job_no: Optional[int] = None, sampling_step: Optional[int] = None) -> None:
        with self._lock:
            if job_no is not None:
                self.job_no = int(job_no)
            if sampling_step is not None:
                self.sampling_step = int(sampling_step)

    def set_textinfo(self, message: str) -> None:
        with self._lock:
            self.textinfo = message

    def skip(self) -> None:
        with self._lock:
            self.skipped = True

    def interrupt(self) -> None:
        with self._lock:
            self.interrupted = True

    def stop_generating(self) -> None:
        with self._lock:
            self.stopping_generation = True

    def clear_flags(self) -> None:
        with self._lock:
            self.skipped = False
            self.interrupted = False
            self.stopping_generation = False

    @property
    def should_stop(self) -> bool:
        return self.interrupted or self.stopping_generation or self.skipped

    def dict(self) -> dict[str, Any]:
        # drop lock and non-serializable fields
        d = asdict(self)
        d.pop("_lock", None)
        return d


# Global singleton used by services
state = BackendState()

__all__ = ["BackendState", "state"]
