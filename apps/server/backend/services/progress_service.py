from __future__ import annotations

import time

from apps.server.backend.core.state import state as backend_state


class ProgressService:
    """Compute progress/ETA and (optionally) current preview image.

    Returns plain dicts suitable for response models; caller can wrap in pydantic.
    """

    def __init__(self, media_service):
        self.media = media_service

    def compute(self, skip_current_image: bool = False):
        # Native implementation; no dependency on legacy shared.state
        if backend_state.job_count == 0:
            return {
                "progress": 0,
                "eta_relative": 0,
                "state": backend_state.dict(),
                "textinfo": backend_state.textinfo,
                "current_task": None,
                "current_image": None,
            }

        progress = 0.01

        if backend_state.job_count > 0:
            progress += backend_state.job_no / backend_state.job_count
        if backend_state.sampling_steps > 0:
            progress += 1 / backend_state.job_count * backend_state.sampling_step / backend_state.sampling_steps

        time_since_start = time.time() - backend_state.time_start
        eta = (time_since_start / max(progress, 1e-6))
        eta_relative = eta - time_since_start

        progress = min(progress, 1)

        backend_state.set_current_image()

        current_image = None
        if backend_state.current_image and not skip_current_image:
            current_image = self.media.encode_image(backend_state.current_image)

        return {
            "progress": progress,
            "eta_relative": eta_relative,
            "state": backend_state.dict(),
            "current_image": current_image,
            "textinfo": backend_state.textinfo,
            "current_task": None,
        }
