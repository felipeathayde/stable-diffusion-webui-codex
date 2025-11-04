from __future__ import annotations

from fastapi.exceptions import HTTPException
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.sampling.catalog import SAMPLER_ALIAS_TO_CANONICAL, SUPPORTED_SAMPLERS


class SamplerService:
    """Sampler/Scheduler resolution and validation."""

    @staticmethod
    def resolve(sampler_name_or_index, scheduler: str | None):
        """Return a tuple (sampler_name, scheduler_name).

        Native resolver using Codex SamplerKind; indices are not supported.
        """
        if isinstance(sampler_name_or_index, int):
            raise HTTPException(status_code=400, detail="Sampler index not supported; use name")
        name = str(sampler_name_or_index or "Automatic")
        canonical = SAMPLER_ALIAS_TO_CANONICAL.get(name.strip().lower(), name.strip().lower())
        try:
            kind = SamplerKind.from_string(canonical)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return kind.value, (scheduler or "Automatic")

    @staticmethod
    def ensure_valid_sampler(name: str) -> str:
        canonical = SAMPLER_ALIAS_TO_CANONICAL.get(name.strip().lower(), name.strip().lower())
        if canonical not in SUPPORTED_SAMPLERS:
            raise HTTPException(status_code=400, detail=f"Sampler '{name}' is not supported by this build")
        try:
            _ = SamplerKind.from_string(canonical)
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex))
        return canonical
