from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar, cast


logger = logging.getLogger("backend.pipeline.debug")

PIPELINE_DEBUG_ENABLED = False

F = TypeVar("F", bound=Callable[..., Any])


def set_pipeline_debug(enabled: bool) -> None:
    global PIPELINE_DEBUG_ENABLED
    PIPELINE_DEBUG_ENABLED = bool(enabled)
    logger.info("pipeline debug %s", "ativado" if PIPELINE_DEBUG_ENABLED else "desativado")


def log(message: str) -> None:
    if PIPELINE_DEBUG_ENABLED:
        logger.info(message)


def pipeline_trace(func: F) -> F:
    qualname = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        if PIPELINE_DEBUG_ENABLED:
            logger.info("entrou em %s", qualname)
        result = func(*args, **kwargs)
        if PIPELINE_DEBUG_ENABLED:
            logger.info("saiu de %s", qualname)
        return result

    return cast(F, wrapper)


__all__ = ["PIPELINE_DEBUG_ENABLED", "set_pipeline_debug", "log", "pipeline_trace"]
