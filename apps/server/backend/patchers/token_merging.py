from __future__ import annotations

from contextlib import contextmanager


def apply_token_merging(model, ratio: float | int | None) -> None:
    """Apply token merging to the current UNet if supported.

    Codex-native stub: until a native implementation is provided, any
    request to merge tokens with a positive ratio raises a clear error to
    avoid silent mismatches.
    """
    r = float(ratio or 0.0)
    if r <= 0.0:
        return
    raise NotImplementedError("Token merging is not implemented natively yet")


@contextmanager
def SkipWritingToConfig():
    """No-op context manager kept for call-site compatibility.

    We do not write legacy config files; keeping this as a placeholder.
    """
    yield


__all__ = ["apply_token_merging", "SkipWritingToConfig"]

