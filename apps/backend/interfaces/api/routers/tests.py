"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bounded backend diagnostics routes.
Exposes narrow live-validation endpoints for runtime-owned diagnostics without overloading the
system health surface or turning the API into an arbitrary test-execution framework.

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for bounded diagnostics endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException

from apps.backend.interfaces.api.public_errors import public_http_error_detail
from apps.backend.runtime.attention.sram.splitkv_validation import (
    SplitKvValidationInternalInvariant,
    SplitKvValidationInvalidRequest,
    SplitKvValidationPreconditionFailure,
    parse_splitkv_validation_request,
    run_splitkv_validation,
)


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/tests/attention/sram/splitkv")
    def validate_attention_sram_splitkv(payload: Any = Body(default=None)) -> dict[str, Any]:
        try:
            request = parse_splitkv_validation_request(payload)
            return run_splitkv_validation(request).to_payload()
        except SplitKvValidationInvalidRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except SplitKvValidationPreconditionFailure as exc:
            raise HTTPException(
                status_code=503,
                detail=public_http_error_detail(exc, fallback="SRAM split-KV precondition failed"),
            ) from exc
        except SplitKvValidationInternalInvariant as exc:
            raise HTTPException(
                status_code=500,
                detail=public_http_error_detail(exc, fallback="SRAM split-KV validation failed"),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive sync route guard
            raise HTTPException(
                status_code=500,
                detail=public_http_error_detail(exc, fallback="internal error"),
            ) from exc

    return router
