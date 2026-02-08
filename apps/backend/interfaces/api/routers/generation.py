"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generation API routes (txt2img/img2img/txt2vid/img2vid/vid2vid).
Contains request parsing and payload validation (including hires tile config via `extras.hires.tile` / `img2img_hires_tile`, Z-Image Turbo/Base
`extras.zimage_variant`, and WAN video export options like `video_return_frames`), and delegates image task workers to
`apps/backend/interfaces/api/tasks/generation_tasks.py`.
Hires supports sampler/scheduler overrides for the hires pass (txt2img: `extras.hires.sampler` / `extras.hires.scheduler`; img2img: `img2img_hires_sampling` / `img2img_hires_scheduler`).
Includes strict ER-SDE option parsing (`extras.er_sde` / `img2img_extras.er_sde`) plus release-scope enforcement for sampler fields and
prompt `<sampler:...>` control tags (Anima-only rollout).
Uses cached inventory slot metadata for sha-selected text encoders (`tenc_sha`) and enforces WAN video `height/width % 16 == 0` (Diffusers parity) to avoid silent patch-grid cropping (returns suggested rounded-up dimensions on invalid requests).
Requires explicit per-request device selection and serializes GPU-heavy execution via the shared inference gate when `CODEX_SINGLE_FLIGHT=1` (default on).

Symbols (top-level; keep in sync; no ghosts):
- `build_router` (function): Build the APIRouter for generation endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile

from apps.backend.interfaces.api.path_utils import _path_from_api
from apps.backend.interfaces.api.inference_gate import acquire_inference_gate, release_inference_gate, single_flight_enabled
from apps.backend.interfaces.api.task_registry import TaskEntry, register_task, unregister_task


def build_router(*, codex_root: Path, media, live_preview, opts_get, opts_snapshot, generation_provenance, save_generated_images, param_utils) -> APIRouter:
    router = APIRouter()
    CODEX_ROOT = codex_root
    _GENERATION_PROVENANCE = generation_provenance(codex_root)
    _save_generated_images = save_generated_images
    _opts_get = opts_get
    _opts_snapshot = opts_snapshot
    _p = param_utils

    from apps.backend.core.engine_interface import TaskType
    from apps.backend.core.orchestrator import InferenceOrchestrator
    from apps.backend.core.requests import (
        ProgressEvent,
        ResultEvent,
        Txt2ImgRequest,
        Img2ImgRequest,
        Txt2VidRequest,
        Img2VidRequest,
        Vid2VidRequest,
    )
    from apps.backend.interfaces.api.device_selection import parse_device_from_payload
    from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine

    def _ensure_default_engines_registered() -> None:
        # Generation endpoints require the engine registry, but API startup should remain import-light.
        # Register engines lazily so health/models endpoints can work without pulling torch-heavy deps.
        from apps.backend.engines import register_default_engines

        register_default_engines(replace=False)

    from apps.backend.types.payloads import TXT2IMG_KEYS, EXTRAS_KEYS
    _TXT2IMG_ALLOWED_KEYS = set(TXT2IMG_KEYS.ALL)
    _TXT2IMG_EXTRAS_KEYS = set(EXTRAS_KEYS.ALL)
    _TXT2IMG_HIRES_KEYS = set(TXT2IMG_KEYS.HIRES_ALL)
    _IMG2IMG_EXTRAS_KEYS = set(EXTRAS_KEYS.ALL) - {"hires", "refiner", "batch_size", "batch_count"}
    _ER_SDE_OPTION_KEYS = {"solver_type", "max_stage", "eta", "s_noise"}
    _PROMPT_SAMPLER_CONTROL_RE = re.compile(
        r"<\s*sampler\s*:\s*([^:>]+)(?::[^:>]+)?\s*>",
        re.IGNORECASE,
    )
    from apps.backend.runtime.vision.upscalers.specs import tile_config_from_payload

    _ANIMA_ALLOWED_SAMPLERS = tuple(ENGINE_SURFACES[SemanticEngine.ANIMA].samplers or ())
    if not _ANIMA_ALLOWED_SAMPLERS:
        raise RuntimeError("Anima capability surface must declare a non-empty sampler allowlist.")

    def _reject_unknown_keys(obj: Mapping[str, Any], allowed: set[str], context: str) -> None:
        unknown = sorted(set(obj.keys()) - allowed)
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unexpected {context} key(s): {', '.join(unknown)}")

    def _resolve_smart_flags(payload: Mapping[str, Any]) -> Tuple[bool, bool, bool]:
        """Resolve per-request smart flags (offload/fallback/cache).

        Precedence: payload value when present (not null) → options snapshot.
        """
        snap = _opts_snapshot()
        smart_offload = (
            bool(payload.get("smart_offload"))
            if payload.get("smart_offload") is not None
            else bool(getattr(snap, "codex_smart_offload", False))
        )
        smart_fallback = (
            bool(payload.get("smart_fallback"))
            if payload.get("smart_fallback") is not None
            else bool(getattr(snap, "codex_smart_fallback", False))
        )
        smart_cache = (
            bool(payload.get("smart_cache"))
            if payload.get("smart_cache") is not None
            else bool(getattr(snap, "codex_smart_cache", False))
        )
        return smart_offload, smart_fallback, smart_cache


    def _require_str_field(payload: Dict[str, Any], key: str, *, allow_empty: bool = False, trim: bool = True) -> str:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a string")
        result = value.strip() if trim else value
        if not allow_empty and result == "":
            raise HTTPException(status_code=400, detail=f"'{key}' must not be empty")
        return result if trim else value


    def _require_int_field(payload: Dict[str, Any], key: str, *, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
        if isinstance(value, float):
            if not value.is_integer():
                raise HTTPException(status_code=400, detail=f"'{key}' must be an integer")
            value = int(value)
        else:
            value = int(value)
        if minimum is not None and value < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return value


    def _require_float_field(payload: Dict[str, Any], key: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
        if key not in payload:
            raise HTTPException(status_code=400, detail=f"Missing '{key}'")
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise HTTPException(status_code=400, detail=f"'{key}' must be a finite number")
        if minimum is not None and result < minimum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be >= {minimum}")
        if maximum is not None and result > maximum:
            raise HTTPException(status_code=400, detail=f"'{key}' must be <= {maximum}")
        return result


    def _resolve_wan_metadata_dir(payload: Dict[str, Any]) -> str:
        """Resolve the WAN metadata directory for GGUF runs.

        Preferred contract: pass `wan_metadata_repo="Org/Repo"` and resolve it under
        `apps/backend/huggingface/` (vendored HF mirror).

        Back-compat: accept `wan_metadata_dir` (or `wan_tokenizer_dir`) as an explicit path.
        """
        raw_repo = payload.get("wan_metadata_repo")
        if isinstance(raw_repo, str) and raw_repo.strip():
            repo_id = raw_repo.strip()
            if repo_id.count("/") != 1:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id like 'Org/Repo'")
            org, repo = repo_id.split("/", 1)
            if not org or not repo or org in {".", ".."} or repo in {".", ".."}:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id like 'Org/Repo'")
            if Path(repo_id).is_absolute():
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' must be a repo id (not a filesystem path)")

            hf_root = (CODEX_ROOT / "apps" / "backend" / "huggingface").resolve()
            local_dir = (hf_root / org / repo).resolve()
            try:
                local_dir.relative_to(hf_root)
            except Exception:
                raise HTTPException(status_code=400, detail="'wan_metadata_repo' resolves outside the vendored HF root")
            if not local_dir.is_dir():
                raise HTTPException(status_code=409, detail=f"WAN metadata repo not found locally: {repo_id}")
            return str(local_dir)

        meta_dir = payload.get("wan_metadata_dir") or payload.get("wan_tokenizer_dir")
        if isinstance(meta_dir, str) and meta_dir.strip():
            return _path_from_api(meta_dir)

        raise HTTPException(status_code=400, detail="'wan_metadata_repo' (or 'wan_metadata_dir') is required for WAN GGUF")


    def _parse_styles(payload: Dict[str, Any]) -> List[str]:
        raw = payload.get('styles')
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
        out: List[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                raise HTTPException(status_code=400, detail="'styles' must be an array of strings")
            text = entry.strip()
            if text:
                out.append(text)
        return out

    def _parse_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
        raw = payload.get('metadata')
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'metadata' must be an object")
        return dict(raw)


    def _normalize_er_sde_solver_type(value: object, *, field_name: str) -> str:
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a string")
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        mapping = {
            "er sde": "er_sde",
            "reverse time sde": "reverse_time_sde",
            "ode": "ode",
        }
        solver_type = mapping.get(normalized)
        if solver_type is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{field_name}' must be one of: ER-SDE, Reverse-time SDE, ODE "
                    "(or canonical tokens: er_sde, reverse_time_sde, ode)"
                ),
            )
        return solver_type


    def _parse_er_sde_options(value: object, *, field_name: str) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be an object")
        _reject_unknown_keys(value, _ER_SDE_OPTION_KEYS, field_name)

        options = dict(value)
        solver_type = _normalize_er_sde_solver_type(
            options.get("solver_type", "er_sde"),
            field_name=f"{field_name}.solver_type",
        )
        max_stage_raw = options.get("max_stage", 3)
        if isinstance(max_stage_raw, bool) or not isinstance(max_stage_raw, (int, float)):
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be an integer in [1, 3]",
            )
        if isinstance(max_stage_raw, float) and not max_stage_raw.is_integer():
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be an integer in [1, 3]",
            )
        max_stage = int(max_stage_raw)
        if max_stage < 1 or max_stage > 3:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}.max_stage' must be in [1, 3]",
            )

        eta_raw = options.get("eta", 1.0)
        if isinstance(eta_raw, bool) or not isinstance(eta_raw, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be numeric")
        eta = float(eta_raw)
        if not math.isfinite(eta):
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be finite")
        if eta < 0.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.eta' must be >= 0")

        s_noise_raw = options.get("s_noise", 1.0)
        if isinstance(s_noise_raw, bool) or not isinstance(s_noise_raw, (int, float)):
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be numeric")
        s_noise = float(s_noise_raw)
        if not math.isfinite(s_noise):
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be finite")
        if s_noise < 0.0:
            raise HTTPException(status_code=400, detail=f"'{field_name}.s_noise' must be >= 0")

        if solver_type == "ode" or (solver_type == "reverse_time_sde" and eta == 0.0):
            eta = 0.0
            s_noise = 0.0

        return {
            "solver_type": solver_type,
            "max_stage": int(max_stage),
            "eta": float(eta),
            "s_noise": float(s_noise),
        }


    def _parse_txt2img_extras(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        raw = payload.get('extras')
        if raw is None:
            return {}, None
        if not isinstance(raw, dict):
            raise HTTPException(status_code=400, detail="'extras' must be an object")
        _reject_unknown_keys(raw, _TXT2IMG_EXTRAS_KEYS, "extras")
        extras: Dict[str, Any] = {}
        if 'randn_source' in raw:
            extras['randn_source'] = str(raw['randn_source'])
        if 'eta_noise_seed_delta' in raw:
            val = raw['eta_noise_seed_delta']
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise HTTPException(status_code=400, detail="'extras.eta_noise_seed_delta' must be numeric")
            extras['eta_noise_seed_delta'] = int(val)
        # SHA keys for asset selection (from dataclass)
        from apps.backend.types.payloads import SHA_KEYS
        for key in SHA_KEYS.ALL:
            if key not in raw:
                continue
            value = raw.get(key)
            if value is None:
                continue
            if key == "tenc_sha":
                if isinstance(value, str):
                    sha = value.strip()
                    if sha:
                        extras[key] = sha
                    continue
                if isinstance(value, list):
                    shas: list[str] = []
                    for entry in value:
                        if not isinstance(entry, str):
                            raise HTTPException(status_code=400, detail="'extras.tenc_sha' must be a string or array of strings")
                        sha = entry.strip()
                        if sha:
                            shas.append(sha)
                    if shas:
                        extras[key] = shas
                    continue
                raise HTTPException(status_code=400, detail="'extras.tenc_sha' must be a string or array of strings")

            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"'extras.{key}' must be a string")
            sha = value.strip()
            if sha:
                extras[key] = sha
        # Batch params
        if 'batch_size' in raw:
            extras['batch_size'] = int(raw['batch_size'])
        if 'batch_count' in raw:
            extras['batch_count'] = int(raw['batch_count'])
        # Z-Image variant selection (Turbo/Base). This is used by the engine to pick
        # variant-specific scheduler semantics (flow_shift) and CFG behavior.
        if 'zimage_variant' in raw:
            val = raw.get('zimage_variant')
            if val is None:
                pass
            elif not isinstance(val, str):
                raise HTTPException(status_code=400, detail="'extras.zimage_variant' must be a string")
            else:
                variant = val.strip().lower()
                if variant not in {"turbo", "base"}:
                    raise HTTPException(
                        status_code=400,
                        detail="'extras.zimage_variant' must be one of: turbo, base",
                    )
                extras['zimage_variant'] = variant
        if "er_sde" in raw:
            extras["er_sde"] = _parse_er_sde_options(raw["er_sde"], field_name="extras.er_sde")
        # Hires options
        hires = raw.get('hires')
        hires_cfg: Optional[Dict[str, Any]] = None
        if hires is not None:
            if not isinstance(hires, dict):
                raise HTTPException(status_code=400, detail="'extras.hires' must be an object")
            _reject_unknown_keys(hires, _TXT2IMG_HIRES_KEYS | {"enable"}, "extras.hires")
            if bool(hires.get('enable')):
                required = ['denoise', 'scale', 'resize_x', 'resize_y', 'steps', 'upscaler']
                for key in required:
                    if key not in hires:
                        raise HTTPException(status_code=400, detail=f"Missing 'extras.hires.{key}'")
                hr_modules = hires.get('modules')
                if hr_modules is not None:
                    if not isinstance(hr_modules, list) or any(not isinstance(entry, str) for entry in hr_modules):
                        raise HTTPException(status_code=400, detail="'extras.hires.modules' must be an array of strings")
                    modules_list = list(hr_modules)
                else:
                    modules_list = []
                refiner_raw = hires.get('refiner')
                refiner_cfg: Optional[Dict[str, Any]] = None
                if refiner_raw is not None:
                    if not isinstance(refiner_raw, dict):
                        raise HTTPException(status_code=400, detail="'extras.hires.refiner' must be an object")
                    _reject_unknown_keys(refiner_raw, {"enable", "switch_at_step", "cfg", "seed", "model", "vae"}, "extras.hires.refiner")
                    if bool(refiner_raw.get('enable')):
                        refiner_cfg = {
                            "switch_at_step": _require_int_field(refiner_raw, 'switch_at_step', minimum=1),
                            "cfg": _require_float_field(refiner_raw, 'cfg'),
                            "seed": _require_int_field(refiner_raw, 'seed'),
                        }
                        if 'model' in refiner_raw:
                            refiner_cfg['model'] = str(refiner_raw['model'])
                        if 'vae' in refiner_raw:
                            refiner_cfg['vae'] = str(refiner_raw['vae'])
                try:
                    tile_cfg = tile_config_from_payload(hires.get("tile"), context="extras.hires.tile")
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from None
                tile = {
                    "tile": int(tile_cfg.tile),
                    "overlap": int(tile_cfg.overlap),
                    "fallback_on_oom": bool(tile_cfg.fallback_on_oom),
                    "min_tile": int(tile_cfg.min_tile),
                }
                hires_cfg = {
                    "denoise": _require_float_field(hires, 'denoise'),
                    "scale": _require_float_field(hires, 'scale'),
                    "resize_x": _require_int_field(hires, 'resize_x'),
                    "resize_y": _require_int_field(hires, 'resize_y'),
                    "steps": _require_int_field(hires, 'steps', minimum=0),
                    "upscaler": _require_str_field(hires, 'upscaler', allow_empty=False, trim=True),
                    "tile": tile,
                    "checkpoint": hires.get('checkpoint'),
                    "modules": modules_list,
                    "sampler": hires.get('sampler'),
                    "scheduler": hires.get('scheduler'),
                    "prompt": hires.get('prompt') or '',
                    "negative_prompt": hires.get('negative_prompt') or '',
                    "cfg": _require_float_field(hires, 'cfg') if hires.get('cfg') is not None else None,
                    "distilled_cfg": _require_float_field(hires, 'distilled_cfg') if hires.get('distilled_cfg') is not None else None,
                    "refiner": refiner_cfg,
                }

        # Swap-model options (global)
        refiner = raw.get('refiner')
        if refiner is not None:
            if not isinstance(refiner, dict):
                raise HTTPException(status_code=400, detail="'extras.refiner' must be an object")
            _reject_unknown_keys(refiner, {"enable", "switch_at_step", "cfg", "seed", "model", "vae"}, "extras.refiner")
            if bool(refiner.get('enable')):
                ref_cfg: Dict[str, Any] = {
                    "switch_at_step": _require_int_field(refiner, 'switch_at_step', minimum=1),
                    "cfg": _require_float_field(refiner, 'cfg'),
                    "seed": _require_int_field(refiner, 'seed'),
                }
                if 'model' in refiner:
                    ref_cfg['model'] = str(refiner['model'])
                if 'vae' in refiner:
                    ref_cfg['vae'] = str(refiner['vae'])
                extras['refiner'] = ref_cfg

        # Text encoder override (family + label [+ optional components])
        te_override = raw.get('text_encoder_override')
        if te_override is not None:
            if not isinstance(te_override, dict):
                raise HTTPException(status_code=400, detail="'extras.text_encoder_override' must be an object")
            _reject_unknown_keys(te_override, {"family", "label", "components"}, "extras.text_encoder_override")
            family_raw = te_override.get("family")
            label_raw = te_override.get("label")
            if not isinstance(family_raw, str) or not family_raw.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'extras.text_encoder_override.family' must be a non-empty string",
                )
            if not isinstance(label_raw, str) or not label_raw.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'extras.text_encoder_override.label' must be a non-empty string",
                )
            family = family_raw.strip()
            label = label_raw.strip()
            # Cheap sanity: UI labels use the pattern '<family>/<path>' (paths.json via /api/paths).
            if "/" in label and not label.startswith(f"{family}/"):
                raise HTTPException(
                    status_code=400,
                    detail="extras.text_encoder_override.label must start with '<family>/'",
                )
            components_val = te_override.get("components")
            components: list[str] | None = None
            if components_val is not None:
                if not isinstance(components_val, list) or any(not isinstance(c, str) for c in components_val):
                    raise HTTPException(
                        status_code=400,
                        detail="'extras.text_encoder_override.components' must be an array of strings",
                    )
                components = [c.strip() for c in components_val if isinstance(c, str) and c.strip()]
            te_cfg: Dict[str, Any] = {"family": family, "label": label}
            if components:
                te_cfg["components"] = components
            extras["text_encoder_override"] = te_cfg

        return extras, hires_cfg


    def _resolve_model_ref_from_sha_or_name(
        *,
        model_override: Any,
        extras: Dict[str, Any],
        field_prefix: str,
        models_api: Any,
    ) -> str:
        """Resolve the checkpoint reference for a request.

        - Prefer `<field_prefix>.model_sha` when present.
        - Back-compat: accept `model` when it looks like a sha (10 or 64 hex).
        - On match, update `extras["model_path"]` so downstream stages can surface the resolved filename.
        """

        model_sha = extras.get("model_sha")
        sha_candidate = None
        if isinstance(model_sha, str) and model_sha.strip():
            sha_candidate = model_sha.strip()
        elif isinstance(model_override, str):
            maybe = model_override.strip()
            if len(maybe) in (10, 64) and all(c in "0123456789abcdef" for c in maybe.lower()):
                sha_candidate = maybe

        resolved = model_override
        if sha_candidate:
            record = models_api.find_checkpoint_by_sha(sha_candidate)
            if record is None:
                raise HTTPException(status_code=409, detail=f"Checkpoint not found for sha: {sha_candidate}")
            resolved = record.filename
            extras["model_path"] = record.filename

        if not isinstance(resolved, str) or not resolved.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Missing model selection: provide 'model' or '{field_prefix}.model_sha'",
            )

        return resolved.strip()

    def _build_hires(cfg: Optional[Dict[str, Any]], width: int, height: int, fallback_cfg: float, fallback_distilled: float = 3.5) -> Dict[str, Any]:
        if cfg is None:
            return {
                "enable": False,
                "denoise": 0.0,
                "scale": 1.0,
                "upscaler": "Use same upscaler",
                "steps": 0,
                "resize_x": width,
                "resize_y": height,
                "hr_checkpoint_name": "Use same checkpoint",
                "hr_additional_modules": [],
                "hr_sampler_name": "Use same sampler",
                "hr_scheduler": "Use same scheduler",
                "hr_prompt": "",
                "hr_negative_prompt": "",
                "hr_cfg": fallback_cfg,
                "hr_distilled_cfg": fallback_distilled,
                "refiner": None,
            }
        return {
            "enable": True,
            "denoise": cfg["denoise"],
            "scale": cfg["scale"],
            "upscaler": cfg["upscaler"],
            "tile": cfg.get("tile"),
            "steps": cfg["steps"],
            "resize_x": cfg["resize_x"],
            "resize_y": cfg["resize_y"],
            "hr_checkpoint_name": cfg.get("checkpoint") or "Use same checkpoint",
            "hr_additional_modules": cfg.get("modules") or [],
            "hr_sampler_name": cfg.get("sampler") or "Use same sampler",
            "hr_scheduler": cfg.get("scheduler") or "Use same scheduler",
            "hr_prompt": cfg.get("prompt") or "",
            "hr_negative_prompt": cfg.get("negative_prompt") or "",
            "hr_cfg": cfg.get("cfg") if cfg.get("cfg") is not None else fallback_cfg,
            "hr_distilled_cfg": cfg.get("distilled_cfg") if cfg.get("distilled_cfg") is not None else fallback_distilled,
            "refiner": cfg.get("refiner"),
        }

    def _canonical_engine_key(value: object) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        key = raw.lower()
        from apps.backend.core.registry import registry as _engine_registry
        try:
            _ensure_default_engines_registered()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Engine registry init failed: {exc}") from exc
        try:
            return _engine_registry.get_descriptor(key).key
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unknown engine key: {key}") from exc

    def _parse_optional_sampler_field(*, value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_name}' must be a string")
        sampler = value.strip()
        if not sampler:
            raise HTTPException(status_code=400, detail=f"'{field_name}' must not be empty")
        return sampler

    def _validate_er_sde_release_scope(*, engine_key: str, sampler: str, field_name: str) -> None:
        if str(sampler).strip().lower() != "er sde":
            return
        if engine_key == SemanticEngine.ANIMA.value:
            return
        raise HTTPException(
            status_code=400,
            detail=(
                f"Sampler 'er sde' in '{field_name}' is currently enabled only for engine 'anima'."
            ),
        )

    def _validate_anima_sampler_allowlist(*, engine_key: str, sampler: str, field_name: str) -> None:
        if engine_key != SemanticEngine.ANIMA.value:
            return
        if sampler in _ANIMA_ALLOWED_SAMPLERS:
            return
        allowed = ", ".join(_ANIMA_ALLOWED_SAMPLERS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported sampler for engine 'anima' in '{field_name}': '{sampler}'. Allowed: {allowed}",
        )

    def _validate_swap_at_step_pointer(*, pointer: int, total_steps: int, field_name: str) -> None:
        if total_steps < 2:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' requires total steps >= 2 (got {total_steps})",
            )
        if pointer < 1 or pointer >= total_steps:
            raise HTTPException(
                status_code=400,
                detail=f"'{field_name}' must be in [1, {total_steps - 1}] (got {pointer})",
            )

    def _validate_prompt_sampler_controls(*, engine_key: str, prompt: str, field_name: str) -> None:
        for match in _PROMPT_SAMPLER_CONTROL_RE.finditer(prompt):
            sampler_raw = match.group(1)
            sampler = str(sampler_raw or "").strip().lower()
            if not sampler:
                continue
            _validate_er_sde_release_scope(
                engine_key=engine_key,
                sampler=sampler,
                field_name=f"{field_name} (prompt <sampler:...> control)",
            )
            _validate_anima_sampler_allowlist(
                engine_key=engine_key,
                sampler=sampler,
                field_name=f"{field_name} (prompt <sampler:...> control)",
            )

    def _is_gguf_checkpoint(_models_api: Any, model_ref: object) -> bool:
        raw = str(model_ref or "").strip()
        if not raw:
            return False
        if Path(raw).suffix.lower() == ".gguf":
            return True
        try:
            record = _models_api.find_checkpoint(raw)
        except Exception:
            record = None
        if record is None:
            return False
        core_only = getattr(record, "core_only", None)
        if isinstance(core_only, bool):
            return bool(core_only)
        filename = str(getattr(record, "filename", "") or "")
        return Path(filename).suffix.lower() == ".gguf"

    from apps.backend.core.contracts.asset_requirements import (
        EngineAssetContract,
        contract_for_request,
        format_text_encoder_kind_label,
    )

    def _normalize_sha_field(value: object, *, field_label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string")
        norm = value.strip().lower()
        return norm or None

    def _normalize_sha_list_field(value: object, *, field_label: str) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            sha = value.strip().lower()
            return [sha] if sha else []
        if isinstance(value, list):
            out: list[str] = []
            for entry in value:
                if not isinstance(entry, str):
                    raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string or array of strings")
                sha = entry.strip().lower()
                if sha:
                    out.append(sha)
            return out
        raise HTTPException(status_code=400, detail=f"'{field_label}' must be a string or array of strings")

    def _format_required_tenc_message(
        *,
        engine_id: str,
        contract: EngineAssetContract,
        field_label: str,
    ) -> str:
        count = int(contract.tenc_count)
        kind = format_text_encoder_kind_label(contract.tenc_kind)
        if count == 1:
            return f"Engine '{engine_id}' requires exactly 1 text encoder ({kind}) via '{field_label}'"
        return f"Engine '{engine_id}' requires exactly {count} text encoders ({kind}) via '{field_label}'"

    def _apply_asset_contract_to_extras(
        *,
        engine_id: str,
        checkpoint_ref: object,
        extras: Dict[str, Any],
        field_prefix: str,
        resolve_asset_by_sha,  # type: ignore[no-untyped-def]
        models_api: Any,
    ) -> None:
        if "vae_path" in extras or "tenc_path" in extras:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{field_prefix} must not include raw '*_path' fields; use sha256 via '{field_prefix}.*_sha'"
                ),
            )

        if engine_id in ("flux1", "flux1_kontext") and "text_encoder_override" in extras:
            raise HTTPException(
                status_code=400,
                detail=f"Do not send {field_prefix}.text_encoder_override for Flux.1; use {field_prefix}.tenc_sha only.",
            )

        is_core_only = _is_gguf_checkpoint(models_api, checkpoint_ref)
        try:
            contract = contract_for_request(engine_id=engine_id, checkpoint_core_only=bool(is_core_only))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Asset contract resolution failed for engine '{engine_id}': {exc}",
            ) from exc

        vae_field = f"{field_prefix}.vae_sha"
        tenc_field = f"{field_prefix}.tenc_sha"

        vae_sha = _normalize_sha_field(extras.get("vae_sha"), field_label=vae_field)
        tenc_shas = _normalize_sha_list_field(extras.get("tenc_sha"), field_label=tenc_field)

        if contract.requires_vae and not vae_sha:
            raise HTTPException(status_code=400, detail=f"Engine '{engine_id}' requires '{vae_field}' (sha256)")

        if contract.requires_text_encoders:
            if len(tenc_shas) == 0:
                raise HTTPException(status_code=400, detail=f"Engine '{engine_id}' requires '{tenc_field}' (sha256)")
            if len(tenc_shas) != int(contract.tenc_count):
                raise HTTPException(
                    status_code=400,
                    detail=_format_required_tenc_message(engine_id=engine_id, contract=contract, field_label=tenc_field),
                )

        if vae_sha:
            vae_path = resolve_asset_by_sha(vae_sha)
            if not vae_path:
                raise HTTPException(status_code=409, detail=f"Asset not found for sha: {vae_sha}")
            extras["vae_path"] = vae_path

        if tenc_shas:
            tenc_paths: list[str] = []
            for sha in tenc_shas:
                path = resolve_asset_by_sha(sha)
                if not path:
                    raise HTTPException(status_code=409, detail=f"Asset not found for sha: {sha}")
                tenc_paths.append(path)

            slot_to_path: dict[str, str] | None = None
            if contract.requires_text_encoders and contract.tenc_slots:
                from apps.backend.core.contracts.text_encoder_slots import (
                    TextEncoderSlotError,
                    classify_text_encoder_slot,
                )
                from apps.backend.inventory.cache import resolve_text_encoder_slot_by_sha

                try:
                    expected = tuple(contract.tenc_slots)
                    slot_to_path = {}
                    for sha, path in zip(tenc_shas, tenc_paths):
                        slot = resolve_text_encoder_slot_by_sha(sha) or ""
                        if not slot:
                            slot = classify_text_encoder_slot(path)
                        if slot not in expected:
                            raise TextEncoderSlotError(
                                f"Text encoder slot mismatch: got slot={slot!r} for sha={sha!r}, expected one of {list(expected)}."
                            )
                        if slot in slot_to_path:
                            raise TextEncoderSlotError(
                                f"Duplicate text encoder slot {slot!r} for slots={list(expected)} (sha={sha!r})."
                            )
                        slot_to_path[slot] = path

                    missing = [slot for slot in expected if slot not in slot_to_path]
                    if missing:
                        raise TextEncoderSlotError(
                            f"Missing required text encoder slot(s) {missing} for slots={list(expected)} (classified={sorted(slot_to_path)})."
                        )
                except TextEncoderSlotError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                # Normalize order to the canonical slot list so downstream code never depends on user-provided ordering.
                tenc_paths = [slot_to_path[slot] for slot in contract.tenc_slots]

            extras["tenc_path"] = tenc_paths[0] if len(tenc_paths) == 1 else tenc_paths

            # Flux.1/Kontext: translate sha-selected encoders into a loader override (paths stay server-side).
            if engine_id in ("flux1", "flux1_kontext"):
                if slot_to_path is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Internal error: slot mapping missing for engine '{engine_id}'",
                    )
                extras["text_encoder_override"] = {
                    "family": engine_id,
                    "label": f"{engine_id}/sha",
                    "components": [f"{slot}={slot_to_path[slot]}" for slot in contract.tenc_slots],
                }

    def prepare_txt2img(payload: Dict[str, Any]) -> Tuple["Txt2ImgRequest", str, Optional[str]]:
        _reject_unknown_keys(payload, _TXT2IMG_ALLOWED_KEYS, "txt2img")
        engine_override = payload.get('engine')
        model_override = payload.get('model')
        engine_key = _canonical_engine_key(engine_override)
        if not engine_key:
            raise HTTPException(status_code=400, detail="Missing engine key (engine)")
        engine_id = engine_key

        prompt = _require_str_field(payload, 'prompt', allow_empty=True)
        negative_prompt = str(payload.get('negative_prompt') or '')
        _validate_prompt_sampler_controls(
            engine_key=engine_key,
            prompt=prompt,
            field_name="prompt",
        )
        width = _require_int_field(payload, 'width', minimum=8)
        height = _require_int_field(payload, 'height', minimum=8)
        steps_val = _require_int_field(payload, 'steps', minimum=1)
        distilled_guidance_engines = {"flux1", "flux1_kontext", "flux1_chroma"}
        if engine_id in distilled_guidance_engines:
            if 'cfg' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"Engine '{engine_id}' does not accept 'cfg'; use 'distilled_cfg'.",
                )
            if 'distilled_cfg' not in payload:
                raise HTTPException(status_code=400, detail=f"Engine '{engine_id}' requires 'distilled_cfg'.")
            # Flow models (Flux/Chroma) use distilled guidance (no classic CFG); keep cfg neutral.
            cfg_scale = 1.0
            distilled_cfg_scale = _require_float_field(payload, 'distilled_cfg')
        else:
            if 'distilled_cfg' in payload:
                raise HTTPException(
                    status_code=400,
                    detail=f"'distilled_cfg' is only supported for engines: {sorted(distilled_guidance_engines)}",
                )
            if 'cfg' not in payload:
                raise HTTPException(status_code=400, detail="Missing 'cfg'")
            # Z-Image uses classic CFG semantics (diffusers parity).
            cfg_scale = _require_float_field(payload, 'cfg')
            distilled_cfg_scale = 3.5
        sampler_name = _require_str_field(payload, 'sampler', allow_empty=False)
        scheduler_name = _require_str_field(payload, 'scheduler', allow_empty=False)
        _validate_er_sde_release_scope(
            engine_key=engine_key,
            sampler=sampler_name,
            field_name="sampler",
        )
        _validate_anima_sampler_allowlist(engine_key=engine_key, sampler=sampler_name, field_name="sampler")
        try:
            from apps.backend.runtime.sampling.registry import get_sampler_spec
            from apps.backend.runtime.sampling.context import SchedulerName

            spec = get_sampler_spec(str(sampler_name))
            SchedulerName.from_string(str(scheduler_name))
            if not spec.is_supported_scheduler(str(scheduler_name)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{sampler_name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        seed_val = _require_int_field(payload, 'seed')
        clip_skip = _require_int_field(payload, 'clip_skip', minimum=0, maximum=12) if 'clip_skip' in payload else None
        styles = _parse_styles(payload)
        metadata = _parse_metadata(payload)
        extras, hires_cfg = _parse_txt2img_extras(payload)
        if hires_cfg is not None:
            hires_prompt = str(hires_cfg.get("prompt") or "")
            _validate_prompt_sampler_controls(
                engine_key=engine_key,
                prompt=hires_prompt,
                field_name="extras.hires.prompt",
            )
            hires_sampler = _parse_optional_sampler_field(value=hires_cfg.get("sampler"), field_name="extras.hires.sampler")
            if hires_sampler is not None:
                hires_cfg["sampler"] = hires_sampler
                _validate_er_sde_release_scope(
                    engine_key=engine_key,
                    sampler=hires_sampler,
                    field_name="extras.hires.sampler",
                )
                _validate_anima_sampler_allowlist(
                    engine_key=engine_key,
                    sampler=hires_sampler,
                    field_name="extras.hires.sampler",
                )
            hires_refiner_cfg = hires_cfg.get("refiner")
            if isinstance(hires_refiner_cfg, dict):
                hires_total_steps = int(hires_cfg.get("steps") or 0)
                if hires_total_steps <= 0:
                    hires_total_steps = int(steps_val)
                _validate_swap_at_step_pointer(
                    pointer=int(hires_refiner_cfg.get("switch_at_step", 0)),
                    total_steps=hires_total_steps,
                    field_name="extras.hires.refiner.switch_at_step",
                )
        global_refiner_cfg = extras.get("refiner")
        if isinstance(global_refiner_cfg, dict):
            _validate_swap_at_step_pointer(
                pointer=int(global_refiner_cfg.get("switch_at_step", 0)),
                total_steps=int(steps_val),
                field_name="extras.refiner.switch_at_step",
            )

        # Read batch params from extras (default to 1)
        batch_size = int(extras.pop('batch_size', 1)) if 'batch_size' in extras else 1
        batch_count = int(extras.pop('batch_count', 1)) if 'batch_count' in extras else 1

        metadata["styles"] = styles
        metadata["n_iter"] = batch_count
        metadata["batch_count"] = batch_count
        metadata["batch_size"] = batch_size
        metadata["hr"] = bool(hires_cfg)
        metadata["distilled_cfg_scale"] = distilled_cfg_scale

        # Smart offload/fallback flags: prefer payload when present, otherwise fall back to options snapshot.
        snap = _opts_snapshot()
        if "smart_offload" in payload:
            smart_offload = bool(payload.get("smart_offload"))
        else:
            smart_offload = bool(getattr(snap, "codex_smart_offload", False))
        if "smart_fallback" in payload:
            smart_fallback = bool(payload.get("smart_fallback"))
        else:
            smart_fallback = bool(getattr(snap, "codex_smart_fallback", False))
        if "smart_cache" in payload:
            smart_cache = bool(payload.get("smart_cache"))
        else:
            smart_cache = bool(getattr(snap, "codex_smart_cache", False))

        # Resolve model assets from SHA (if provided in extras)
        from apps.backend.inventory.cache import resolve_asset_by_sha
        from apps.backend.runtime.models import api as _models_api
        model_override = _resolve_model_ref_from_sha_or_name(
            model_override=model_override,
            extras=extras,
            field_prefix="extras",
            models_api=_models_api,
        )
        model_ref_for_contract = model_override
        _apply_asset_contract_to_extras(
            engine_id=engine_id,
            checkpoint_ref=model_ref_for_contract,
            extras=extras,
            field_prefix="extras",
            resolve_asset_by_sha=resolve_asset_by_sha,
            models_api=_models_api,
        )

        req = Txt2ImgRequest(
            task=TaskType.TXT2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps_val,
            guidance_scale=cfg_scale,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            batch_size=batch_size,
            clip_skip=clip_skip,
            metadata=metadata,
            hires=_build_hires(hires_cfg, width, height, cfg_scale, distilled_cfg_scale) if hires_cfg is not None else None,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
        )

        return req, engine_key, model_override

    def _parse_explicit_device(payload: Dict[str, Any]) -> str:
        """Parse/validate the per-request device selection (fail loud).

        Note: do not apply `switch_primary_device()` here; apply it only when the task actually starts running
        (single-flight-safe).
        """
        try:
            return parse_device_from_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    _ORCH = InferenceOrchestrator()


    def run_txt2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, *, device: str) -> None:
        from apps.backend.interfaces.api.tasks.generation_tasks import run_image_task as _run_image_task

        _run_image_task(
            task_id=task_id,
            payload=payload,
            entry=entry,
            device=device,
            task_type=TaskType.TXT2IMG,
            prepare=prepare_txt2img,
            orch=_ORCH,
            ensure_default_engines_registered=_ensure_default_engines_registered,
            live_preview=live_preview,
            opts_get=_opts_get,
            opts_snapshot=_opts_snapshot,
            generation_provenance=_GENERATION_PROVENANCE,
            save_generated_images=_save_generated_images,
        )

    def prepare_img2img(payload: Dict[str, Any]) -> Tuple[Img2ImgRequest, str, Optional[str]]:
        init_image_data = _p.require(payload, 'img2img_init_image')
        init_image = media.decode_image(init_image_data)
        init_w, init_h = 0, 0
        try:
            init_w, init_h = init_image.size  # type: ignore[attr-defined]
        except Exception:
            init_w, init_h = 0, 0
        mask_data = payload.get('img2img_mask')
        mask_image = media.decode_image(mask_data) if mask_data else None

        mask_enforcement = None
        inpainting_fill = 0
        inpaint_full_res = True
        inpaint_full_res_padding = 0
        inpainting_mask_invert = 0
        mask_blur = 4
        mask_blur_x = 4
        mask_blur_y = 4
        mask_round = True

        if mask_image is not None:
            raw_enforcement = payload.get("img2img_mask_enforcement")
            if not isinstance(raw_enforcement, str) or not raw_enforcement.strip():
                raise HTTPException(
                    status_code=400,
                    detail="'img2img_mask_enforcement' is required when 'img2img_mask' is provided",
                )
            mask_enforcement = raw_enforcement.strip()
            if mask_enforcement not in ("post_blend", "per_step_clamp"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid 'img2img_mask_enforcement' (allowed: post_blend, per_step_clamp)",
                )

            if "img2img_inpainting_fill" in payload:
                inpainting_fill = _p.as_int(payload, "img2img_inpainting_fill")
            if inpainting_fill not in (0, 1, 2, 3):
                raise HTTPException(status_code=400, detail="'img2img_inpainting_fill' must be 0,1,2,3")

            if "img2img_inpaint_full_res" in payload:
                inpaint_full_res = _p.as_bool(payload, "img2img_inpaint_full_res")
            if "img2img_inpaint_full_res_padding" in payload:
                inpaint_full_res_padding = _p.as_int(payload, "img2img_inpaint_full_res_padding")
            if inpaint_full_res_padding < 0:
                raise HTTPException(status_code=400, detail="'img2img_inpaint_full_res_padding' must be >= 0")

            if "img2img_inpainting_mask_invert" in payload:
                inpainting_mask_invert = _p.as_int(payload, "img2img_inpainting_mask_invert")
            if inpainting_mask_invert not in (0, 1):
                raise HTTPException(status_code=400, detail="'img2img_inpainting_mask_invert' must be 0 or 1")

            if "img2img_mask_blur" in payload:
                mask_blur = _p.as_int(payload, "img2img_mask_blur")
                mask_blur_x = mask_blur
                mask_blur_y = mask_blur
            if "img2img_mask_blur_x" in payload:
                mask_blur_x = _p.as_int(payload, "img2img_mask_blur_x")
            if "img2img_mask_blur_y" in payload:
                mask_blur_y = _p.as_int(payload, "img2img_mask_blur_y")
            if mask_blur_x < 0 or mask_blur_y < 0:
                raise HTTPException(status_code=400, detail="'img2img_mask_blur' must be >= 0")

            if "img2img_mask_round" in payload:
                mask_round = _p.as_bool(payload, "img2img_mask_round")
        else:
            raw_enforcement = payload.get("img2img_mask_enforcement")
            if isinstance(raw_enforcement, str) and raw_enforcement.strip():
                raise HTTPException(status_code=400, detail="'img2img_mask_enforcement' requires 'img2img_mask'")

        engine_override = payload.get('engine')
        model_override = payload.get('model')
        engine_key = _canonical_engine_key(engine_override)
        if not engine_key:
            raise HTTPException(status_code=400, detail="Missing engine key (engine)")
        model_ref = model_override

        prompt = _p.require(payload, 'img2img_prompt') or ''
        negative_prompt = _p.require(payload, 'img2img_neg_prompt') or ''
        _validate_prompt_sampler_controls(
            engine_key=engine_key,
            prompt=str(prompt),
            field_name="img2img_prompt",
        )
        styles = _p.as_list(payload, 'img2img_styles') if 'img2img_styles' in payload else []
        batch_count = _p.as_int(payload, 'img2img_batch_count') if 'img2img_batch_count' in payload else 1
        batch_size = _p.as_int(payload, 'img2img_batch_size') if 'img2img_batch_size' in payload else 1
        if 'img2img_steps' in payload:
            steps_val = _p.as_int(payload, 'img2img_steps')
        else:
            raise HTTPException(status_code=400, detail="'img2img_steps' is required")

        if 'img2img_cfg_scale' in payload:
            cfg_scale = _require_float_field(payload, 'img2img_cfg_scale')
        else:
            raise HTTPException(status_code=400, detail="'img2img_cfg_scale' is required")

        distilled_cfg_scale = _require_float_field(payload, 'img2img_distilled_cfg_scale') if 'img2img_distilled_cfg_scale' in payload else None
        image_cfg_scale = _require_float_field(payload, 'img2img_image_cfg_scale') if 'img2img_image_cfg_scale' in payload else None
        denoise = _require_float_field(payload, 'img2img_denoising_strength')
        def _snap_dim(value: int) -> int:
            if not value:
                return 0
            value = max(8, min(8192, int(value)))
            return int(((value + 4) // 8) * 8)

        if 'img2img_width' in payload:
            width_val = _p.as_int(payload, 'img2img_width')
        else:
            width_val = _snap_dim(int(init_w) if init_w else 0)
            if not width_val:
                raise HTTPException(status_code=400, detail="'img2img_width' is required")

        if 'img2img_height' in payload:
            height_val = _p.as_int(payload, 'img2img_height')
        else:
            height_val = _snap_dim(int(init_h) if init_h else 0)
            if not height_val:
                raise HTTPException(status_code=400, detail="'img2img_height' is required")
        sampler_name = _p.require(payload, 'img2img_sampling')
        scheduler_name = _p.require(payload, 'img2img_scheduler')
        _validate_er_sde_release_scope(
            engine_key=engine_key,
            sampler=str(sampler_name),
            field_name="img2img_sampling",
        )
        _validate_anima_sampler_allowlist(
            engine_key=engine_key,
            sampler=str(sampler_name),
            field_name="img2img_sampling",
        )
        try:
            from apps.backend.runtime.sampling.registry import get_sampler_spec
            from apps.backend.runtime.sampling.context import SchedulerName

            spec = get_sampler_spec(str(sampler_name))
            SchedulerName.from_string(str(scheduler_name))
            if not spec.is_supported_scheduler(str(scheduler_name)):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{sampler_name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    ),
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        seed_val = _p.as_int(payload, 'img2img_seed')
        clip_skip = _p.as_int(payload, 'img2img_clip_skip') if 'img2img_clip_skip' in payload else None
        if clip_skip is not None:
            if clip_skip < 0 or clip_skip > 12:
                raise HTTPException(status_code=400, detail="'img2img_clip_skip' must be in [0, 12]")
        noise_source = payload.get('img2img_randn_source') or payload.get('img2img_noise_source')
        ensd_raw = payload.get('img2img_eta_noise_seed_delta')

        def _reject_legacy_hires_keys(payload: Mapping[str, Any]) -> None:
            prefix = "img2img_"
            legacy_marker = "hr_"
            for key in payload.keys():
                if not isinstance(key, str):
                    continue
                if key.startswith(prefix) and key[len(prefix):].startswith(legacy_marker):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported legacy hires key: {key}. Use 'img2img_hires_*'.",
                    )

        _reject_legacy_hires_keys(payload)

        enable_hires = _p.as_bool(payload, 'img2img_hires_enable') if 'img2img_hires_enable' in payload else False
        if enable_hires:
            try:
                hr_tile_cfg = tile_config_from_payload(payload.get("img2img_hires_tile"), context="img2img_hires_tile")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            hr_tile = {
                "tile": int(hr_tile_cfg.tile),
                "overlap": int(hr_tile_cfg.overlap),
                "fallback_on_oom": bool(hr_tile_cfg.fallback_on_oom),
                "min_tile": int(hr_tile_cfg.min_tile),
            }
            hr_sampler_name = payload.get("img2img_hires_sampling")
            hr_sampler_name = _parse_optional_sampler_field(
                value=hr_sampler_name,
                field_name="img2img_hires_sampling",
            )
            if hr_sampler_name is not None:
                _validate_er_sde_release_scope(
                    engine_key=engine_key,
                    sampler=hr_sampler_name,
                    field_name="img2img_hires_sampling",
                )
                _validate_anima_sampler_allowlist(
                    engine_key=engine_key,
                    sampler=hr_sampler_name,
                    field_name="img2img_hires_sampling",
                )
            hr_scheduler = payload.get("img2img_hires_scheduler")
            if hr_scheduler is not None:
                if not isinstance(hr_scheduler, str):
                    raise HTTPException(status_code=400, detail="'img2img_hires_scheduler' must be a string")
                if not hr_scheduler.strip():
                    raise HTTPException(status_code=400, detail="'img2img_hires_scheduler' must not be empty")
            hires_data = {
                "enable": True,
                "scale": _require_float_field(payload, 'img2img_hires_scale') if 'img2img_hires_scale' in payload else 1.0,
                "resize_x": _p.as_int(payload, 'img2img_hires_resize_x') if 'img2img_hires_resize_x' in payload else 0,
                "resize_y": _p.as_int(payload, 'img2img_hires_resize_y') if 'img2img_hires_resize_y' in payload else 0,
                "steps": _p.as_int(payload, 'img2img_hires_steps') if 'img2img_hires_steps' in payload else 0,
                "denoise": _require_float_field(payload, 'img2img_hires_denoise') if 'img2img_hires_denoise' in payload else denoise,
                "upscaler": payload.get('img2img_hires_upscaler', 'Latent'),
                "tile": hr_tile,
                "hr_sampler_name": hr_sampler_name,
                "hr_scheduler": hr_scheduler.strip() if isinstance(hr_scheduler, str) and hr_scheduler.strip() else None,
                "hr_prompt": payload.get('img2img_hires_prompt', ''),
                "hr_negative_prompt": payload.get('img2img_hires_neg_prompt', ''),
                "hr_cfg": _require_float_field(payload, 'img2img_hires_cfg') if 'img2img_hires_cfg' in payload else cfg_scale,
                "hr_distilled_cfg": _require_float_field(payload, 'img2img_hires_distilled_cfg') if 'img2img_hires_distilled_cfg' in payload else (distilled_cfg_scale or 3.5),
            }
            _validate_prompt_sampler_controls(
                engine_key=engine_key,
                prompt=str(hires_data.get("hr_prompt") or ""),
                field_name="img2img_hires_prompt",
            )
        else:
            hires_data = {"enable": False}

        extras: Dict[str, Any] = {}
        raw_extras = payload.get("img2img_extras")
        if raw_extras is not None:
            if not isinstance(raw_extras, dict):
                raise HTTPException(status_code=400, detail="'img2img_extras' must be an object")
            _reject_unknown_keys(raw_extras, _IMG2IMG_EXTRAS_KEYS, "img2img_extras")
            raw_extras = dict(raw_extras)

            te_override = raw_extras.get("text_encoder_override")
            if te_override is not None:
                if not isinstance(te_override, dict):
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override' must be an object")
                _reject_unknown_keys(te_override, {"family", "label", "components"}, "img2img_extras.text_encoder_override")
                family_raw = te_override.get("family")
                label_raw = te_override.get("label")
                if not isinstance(family_raw, str) or not family_raw.strip():
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.family' must be a non-empty string")
                if not isinstance(label_raw, str) or not label_raw.strip():
                    raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.label' must be a non-empty string")
                family = family_raw.strip()
                label = label_raw.strip()
                if "/" in label and not label.startswith(f"{family}/"):
                    raise HTTPException(
                        status_code=400,
                        detail="img2img_extras.text_encoder_override.label must start with '<family>/'",
                    )
                components_val = te_override.get("components")
                components: list[str] | None = None
                if components_val is not None:
                    if not isinstance(components_val, list) or any(not isinstance(c, str) for c in components_val):
                        raise HTTPException(status_code=400, detail="'img2img_extras.text_encoder_override.components' must be an array of strings")
                    components = [c.strip() for c in components_val if isinstance(c, str) and c.strip()]
                te_cfg: Dict[str, Any] = {"family": family, "label": label}
                if components:
                    te_cfg["components"] = components
                raw_extras["text_encoder_override"] = te_cfg

            if "er_sde" in raw_extras:
                raw_extras["er_sde"] = _parse_er_sde_options(
                    raw_extras["er_sde"],
                    field_name="img2img_extras.er_sde",
                )

            extras.update(raw_extras)
        # Z-Image variant selection (Turbo/Base) for img2img runs.
        if "zimage_variant" in extras:
            val = extras.get("zimage_variant")
            if val is None:
                extras.pop("zimage_variant", None)
            elif not isinstance(val, str):
                raise HTTPException(status_code=400, detail="'img2img_extras.zimage_variant' must be a string")
            else:
                variant = val.strip().lower()
                if not variant:
                    extras.pop("zimage_variant", None)
                elif variant not in {"turbo", "base"}:
                    raise HTTPException(
                        status_code=400,
                        detail="'img2img_extras.zimage_variant' must be one of: turbo, base",
                    )
                else:
                    extras["zimage_variant"] = variant
        if noise_source:
            extras['randn_source'] = str(noise_source)
        if ensd_raw is not None:
            try:
                extras['eta_noise_seed_delta'] = int(float(ensd_raw))
            except Exception:
                raise HTTPException(status_code=400, detail="img2img_eta_noise_seed_delta must be numeric")

        # Resolve SHA-based assets (if provided in img2img_extras)
        from apps.backend.inventory.cache import resolve_asset_by_sha
        from apps.backend.runtime.models import api as _models_api
        engine_id = engine_key

        if "vae_path" in extras or "tenc_path" in extras:
            raise HTTPException(status_code=400, detail="img2img_extras must not include raw '*_path' fields; use sha256 via '*_sha'")

        model_ref = _resolve_model_ref_from_sha_or_name(
            model_override=model_ref,
            extras=extras,
            field_prefix="img2img_extras",
            models_api=_models_api,
        )

        _apply_asset_contract_to_extras(
            engine_id=engine_id,
            checkpoint_ref=model_ref,
            extras=extras,
            field_prefix="img2img_extras",
            resolve_asset_by_sha=resolve_asset_by_sha,
            models_api=_models_api,
        )

        metadata = {
            "styles": styles,
            "distilled_cfg_scale": distilled_cfg_scale,
            "image_cfg_scale": image_cfg_scale,
            "batch_count": batch_count,
        }
        if noise_source:
            metadata["randn_source"] = str(noise_source)
        if 'eta_noise_seed_delta' in extras:
            metadata["eta_noise_seed_delta"] = extras['eta_noise_seed_delta']

        req = Img2ImgRequest(
            task=TaskType.IMG2IMG,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sampler=str(sampler_name),
            scheduler=str(scheduler_name),
            seed=seed_val,
            guidance_scale=cfg_scale,
            batch_size=batch_size,
            clip_skip=clip_skip,
            metadata=metadata,
            init_image=init_image,
            mask=mask_image,
            mask_enforcement=mask_enforcement,
            inpainting_fill=inpainting_fill,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            mask_blur=mask_blur,
            mask_blur_x=mask_blur_x,
            mask_blur_y=mask_blur_y,
            mask_round=mask_round,
            denoise_strength=denoise,
            width=width_val,
            height=height_val,
            steps=steps_val,
            extras=extras,
            hires=hires_data if hires_data.get("enable") else None,
            smart_offload=bool(payload.get("smart_offload")) if payload.get("smart_offload") is not None else bool(getattr(_opts_snapshot(), "codex_smart_offload", False)),
            smart_fallback=bool(payload.get("smart_fallback")) if payload.get("smart_fallback") is not None else bool(getattr(_opts_snapshot(), "codex_smart_fallback", False)),
            smart_cache=bool(payload.get("smart_cache")) if payload.get("smart_cache") is not None else bool(getattr(_opts_snapshot(), "codex_smart_cache", False)),
        )

        return req, engine_key, model_ref

    def run_img2img_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, *, device: str) -> None:
        from apps.backend.interfaces.api.tasks.generation_tasks import run_image_task as _run_image_task

        _run_image_task(
            task_id=task_id,
            payload=payload,
            entry=entry,
            device=device,
            task_type=TaskType.IMG2IMG,
            prepare=prepare_img2img,
            orch=_ORCH,
            ensure_default_engines_registered=_ensure_default_engines_registered,
            live_preview=live_preview,
            opts_get=_opts_get,
            opts_snapshot=_opts_snapshot,
            generation_provenance=_GENERATION_PROVENANCE,
            save_generated_images=_save_generated_images,
        )

    def _wan_require_dims_multiple_of_16(*, task: str, width: int, height: int) -> None:
        """WAN video geometry guard (Diffusers parity).

        WAN requires width/height divisible by 16; otherwise the latent patch grid silently crops.
        The frontend rounds up, but the backend must fail loud for direct API callers.
        """

        if height % 16 == 0 and width % 16 == 0:
            return
        w_up = ((int(width) + 15) // 16) * 16
        h_up = ((int(height) + 15) // 16) * 16
        raise HTTPException(
            status_code=400,
            detail=(
                f"WAN22 {task}: width/height must be divisible by 16 (Diffusers parity). "
                f"Got {int(width)}x{int(height)}. Suggested: {w_up}x{h_up} (rounded up)."
            ),
        )

    def prepare_txt2vid(payload: Dict[str, Any]) -> Tuple[Txt2VidRequest, str, Optional[str]]:
        prompt = payload.get('txt2vid_prompt', '')
        negative_prompt = payload.get('txt2vid_neg_prompt', '')
        width_val = int(payload.get('txt2vid_width', 768))
        height_val = int(payload.get('txt2vid_height', 432))
        _wan_require_dims_multiple_of_16(task="txt2vid", width=width_val, height=height_val)
        steps_val = int(payload.get('txt2vid_steps', 30))
        fps_val = int(payload.get('txt2vid_fps', 24))
        frames_val = int(payload.get('txt2vid_num_frames', 16))
        sampler_name = str(payload.get('txt2vid_sampler', payload.get('txt2vid_sampling', 'uni-pc')))
        scheduler_name = str(payload.get('txt2vid_scheduler', 'simple'))
        try:
            from apps.backend.types.samplers import SamplerKind
            from apps.backend.runtime.sampling.context import SchedulerName

            SamplerKind.from_string(sampler_name)
            SchedulerName.from_string(scheduler_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        seed_val = int(payload.get('txt2vid_seed', -1))
        cfg_val = float(payload.get('txt2vid_cfg_scale', 7.0))

        extras: Dict[str, Any] = {}
        if "video_return_frames" in payload:
            raw_return_frames = payload.get("video_return_frames")
            if raw_return_frames is not None and not isinstance(raw_return_frames, bool):
                raise HTTPException(status_code=400, detail="'video_return_frames' must be a boolean when provided")
            if isinstance(raw_return_frames, bool):
                extras["video_return_frames"] = bool(raw_return_frames)
        # Video export options (structured in request.video_options; also kept in extras.video for debugging)
        video_options = None
        try:
            from apps.backend.core.params.video import VideoExportOptions

            video_options = VideoExportOptions(
                filename_prefix=(str(payload.get("video_filename_prefix")).strip() if payload.get("video_filename_prefix") else None),
                format=(str(payload.get("video_format")).strip() if payload.get("video_format") else None),
                pix_fmt=(str(payload.get("video_pix_fmt")).strip() if payload.get("video_pix_fmt") else None),
                crf=(int(payload.get("video_crf")) if payload.get("video_crf") is not None else None),
                loop_count=(int(payload.get("video_loop_count")) if payload.get("video_loop_count") is not None else None),
                pingpong=(bool(payload.get("video_pingpong")) if payload.get("video_pingpong") is not None else None),
                save_metadata=(bool(payload.get("video_save_metadata")) if payload.get("video_save_metadata") is not None else None),
                save_output=(bool(payload.get("video_save_output")) if payload.get("video_save_output") is not None else None),
                trim_to_audio=(bool(payload.get("video_trim_to_audio")) if payload.get("video_trim_to_audio") is not None else None),
            ).as_dict()
        except Exception:
            video_options = None
        if video_options:
            extras["video"] = {
                "video_filename_prefix": payload.get("video_filename_prefix"),
                "video_format": payload.get("video_format"),
                "video_pix_fmt": payload.get("video_pix_fmt"),
                "video_crf": payload.get("video_crf"),
                "video_loop_count": payload.get("video_loop_count"),
                "video_pingpong": payload.get("video_pingpong"),
                "video_save_metadata": payload.get("video_save_metadata"),
                "video_save_output": payload.get("video_save_output"),
                "video_trim_to_audio": payload.get("video_trim_to_audio"),
            }
        if isinstance(payload.get('video_interpolation'), dict):
            extras['video_interpolation'] = payload.get('video_interpolation')
        # WAN (GGUF-only): strict sha-only selection for model parts (no raw paths).
        from apps.backend.inventory.cache import resolve_asset_by_sha

        def _require_sha_field(key: str) -> str:
            val = payload.get(key)
            if isinstance(val, dict):
                raise HTTPException(status_code=400, detail=f"'{key}' must be a string sha256, got object")
            if not isinstance(val, str) or not val.strip():
                raise HTTPException(status_code=400, detail=f"'{key}' is required and must be a non-empty sha256 string")
            return val.strip().lower()

        def _resolve_wan_stage(stage_key: str) -> dict[str, object]:
            raw = payload.get(stage_key)
            if not isinstance(raw, dict):
                raise HTTPException(status_code=400, detail=f"'{stage_key}' is required and must be an object")
            if isinstance(raw.get("model_dir"), str) and str(raw.get("model_dir")).strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.model_dir' is unsupported; use '{stage_key}.model_sha'")
            if isinstance(raw.get("lora_path"), str) and str(raw.get("lora_path")).strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_path' is unsupported; use '{stage_key}.lora_sha'")
            sha_raw = raw.get("model_sha")
            if not isinstance(sha_raw, str) or not sha_raw.strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.model_sha' is required (sha256)")
            sha = sha_raw.strip().lower()
            model_path = resolve_asset_by_sha(sha)
            if not model_path:
                raise HTTPException(status_code=409, detail=f"WAN stage model not found for sha: {sha}")
            if not str(model_path).lower().endswith(".gguf"):
                raise HTTPException(status_code=409, detail=f"WAN stage sha does not resolve to a .gguf file: {sha}")
            out: dict[str, object] = dict(raw)
            out.pop("model_sha", None)
            out["model_dir"] = model_path
            if out.get("lora_weight") not in (None, "") and not (isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip()):
                raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_weight' requires '{stage_key}.lora_sha'")
            if isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip():
                lora_sha = str(out.get("lora_sha")).strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_sha' must be sha256 (64 lowercase hex)")
                lora_path = resolve_asset_by_sha(lora_sha)
                if not lora_path:
                    raise HTTPException(status_code=409, detail=f"WAN stage LoRA not found for sha: {lora_sha}")
                if not str(lora_path).lower().endswith(".safetensors"):
                    raise HTTPException(status_code=409, detail=f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}")
                out["lora_sha"] = lora_sha
            return out

        extras["wan_high"] = _resolve_wan_stage("wan_high")
        extras["wan_low"] = _resolve_wan_stage("wan_low")

        # Resolve sha-selected WAN assets
        if payload.get("wan_vae_path") or payload.get("wan_text_encoder_path") or payload.get("wan_text_encoder_dir"):
            raise HTTPException(status_code=400, detail="WAN sha-only mode: do not send wan_*_path fields; send wan_vae_sha/wan_tenc_sha instead.")

        wan_vae_sha = _require_sha_field("wan_vae_sha")
        wan_tenc_sha = _require_sha_field("wan_tenc_sha")

        wan_vae_path = resolve_asset_by_sha(wan_vae_sha)
        if not wan_vae_path:
            raise HTTPException(status_code=409, detail=f"WAN VAE not found for sha: {wan_vae_sha}")
        extras["wan_vae_path"] = wan_vae_path

        wan_tenc_path = resolve_asset_by_sha(wan_tenc_sha)
        if not wan_tenc_path:
            raise HTTPException(status_code=409, detail=f"WAN text encoder not found for sha: {wan_tenc_sha}")
        te_lower = str(wan_tenc_path).lower()
        if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
            raise HTTPException(
                status_code=409,
                detail=f"WAN text encoder sha must resolve to a .safetensors or .gguf file: {wan_tenc_sha}",
            )
        extras["wan_text_encoder_path"] = wan_tenc_path

        extras["wan_metadata_dir"] = _resolve_wan_metadata_dir(payload)

        # Pass-through of runtime controls (non-model-part config)
        for key in (
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            'gguf_te_device',
            'gguf_te_impl',
            'gguf_te_kernel_required',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)

        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags(payload)
        req = Txt2VidRequest(
            task=TaskType.TXT2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            video_options=video_options,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            metadata={
                "styles": payload.get('txt2vid_styles', []),
            },
        )

        engine_key = "wan22_5b"
        model_ref = str(extras["wan_high"]["model_dir"])  # type: ignore[index]
        return req, engine_key, model_ref

    def prepare_img2vid(payload: Dict[str, Any]) -> Tuple[Img2VidRequest, str, Optional[str]]:
        logging.getLogger('backend.api').info('[api] DEBUG: enter prepare_img2vid')
        prompt = payload.get('img2vid_prompt', '')
        negative_prompt = payload.get('img2vid_neg_prompt', '')
        width_val = int(payload.get('img2vid_width', 768))
        height_val = int(payload.get('img2vid_height', 432))
        _wan_require_dims_multiple_of_16(task="img2vid", width=width_val, height=height_val)
        steps_val = int(payload.get('img2vid_steps', 30))
        fps_val = int(payload.get('img2vid_fps', 24))
        frames_val = int(payload.get('img2vid_num_frames', 16))
        sampler_name = str(payload.get('img2vid_sampler', payload.get('img2vid_sampling', 'uni-pc')))
        scheduler_name = str(payload.get('img2vid_scheduler', 'simple'))
        try:
            from apps.backend.types.samplers import SamplerKind
            from apps.backend.runtime.sampling.context import SchedulerName

            SamplerKind.from_string(sampler_name)
            SchedulerName.from_string(scheduler_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        seed_val = int(payload.get('img2vid_seed', -1))
        cfg_val = float(payload.get('img2vid_cfg_scale', 7.0))

        init_image_data = payload.get('img2vid_init_image')
        init_image = media.decode_image(init_image_data) if init_image_data else None

        extras: Dict[str, Any] = {}
        if "video_return_frames" in payload:
            raw_return_frames = payload.get("video_return_frames")
            if raw_return_frames is not None and not isinstance(raw_return_frames, bool):
                raise HTTPException(status_code=400, detail="'video_return_frames' must be a boolean when provided")
            if isinstance(raw_return_frames, bool):
                extras["video_return_frames"] = bool(raw_return_frames)
        video_options = None
        try:
            from apps.backend.core.params.video import VideoExportOptions

            video_options = VideoExportOptions(
                filename_prefix=(str(payload.get("video_filename_prefix")).strip() if payload.get("video_filename_prefix") else None),
                format=(str(payload.get("video_format")).strip() if payload.get("video_format") else None),
                pix_fmt=(str(payload.get("video_pix_fmt")).strip() if payload.get("video_pix_fmt") else None),
                crf=(int(payload.get("video_crf")) if payload.get("video_crf") is not None else None),
                loop_count=(int(payload.get("video_loop_count")) if payload.get("video_loop_count") is not None else None),
                pingpong=(bool(payload.get("video_pingpong")) if payload.get("video_pingpong") is not None else None),
                save_metadata=(bool(payload.get("video_save_metadata")) if payload.get("video_save_metadata") is not None else None),
                save_output=(bool(payload.get("video_save_output")) if payload.get("video_save_output") is not None else None),
                trim_to_audio=(bool(payload.get("video_trim_to_audio")) if payload.get("video_trim_to_audio") is not None else None),
            ).as_dict()
        except Exception:
            video_options = None
        if video_options:
            extras["video"] = {
                "video_filename_prefix": payload.get("video_filename_prefix"),
                "video_format": payload.get("video_format"),
                "video_pix_fmt": payload.get("video_pix_fmt"),
                "video_crf": payload.get("video_crf"),
                "video_loop_count": payload.get("video_loop_count"),
                "video_pingpong": payload.get("video_pingpong"),
                "video_save_metadata": payload.get("video_save_metadata"),
                "video_save_output": payload.get("video_save_output"),
                "video_trim_to_audio": payload.get("video_trim_to_audio"),
            }
        if isinstance(payload.get('video_interpolation'), dict):
            extras['video_interpolation'] = payload.get('video_interpolation')
        # WAN (GGUF-only): strict sha-only selection for model parts (no raw paths).
        from apps.backend.inventory.cache import resolve_asset_by_sha

        def _require_sha_field(key: str) -> str:
            val = payload.get(key)
            if isinstance(val, dict):
                raise HTTPException(status_code=400, detail=f"'{key}' must be a string sha256, got object")
            if not isinstance(val, str) or not val.strip():
                raise HTTPException(status_code=400, detail=f"'{key}' is required and must be a non-empty sha256 string")
            return val.strip().lower()

        def _resolve_wan_stage(stage_key: str) -> dict[str, object]:
            raw = payload.get(stage_key)
            if not isinstance(raw, dict):
                raise HTTPException(status_code=400, detail=f"'{stage_key}' is required and must be an object")
            if isinstance(raw.get("model_dir"), str) and str(raw.get("model_dir")).strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.model_dir' is unsupported; use '{stage_key}.model_sha'")
            if isinstance(raw.get("lora_path"), str) and str(raw.get("lora_path")).strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_path' is unsupported; use '{stage_key}.lora_sha'")
            sha_raw = raw.get("model_sha")
            if not isinstance(sha_raw, str) or not sha_raw.strip():
                raise HTTPException(status_code=400, detail=f"'{stage_key}.model_sha' is required (sha256)")
            sha = sha_raw.strip().lower()
            model_path = resolve_asset_by_sha(sha)
            if not model_path:
                raise HTTPException(status_code=409, detail=f"WAN stage model not found for sha: {sha}")
            if not str(model_path).lower().endswith(".gguf"):
                raise HTTPException(status_code=409, detail=f"WAN stage sha does not resolve to a .gguf file: {sha}")
            out: dict[str, object] = dict(raw)
            out.pop("model_sha", None)
            out["model_dir"] = model_path
            if out.get("lora_weight") not in (None, "") and not (isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip()):
                raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_weight' requires '{stage_key}.lora_sha'")
            if isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip():
                lora_sha = str(out.get("lora_sha")).strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_sha' must be sha256 (64 lowercase hex)")
                lora_path = resolve_asset_by_sha(lora_sha)
                if not lora_path:
                    raise HTTPException(status_code=409, detail=f"WAN stage LoRA not found for sha: {lora_sha}")
                if not str(lora_path).lower().endswith(".safetensors"):
                    raise HTTPException(status_code=409, detail=f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}")
                out["lora_sha"] = lora_sha
            return out

        extras["wan_high"] = _resolve_wan_stage("wan_high")
        extras["wan_low"] = _resolve_wan_stage("wan_low")

        # Resolve sha-selected WAN assets
        if payload.get("wan_vae_path") or payload.get("wan_text_encoder_path") or payload.get("wan_text_encoder_dir"):
            raise HTTPException(status_code=400, detail="WAN sha-only mode: do not send wan_*_path fields; send wan_vae_sha/wan_tenc_sha instead.")

        wan_vae_sha = _require_sha_field("wan_vae_sha")
        wan_tenc_sha = _require_sha_field("wan_tenc_sha")

        wan_vae_path = resolve_asset_by_sha(wan_vae_sha)
        if not wan_vae_path:
            raise HTTPException(status_code=409, detail=f"WAN VAE not found for sha: {wan_vae_sha}")
        extras["wan_vae_path"] = wan_vae_path

        wan_tenc_path = resolve_asset_by_sha(wan_tenc_sha)
        if not wan_tenc_path:
            raise HTTPException(status_code=409, detail=f"WAN text encoder not found for sha: {wan_tenc_sha}")
        te_lower = str(wan_tenc_path).lower()
        if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
            raise HTTPException(
                status_code=409,
                detail=f"WAN text encoder sha must resolve to a .safetensors or .gguf file: {wan_tenc_sha}",
            )
        extras["wan_text_encoder_path"] = wan_tenc_path

        extras["wan_metadata_dir"] = _resolve_wan_metadata_dir(payload)

        # Pass-through of runtime controls (non-model-part config)
        for key in (
            'gguf_offload',
            'gguf_offload_level',
            'gguf_sdpa_policy',
            'gguf_attn_chunk',
            'gguf_cache_policy',
            'gguf_cache_limit_mb',
            'gguf_log_mem_interval',
            'gguf_te_device',
            'gguf_te_impl',
            'gguf_te_kernel_required',
        ):
            if key in payload and payload.get(key) is not None:
                extras[key] = payload.get(key)

        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags(payload)
        req = Img2VidRequest(
            task=TaskType.IMG2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            sampler=sampler_name,
            scheduler=scheduler_name,
            seed=seed_val,
            guidance_scale=cfg_val,
            video_options=video_options,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
            metadata={
                "styles": payload.get('img2vid_styles', []),
            },
        )

        engine_key = "wan22_5b"
        model_ref = str(extras["wan_high"]["model_dir"])  # type: ignore[index]
        logging.getLogger('backend.api').info('[api] DEBUG: exit prepare_img2vid engine=%s model_ref=%s size=%dx%d frames=%d', engine_key, model_ref, width_val, height_val, frames_val)
        return req, engine_key, model_ref

    def _resolve_vid2vid_input_path(raw: str, *, field: str) -> str:
        """Resolve a user-supplied input path safely (root-scoped).

        Policy: by default, only paths under the backend working directory are allowed.
        Use upload (multipart) to avoid path permission issues.
        """
        v = str(raw or "").strip()
        if not v:
            raise RuntimeError(f"vid2vid {field} path is empty")
        p = Path(os.path.expanduser(v))
        if not p.is_absolute():
            p = CODEX_ROOT / p
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        root = CODEX_ROOT.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise RuntimeError(
                f"vid2vid {field} must be under the repo root ({root}); "
                "use upload instead for external files."
            ) from None
        if not resolved.is_file():
            raise RuntimeError(f"vid2vid {field} not found: {resolved}")
        return str(resolved)

    def _resolve_vid2vid_input_dir(raw: str, *, field: str) -> str:
        v = str(raw or "").strip()
        if not v:
            raise RuntimeError(f"vid2vid {field} path is empty")
        p = Path(os.path.expanduser(v))
        if not p.is_absolute():
            p = CODEX_ROOT / p
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        root = CODEX_ROOT.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise RuntimeError(
                f"vid2vid {field} must be under the repo root ({root}); "
                "use upload instead for external files."
            ) from None
        if not resolved.is_dir():
            raise RuntimeError(f"vid2vid {field} not found: {resolved}")
        return str(resolved)

    def _normalize_wan_stage_payload_strict(stage: object, *, field: str) -> object:
        if not isinstance(stage, dict):
            return stage
        out: dict[str, object] = dict(stage)
        if isinstance(out.get("model_dir"), str) and str(out.get("model_dir")).strip():
            # model_dir may refer to a GGUF file or a diffusers directory; enforce repo-root scoping either way.
            raw_model_dir = str(out.get("model_dir") or "")
            p = Path(_path_from_api(raw_model_dir)).expanduser()
            try:
                resolved = p.resolve()
            except Exception:
                resolved = p
            root = CODEX_ROOT.resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{field}.model_dir' must be under the repo root ({root}); "
                        "use sha-only mode for WAN GGUF or upload for external files."
                    ),
                ) from None
            if not (resolved.is_file() or resolved.is_dir()):
                raise HTTPException(status_code=400, detail=f"'{field}.model_dir' not found: {resolved}")
            out["model_dir"] = str(resolved)
        if isinstance(out.get("lora_path"), str) and str(out.get("lora_path")).strip():
            raise HTTPException(status_code=400, detail=f"'{field}.lora_path' is unsupported; use '{field}.lora_sha'")

        if out.get("lora_weight") not in (None, "") and not (isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip()):
            raise HTTPException(status_code=400, detail=f"'{field}.lora_weight' requires '{field}.lora_sha'")
        if isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip():
            from apps.backend.inventory.cache import resolve_asset_by_sha

            lora_sha = str(out.get("lora_sha")).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                raise HTTPException(status_code=400, detail=f"'{field}.lora_sha' must be sha256 (64 lowercase hex)")
            lora_path = resolve_asset_by_sha(lora_sha)
            if not lora_path:
                raise HTTPException(status_code=409, detail=f"WAN stage LoRA not found for sha: {lora_sha}")
            if not str(lora_path).lower().endswith(".safetensors"):
                raise HTTPException(status_code=409, detail=f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}")
            out["lora_sha"] = lora_sha
        return out

    def prepare_vid2vid(payload: Dict[str, Any]) -> Tuple[Vid2VidRequest, str, Optional[str]]:
        prompt = payload.get("vid2vid_prompt", "")
        negative_prompt = payload.get("vid2vid_neg_prompt", "")
        width_val = int(payload.get("vid2vid_width", 768))
        height_val = int(payload.get("vid2vid_height", 432))
        _wan_require_dims_multiple_of_16(task="vid2vid", width=width_val, height=height_val)
        steps_val = int(payload.get("vid2vid_steps", 30))
        fps_val = int(payload.get("vid2vid_fps", 24))
        frames_val = int(payload.get("vid2vid_num_frames", 16))
        sampler_name = str(payload.get("vid2vid_sampler", ""))
        scheduler_name = str(payload.get("vid2vid_scheduler", ""))
        if sampler_name.strip() or scheduler_name.strip():
            try:
                from apps.backend.types.samplers import SamplerKind
                from apps.backend.runtime.sampling.context import SchedulerName

                if sampler_name.strip():
                    SamplerKind.from_string(sampler_name.strip())
                if scheduler_name.strip():
                    SchedulerName.from_string(scheduler_name.strip())
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        seed_val = int(payload.get("vid2vid_seed", -1))
        cfg_val = float(payload.get("vid2vid_cfg_scale", 7.0))
        strength_val = payload.get("vid2vid_strength")
        strength = float(strength_val) if strength_val is not None else None

        method_raw = str(payload.get("vid2vid_method") or "flow_chunks")
        method = method_raw.strip().lower()

        # Driving/original video is required for classic vid2vid methods; optional for wan_animate (used for audio copy).
        video_path = ""
        video_path_raw = payload.get("vid2vid_video_path") or payload.get("vid2vid_video")
        if video_path_raw:
            video_path = _resolve_vid2vid_input_path(video_path_raw, field="video_path")
        elif method != "wan_animate":
            raise RuntimeError("vid2vid_video_path is required (use upload)")

        # WAN-Animate inputs (preprocessed pose/face, optional bg/mask, plus reference image).
        ref_image = None
        pose_path = ""
        face_path = ""
        bg_path = ""
        mask_path = ""
        animate_mode = str(payload.get("vid2vid_animate_mode") or "animate")
        seg_val = int(payload.get("vid2vid_segment_frame_length", 77))
        prev_val = int(payload.get("vid2vid_prev_segment_conditioning_frames", 1))
        motion_bs_val = payload.get("vid2vid_motion_encode_batch_size")
        motion_bs = int(motion_bs_val) if motion_bs_val is not None else None

        if method == "wan_animate":
            ref_b64 = payload.get("vid2vid_reference_image")
            if ref_b64:
                ref_image = media.decode_image(ref_b64)
            else:
                ref_path = payload.get("vid2vid_reference_image_path")
                if ref_path:
                    from PIL import Image  # type: ignore

                    rp = _resolve_vid2vid_input_path(ref_path, field="reference_image")
                    img = Image.open(Path(rp))
                    ref_image = img.copy()
                    img.close()
            if ref_image is None:
                raise RuntimeError("vid2vid wan_animate requires a reference image (upload or vid2vid_reference_image)")

            pose_raw = payload.get("vid2vid_pose_video_path") or payload.get("vid2vid_pose_video")
            face_raw = payload.get("vid2vid_face_video_path") or payload.get("vid2vid_face_video")
            pose_path = _resolve_vid2vid_input_path(pose_raw, field="pose_video")
            face_path = _resolve_vid2vid_input_path(face_raw, field="face_video")

            mode_lc = animate_mode.strip().lower()
            if mode_lc in {"replace", "replacement"}:
                bg_raw = payload.get("vid2vid_background_video_path") or payload.get("vid2vid_background_video")
                mask_raw = payload.get("vid2vid_mask_video_path") or payload.get("vid2vid_mask_video")
                bg_path = _resolve_vid2vid_input_path(bg_raw, field="background_video")
                mask_path = _resolve_vid2vid_input_path(mask_raw, field="mask_video")

        extras: Dict[str, Any] = {}
        if "video_return_frames" in payload:
            raw_return_frames = payload.get("video_return_frames")
            if raw_return_frames is not None and not isinstance(raw_return_frames, bool):
                raise HTTPException(status_code=400, detail="'video_return_frames' must be a boolean when provided")
            if isinstance(raw_return_frames, bool):
                extras["video_return_frames"] = bool(raw_return_frames)
        video_options = None
        try:
            from apps.backend.core.params.video import VideoExportOptions

            video_options = VideoExportOptions(
                filename_prefix=(str(payload.get("video_filename_prefix")).strip() if payload.get("video_filename_prefix") else None),
                format=(str(payload.get("video_format")).strip() if payload.get("video_format") else None),
                pix_fmt=(str(payload.get("video_pix_fmt")).strip() if payload.get("video_pix_fmt") else None),
                crf=(int(payload.get("video_crf")) if payload.get("video_crf") is not None else None),
                loop_count=(int(payload.get("video_loop_count")) if payload.get("video_loop_count") is not None else None),
                pingpong=(bool(payload.get("video_pingpong")) if payload.get("video_pingpong") is not None else None),
                save_metadata=(bool(payload.get("video_save_metadata")) if payload.get("video_save_metadata") is not None else None),
                save_output=(bool(payload.get("video_save_output")) if payload.get("video_save_output") is not None else None),
                trim_to_audio=(bool(payload.get("video_trim_to_audio")) if payload.get("video_trim_to_audio") is not None else None),
            ).as_dict()
        except Exception:
            video_options = None
        if video_options:
            extras["video"] = {
                "video_filename_prefix": payload.get("video_filename_prefix"),
                "video_format": payload.get("video_format"),
                "video_pix_fmt": payload.get("video_pix_fmt"),
                "video_crf": payload.get("video_crf"),
                "video_loop_count": payload.get("video_loop_count"),
                "video_pingpong": payload.get("video_pingpong"),
                "video_save_metadata": payload.get("video_save_metadata"),
                "video_save_output": payload.get("video_save_output"),
                "video_trim_to_audio": payload.get("video_trim_to_audio"),
            }
        if isinstance(payload.get("video_interpolation"), dict):
            extras["video_interpolation"] = payload.get("video_interpolation")
        if method != "wan_animate":
            # WAN (GGUF-only): strict sha-only selection for model parts (no raw paths).
            from apps.backend.inventory.cache import resolve_asset_by_sha

            def _require_sha_field(key: str) -> str:
                val = payload.get(key)
                if isinstance(val, dict):
                    raise HTTPException(status_code=400, detail=f"'{key}' must be a string sha256, got object")
                if not isinstance(val, str) or not val.strip():
                    raise HTTPException(status_code=400, detail=f"'{key}' is required and must be a non-empty sha256 string")
                return val.strip().lower()

            def _resolve_wan_stage(stage_key: str) -> dict[str, object]:
                raw = payload.get(stage_key)
                if not isinstance(raw, dict):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}' is required and must be an object")
                if isinstance(raw.get("model_dir"), str) and str(raw.get("model_dir")).strip():
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.model_dir' is unsupported; use '{stage_key}.model_sha'")
                if isinstance(raw.get("lora_path"), str) and str(raw.get("lora_path")).strip():
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_path' is unsupported; use '{stage_key}.lora_sha'")
                sha_raw = raw.get("model_sha")
                if not isinstance(sha_raw, str) or not sha_raw.strip():
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.model_sha' is required (sha256)")
                sha = sha_raw.strip().lower()
                model_path = resolve_asset_by_sha(sha)
                if not model_path:
                    raise HTTPException(status_code=409, detail=f"WAN stage model not found for sha: {sha}")
                if not str(model_path).lower().endswith(".gguf"):
                    raise HTTPException(status_code=409, detail=f"WAN stage sha does not resolve to a .gguf file: {sha}")
                out: dict[str, object] = dict(raw)
                out.pop("model_sha", None)
                out["model_dir"] = model_path
                if out.get("lora_weight") not in (None, "") and not (isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip()):
                    raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_weight' requires '{stage_key}.lora_sha'")
                if isinstance(out.get("lora_sha"), str) and str(out.get("lora_sha")).strip():
                    lora_sha = str(out.get("lora_sha")).strip().lower()
                    if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                        raise HTTPException(status_code=400, detail=f"'{stage_key}.lora_sha' must be sha256 (64 lowercase hex)")
                    lora_path = resolve_asset_by_sha(lora_sha)
                    if not lora_path:
                        raise HTTPException(status_code=409, detail=f"WAN stage LoRA not found for sha: {lora_sha}")
                    if not str(lora_path).lower().endswith(".safetensors"):
                        raise HTTPException(status_code=409, detail=f"WAN stage LoRA sha must resolve to a .safetensors file: {lora_sha}")
                    out["lora_sha"] = lora_sha
                return out

            extras["wan_high"] = _resolve_wan_stage("wan_high")
            extras["wan_low"] = _resolve_wan_stage("wan_low")

            if payload.get("wan_vae_path") or payload.get("wan_text_encoder_path") or payload.get("wan_text_encoder_dir"):
                raise HTTPException(status_code=400, detail="WAN sha-only mode: do not send wan_*_path fields; send wan_vae_sha/wan_tenc_sha instead.")

            wan_vae_sha = _require_sha_field("wan_vae_sha")
            wan_tenc_sha = _require_sha_field("wan_tenc_sha")

            wan_vae_path = resolve_asset_by_sha(wan_vae_sha)
            if not wan_vae_path:
                raise HTTPException(status_code=409, detail=f"WAN VAE not found for sha: {wan_vae_sha}")
            extras["wan_vae_path"] = wan_vae_path

            wan_tenc_path = resolve_asset_by_sha(wan_tenc_sha)
            if not wan_tenc_path:
                raise HTTPException(status_code=409, detail=f"WAN text encoder not found for sha: {wan_tenc_sha}")
            te_lower = str(wan_tenc_path).lower()
            if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
                raise HTTPException(
                    status_code=409,
                    detail=f"WAN text encoder sha must resolve to a .safetensors or .gguf file: {wan_tenc_sha}",
                )
            extras["wan_text_encoder_path"] = wan_tenc_path

            extras["wan_metadata_dir"] = _resolve_wan_metadata_dir(payload)

            for key in (
                "gguf_offload",
                "gguf_offload_level",
                "gguf_sdpa_policy",
                "gguf_attn_chunk",
                "gguf_cache_policy",
                "gguf_cache_limit_mb",
                "gguf_log_mem_interval",
                "gguf_te_device",
                "gguf_te_impl",
                "gguf_te_kernel_required",
            ):
                if key in payload and payload.get(key) is not None:
                    extras[key] = payload.get(key)
        else:
            # wan_animate: keep repo-scoped model_dir normalization (Diffusers dir inputs); stage LoRA is sha-only (`lora_sha`).
            if isinstance(payload.get("wan_high"), dict):
                extras["wan_high"] = _normalize_wan_stage_payload_strict(payload.get("wan_high"), field="wan_high")
            if isinstance(payload.get("wan_low"), dict):
                extras["wan_low"] = _normalize_wan_stage_payload_strict(payload.get("wan_low"), field="wan_low")
            for key in ("wan_metadata_dir", "wan_tokenizer_dir"):
                if key in payload and payload.get(key) is not None:
                    extras[key] = _resolve_vid2vid_input_dir(str(payload.get(key)), field=key)

        extras["vid2vid"] = {
            "method": method_raw,
            "use_source_fps": bool(payload.get("vid2vid_use_source_fps", True)),
            "use_source_frames": bool(payload.get("vid2vid_use_source_frames", True)),
            "start_seconds": payload.get("vid2vid_start_seconds"),
            "end_seconds": payload.get("vid2vid_end_seconds"),
            "max_frames": payload.get("vid2vid_max_frames"),
            "chunk_frames": payload.get("vid2vid_chunk_frames"),
            "overlap_frames": payload.get("vid2vid_overlap_frames"),
            "preview_frames": payload.get("vid2vid_preview_frames"),
        }
        extras["vid2vid_flow"] = {
            "enabled": bool(payload.get("vid2vid_flow_enabled", True)),
            "use_large": bool(payload.get("vid2vid_flow_use_large", False)),
            "downscale": payload.get("vid2vid_flow_downscale", 2),
            "device": payload.get("vid2vid_flow_device", None),
        }

        smart_offload, smart_fallback, smart_cache = _resolve_smart_flags(payload)
        req = Vid2VidRequest(
            task=TaskType.VID2VID,
            prompt=prompt,
            negative_prompt=negative_prompt,
            sampler=(sampler_name.strip() or None),
            scheduler=(scheduler_name.strip() or None),
            video_path=video_path,
            reference_image=ref_image,
            pose_video_path=pose_path,
            face_video_path=face_path,
            background_video_path=bg_path,
            mask_video_path=mask_path,
            animate_mode=animate_mode,
            segment_frame_length=seg_val,
            prev_segment_conditioning_frames=prev_val,
            motion_encode_batch_size=motion_bs,
            width=width_val,
            height=height_val,
            steps=steps_val,
            fps=fps_val,
            num_frames=frames_val,
            seed=seed_val,
            guidance_scale=cfg_val,
            strength=strength,
            video_options=video_options,
            extras=extras,
            smart_offload=smart_offload,
            smart_fallback=smart_fallback,
            smart_cache=smart_cache,
        )

        if method == "wan_animate":
            engine_key = "wan22_animate_14b"
            model_dir = payload.get("vid2vid_model_dir")
            if not isinstance(model_dir, str) or not model_dir.strip():
                raise RuntimeError("vid2vid wan_animate requires 'vid2vid_model_dir' (repo-scoped path)")
            model_ref = _resolve_vid2vid_input_dir(model_dir.strip(), field="model_dir")
            return req, engine_key, model_ref

        engine_key = "wan22_5b"
        model_ref = str(extras["wan_high"]["model_dir"])  # type: ignore[index]
        return req, engine_key, model_ref

    def run_video_task(task_id: str, payload: Dict[str, Any], entry: TaskEntry, task_type: TaskType, *, device: str) -> None:
        def push(event: Dict[str, Any]) -> None:
            entry.push_event(event)

        push({"type": "status", "stage": "queued"})
        try:
            _ensure_default_engines_registered()
            if task_type == TaskType.TXT2VID:
                req, engine_key, model_ref = prepare_txt2vid(payload)
            elif task_type == TaskType.IMG2VID:
                req, engine_key, model_ref = prepare_img2vid(payload)
            elif task_type == TaskType.VID2VID:
                req, engine_key, model_ref = prepare_vid2vid(payload)
            else:
                raise RuntimeError(f"Unsupported video task: {task_type}")
        except Exception as err:
            entry.error = str(err)
            entry.mark_finished(success=False)
            unregister_task(task_id)
            raise

        def worker() -> None:
            acquired = False
            success = False
            try:
                if single_flight_enabled():
                    push({"type": "status", "stage": "waiting_for_inference"})

                acquired = acquire_inference_gate(
                    should_cancel=lambda: bool(entry.cancel_requested and entry.cancel_mode == "immediate"),
                )
                if not acquired:
                    entry.error = "cancelled"
                    return

                push({"type": "status", "stage": "running"})
                from apps.backend.interfaces.api.device_selection import apply_primary_device

                apply_primary_device(device)

                engine_opts = {"export_video": bool(_opts_snapshot().codex_export_video)}
                from apps.backend.interfaces.api.tasks.generation_tasks import encode_images as _encode_images
                from apps.backend.runtime.memory.smart_offload import smart_runtime_overrides

                with smart_runtime_overrides(
                    smart_offload=bool(getattr(req, "smart_offload", False)),
                    smart_fallback=bool(getattr(req, "smart_fallback", False)),
                    smart_cache=bool(getattr(req, "smart_cache", False)),
                ):
                    for ev in _ORCH.run(task_type, engine_key, req, model_ref=model_ref, engine_options=engine_opts):
                        if entry.cancel_requested and entry.cancel_mode == "immediate":
                            entry.error = "cancelled"
                            return
                        if isinstance(ev, ProgressEvent):
                            push(
                                {
                                    "type": "progress",
                                    "stage": ev.stage,
                                    "percent": ev.percent,
                                    "step": ev.step,
                                    "total_steps": ev.total_steps,
                                    "eta_seconds": ev.eta_seconds,
                                }
                            )
                        elif isinstance(ev, ResultEvent):
                            payload_obj = ev.payload or {}
                            info_raw = payload_obj.get("info", "{}")
                            try:
                                info_obj = json.loads(info_raw)
                            except Exception:
                                info_obj = info_raw
                            encoded = _encode_images(payload_obj.get("images", []))
                            result = {"images": encoded, "info": info_obj}
                            if isinstance(payload_obj.get("video"), dict):
                                result["video"] = payload_obj.get("video")
                            entry.result = {"status": "completed", "result": result}
                success = True
            except Exception as err:
                try:
                    from apps.backend.runtime.diagnostics.exception_hook import dump_exception as _dump_exc
                    _dump_exc(type(err), err, err.__traceback__, where=f'{label}_worker', context={'task_id': task_id})
                except Exception:
                    pass
                entry.error = str(err)
                success = False
            finally:
                entry.mark_finished(success=success)
                entry.schedule_cleanup(task_id)
                if acquired:
                    try:
                        release_inference_gate()
                    except Exception:
                        pass
                if task_type == TaskType.VID2VID:
                    try:
                        uploaded_paths: list[str] = []
                        if payload.get("__vid2vid_uploaded_paths"):
                            if isinstance(payload.get("__vid2vid_uploaded_paths"), list):
                                uploaded_paths = [str(x) for x in payload.get("__vid2vid_uploaded_paths") or []]
                        elif payload.get("__vid2vid_uploaded_path"):
                            uploaded_paths = [str(payload.get("__vid2vid_uploaded_path"))]

                        if uploaded_paths:
                            up_root = (CODEX_ROOT / ".tmp" / "uploads" / "vid2vid").resolve()
                            for item in uploaded_paths:
                                up_path = Path(str(item))
                                try:
                                    resolved = up_path.resolve()
                                except Exception:
                                    resolved = up_path
                                try:
                                    resolved.relative_to(up_root)
                                except ValueError:
                                    continue
                                try:
                                    resolved.unlink()
                                except Exception:
                                    pass
                    except Exception:
                        pass

        label = "txt2vid" if task_type == TaskType.TXT2VID else ("img2vid" if task_type == TaskType.IMG2VID else "vid2vid")
        thread = threading.Thread(target=worker, name=f"{label}-task-{task_id}", daemon=True)
        thread.start()

    @router.post('/api/txt2img')
    async def txt2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")

        device = _parse_explicit_device(payload)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-{uuid4().hex})"
        register_task(task_id, entry)
        run_txt2img_task(task_id, payload, entry, device=device)
        return {"task_id": task_id}

    @router.post('/api/img2img')
    async def img2img(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")

        device = _parse_explicit_device(payload)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2img-{uuid4().hex})"
        register_task(task_id, entry)
        run_img2img_task(task_id, payload, entry, device=device)
        return {"task_id": task_id}

    @router.post('/api/txt2vid')
    async def txt2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")

        device = _parse_explicit_device(payload)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-txt2vid-{uuid4().hex})"
        register_task(task_id, entry)
        run_video_task(task_id, payload, entry, TaskType.TXT2VID, device=device)
        return {"task_id": task_id}

    @router.post('/api/img2vid')
    async def img2vid(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        logging.getLogger('backend.api').info('[api] DEBUG: POST /api/img2vid received')
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be JSON object")

        device = _parse_explicit_device(payload)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-img2vid-{uuid4().hex})"
        register_task(task_id, entry)
        logging.getLogger('backend.api').info('[api] DEBUG: scheduling img2vid task_id=%s', task_id)
        run_video_task(task_id, payload, entry, TaskType.IMG2VID, device=device)
        return {"task_id": task_id}

    @router.post('/api/vid2vid')
    async def vid2vid(
        video: UploadFile | None = File(default=None),
        reference_image: UploadFile | None = File(default=None),
        pose_video: UploadFile | None = File(default=None),
        face_video: UploadFile | None = File(default=None),
        background_video: UploadFile | None = File(default=None),
        mask_video: UploadFile | None = File(default=None),
        payload: str = Form(default="{}"),
    ) -> Dict[str, Any]:
        """Video-to-video endpoint.

        Accepts multipart form-data:
          - video: driving/original video (required for flow_chunks/native; optional for wan_animate)
          - reference_image: character image (wan_animate only)
          - pose_video / face_video: preprocessed videos (wan_animate only)
          - background_video / mask_video: replacement mode only (wan_animate)
          - payload: JSON string with vid2vid_* keys (and WAN extras)

        For security, path-based inputs are restricted to the backend working directory.
        """
        try:
            data = json.loads(payload) if payload else {}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"payload must be JSON: {exc}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="payload must be JSON object")

        device = _parse_explicit_device(data)
        loop = asyncio.get_running_loop()
        entry = TaskEntry(loop)
        task_id = f"task(api-vid2vid-{uuid4().hex})"
        register_task(task_id, entry)

        uploaded_paths: list[str] = []
        try:
            import shutil as _shutil

            up_dir = CODEX_ROOT / ".tmp" / "uploads" / "vid2vid"
            up_dir.mkdir(parents=True, exist_ok=True)

            def _save(upload: UploadFile, *, default_suffix: str) -> str:
                suffix = default_suffix
                try:
                    name = str(upload.filename or "")
                    if "." in name:
                        suffix = "." + name.rsplit(".", 1)[1].lower()
                except Exception:
                    suffix = default_suffix
                dst = up_dir / f"{uuid4().hex}{suffix}"
                with dst.open("wb") as f:
                    _shutil.copyfileobj(upload.file, f)
                uploaded_paths.append(str(dst))
                return str(dst)

            if video is not None:
                data["vid2vid_video_path"] = _save(video, default_suffix=".mp4")
            if reference_image is not None:
                data["vid2vid_reference_image_path"] = _save(reference_image, default_suffix=".png")
            if pose_video is not None:
                data["vid2vid_pose_video_path"] = _save(pose_video, default_suffix=".mp4")
            if face_video is not None:
                data["vid2vid_face_video_path"] = _save(face_video, default_suffix=".mp4")
            if background_video is not None:
                data["vid2vid_background_video_path"] = _save(background_video, default_suffix=".mp4")
            if mask_video is not None:
                data["vid2vid_mask_video_path"] = _save(mask_video, default_suffix=".mp4")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to save uploaded files: {exc}")

        if uploaded_paths:
            data["__vid2vid_uploaded_paths"] = uploaded_paths

        run_video_task(task_id, data, entry, TaskType.VID2VID, device=device)
        return {"task_id": task_id}

    return router
