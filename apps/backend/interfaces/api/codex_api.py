from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

import base64
import logging
import secrets

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, field_validator

from apps.backend.codex import options as codex_options
from apps.backend.core.engine_interface import TaskType
from apps.backend.core.requests import Img2ImgRequest, Txt2ImgRequest
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models import api as model_api
from apps.backend.services.image_service import ImageService
from apps.backend.services.media_service import MediaService
from apps.backend.services.options_service import OptionsService
from apps.backend.services.progress_service import ProgressService
from apps.backend.services.sampler_service import SamplerService
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.sampling.catalog import SAMPLER_OPTIONS, SCHEDULER_OPTIONS


logger = logging.getLogger("backend.api.codex")
router = APIRouter(prefix="/codex/api/v1", tags=["codex-api"])

_media_service = MediaService()
_options_service = OptionsService()
_sampler_service = SamplerService()


def _random_seed() -> int:
    return secrets.randbelow(2**32)


def _raise_not_implemented(feature: str, backlog: str) -> None:
    raise HTTPException(
        status_code=501,
        detail=f"'{feature}' is not yet available in the Codex API (tracked under {backlog}).",
    )


class MemoryUnloadRequest(BaseModel):
    """Request payload for memory unload operations.

    - scope='all'  → unload all models tracked by the memory manager.
    - scope='ids'  → unload only models whose Python id() is listed in `ids`.
    """

    scope: Literal["all", "ids"] = "all"
    ids: Optional[List[int]] = None


class HighresOptions(BaseModel):
    enable: bool = Field(default=False, description="Enable highres second pass")
    denoise: float = Field(default=0.0, ge=0.0, le=1.0)
    scale: float = Field(default=1.0, ge=1.0)
    upscaler: str = Field(default="Use same upscaler")
    steps: int = Field(default=0, ge=0)
    resize_x: int = Field(default=0, ge=0)
    resize_y: int = Field(default=0, ge=0)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    cfg: Optional[float] = Field(default=None, ge=0.0)
    distilled_cfg: Optional[float] = Field(default=None, ge=0.0)
    additional_modules: Optional[List[str]] = None
    checkpoint: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None

    def as_payload(self) -> Dict[str, Any]:
        if not self.enable:
            return {
                "enable": False,
                "denoise": 0.0,
                "scale": 1.0,
                "upscaler": "Use same upscaler",
                "steps": 0,
                "resize_x": 0,
                "resize_y": 0,
            }
        data = self.model_dump(exclude_none=True)
        data.setdefault("enable", True)
        return data


class Txt2ImgPayload(BaseModel):
    prompt: str = Field(description="Positive prompt text")
    negative_prompt: str = Field(default="", description="Negative prompt text")
    width: int = Field(default=512, ge=8, le=8192)
    height: int = Field(default=512, ge=8, le=8192)
    steps: int = Field(default=20, ge=1)
    guidance_scale: float = Field(default=7.0, ge=0.0)
    sampler: str = Field(default="automatic")
    scheduler: str = Field(default="Automatic")
    seed: int = Field(default_factory=_random_seed)
    batch_size: int = Field(default=1, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extras: Dict[str, Any] = Field(default_factory=dict)
    highres: Optional[HighresOptions] = None
    engine: Optional[str] = Field(
        default=None, description="Engine identifier (defaults to codex_engine option)"
    )
    model: Optional[str] = Field(
        default=None, description="Model reference (defaults to sd_model_checkpoint option)"
    )

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_eight(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("value must be a multiple of 8")
        return value


class Img2ImgPayload(BaseModel):
    prompt: str = Field(description="Positive prompt text")
    negative_prompt: str = Field(default="", description="Negative prompt text")
    steps: int = Field(default=20, ge=1)
    guidance_scale: float = Field(default=7.0, ge=0.0)
    sampler: str = Field(default="automatic")
    scheduler: str = Field(default="Automatic")
    seed: int = Field(default_factory=_random_seed)
    denoise_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    width: int = Field(default=512, ge=8, le=8192)
    height: int = Field(default=512, ge=8, le=8192)
    init_image: str = Field(description="Base64 encoded init image")
    mask: Optional[str] = Field(default=None, description="Optional base64 encoded mask image")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extras: Dict[str, Any] = Field(default_factory=dict)
    engine: Optional[str] = None
    model: Optional[str] = None

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_eight(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("value must be a multiple of 8")
        return value


class GenerationResponse(BaseModel):
    images: List[str]
    info: Any
    parameters: Dict[str, Any]


class ProgressResponse(BaseModel):
    progress: float
    eta_relative: float
    state: Dict[str, Any]
    current_image: Optional[str] = None
    textinfo: Optional[str] = None
    current_task: Optional[str] = None


@router.post("/txt2img", response_model=GenerationResponse)
def txt2img(payload: Txt2ImgPayload) -> GenerationResponse:
    snapshot = codex_options.get_snapshot()
    engine_key = payload.engine or snapshot.codex_engine
    model_ref = payload.model or snapshot.sd_model_checkpoint

    metadata = dict(payload.metadata)
    metadata.setdefault("mode", snapshot.codex_mode)
    metadata.setdefault("batch_size", payload.batch_size)

    sampler_name, scheduler_name = _sampler_service.resolve(payload.sampler, payload.scheduler)

    req = Txt2ImgRequest(
        task=TaskType.TXT2IMG,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        sampler=sampler_name,
        scheduler=scheduler_name,
        seed=payload.seed,
        batch_size=payload.batch_size,
        metadata=metadata,
        extras=dict(payload.extras),
        highres_fix=payload.highres.as_payload() if payload.highres else None,
    )

    service = ImageService()
    result = service.txt2img(req, engine_key=engine_key, model_ref=model_ref)
    images = [item.get("data") for item in result.get("images", []) if "data" in item]
    return GenerationResponse(images=images, info=result.get("info", {}), parameters=payload.model_dump())


@router.post("/img2img", response_model=GenerationResponse)
def img2img(payload: Img2ImgPayload) -> GenerationResponse:
    snapshot = codex_options.get_snapshot()
    engine_key = payload.engine or snapshot.codex_engine
    model_ref = payload.model or snapshot.sd_model_checkpoint

    try:
        init_image = _media_service.decode_image(payload.init_image)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid init image: {err}") from err

    mask_image = None
    if payload.mask:
        try:
            mask_image = _media_service.decode_image(payload.mask)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid mask image: {err}") from err

    metadata = dict(payload.metadata)
    metadata.setdefault("mode", snapshot.codex_mode)

    sampler_name, scheduler_name = _sampler_service.resolve(payload.sampler, payload.scheduler)

    req = Img2ImgRequest(
        task=TaskType.IMG2IMG,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        sampler=sampler_name,
        scheduler=scheduler_name,
        seed=payload.seed,
        denoise_strength=payload.denoise_strength,
        init_image=init_image,
        mask=mask_image,
        metadata=metadata,
        extras=dict(payload.extras),
    )

    service = ImageService()
    result = service.img2img(req, engine_key=engine_key, model_ref=model_ref)
    images = [item.get("data") for item in result.get("images", []) if "data" in item]
    return GenerationResponse(images=images, info=result.get("info", {}), parameters=payload.model_dump())


@router.get("/progress", response_model=ProgressResponse)
def progress(skip_current_image: bool = False) -> ProgressResponse:
    progress_service = ProgressService(_media_service)
    data = progress_service.compute(skip_current_image=skip_current_image)
    image_blob = data.get("current_image")
    if isinstance(image_blob, bytes):
        image_blob = base64.b64encode(image_blob).decode()
    return ProgressResponse(
        progress=float(data.get("progress", 0.0)),
        eta_relative=float(data.get("eta_relative", 0.0)),
        state=dict(data.get("state", {})),
        current_image=image_blob,
        textinfo=data.get("textinfo"),
        current_task=data.get("current_task"),
    )


@router.post("/interrupt")
def interrupt() -> Dict[str, Any]:
    backend_state.interrupt()
    return {"status": "ok"}


@router.post("/skip")
def skip() -> Dict[str, Any]:
    backend_state.skip()
    return {"status": "ok"}


@router.get("/options")
def get_options() -> Dict[str, Any]:
    return _options_service.get_config()


@router.post("/options")
def set_options(payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = _options_service.set_config(payload)
    return {"updated": bool(updated)}


@router.get("/cmd-flags")
def get_cmd_flags() -> Dict[str, Any]:
    return _options_service.get_cmd_flags()


@router.get("/samplers")
def list_samplers() -> Dict[str, Any]:
    from apps.backend.runtime.sampling.registry import get_sampler_spec

    payload = []
    for entry in SAMPLER_OPTIONS:
        if not entry.get("supported", True):
            continue
        spec = None
        try:
            spec = get_sampler_spec(entry["name"])
        except Exception:
            pass
        payload.append(
            {
                "name": entry["name"],
                "label": entry.get("label", entry["name"].title()),
                "aliases": [alias.strip() for alias in entry.get("aliases", []) if isinstance(alias, str) and alias.strip()],
                "supported": bool(entry.get("supported", True)),
                "default_scheduler": spec.default_scheduler if spec else None,
                "allowed_schedulers": sorted(spec.allowed_schedulers) if spec else [],
            }
        )
    return {"samplers": payload}


@router.get("/schedulers")
def list_schedulers() -> Dict[str, Any]:
    schedulers = []
    for entry in SCHEDULER_OPTIONS:
        if not entry.get("supported", True):
            continue
        schedulers.append(
            {
                "name": entry["name"],
                "label": entry.get("label", entry["name"].title()),
                "aliases": [alias.strip() for alias in entry.get("aliases", []) if isinstance(alias, str) and alias.strip()],
                "supported": bool(entry.get("supported", True)),
            }
        )
    return {"schedulers": schedulers}


@router.get("/sd-models")
def list_sd_models() -> Dict[str, Any]:
    try:
        entries = model_api.list_checkpoints()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to list checkpoints: {err}") from err

    models = [
        {
            "title": entry.title,
            "name": entry.name,
            "model_name": entry.model_name,
            "hash": entry.short_hash,
            "filename": entry.filename,
            "path": entry.path,
            "metadata": dict(entry.metadata),
        }
        for entry in entries
    ]
    snapshot = codex_options.get_snapshot()
    return {"models": models, "current": snapshot.sd_model_checkpoint}


@router.post("/refresh-checkpoints")
def refresh_checkpoints() -> Dict[str, Any]:
    try:
        model_api.refresh()
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to refresh checkpoints: {err}") from err
    return {"refreshed": True}


@router.get("/memory")
def memory() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        res = process.memory_info()
        total = res.rss / max(process.memory_percent() or 1.0, 1e-6)
        stats["ram"] = {
            "free": max(total - res.rss, 0),
            "used": res.rss,
            "total": total,
        }
    except Exception as err:
        stats["ram"] = {"error": str(err)}

    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            stats["cuda"] = {
                "system": {"free": free, "total": total, "used": total - free},
                "events": {
                    "retries": torch.cuda.memory_stats().get("num_alloc_retries", 0),
                    "oom": torch.cuda.memory_stats().get("num_ooms", 0),
                },
            }
        else:
            stats["cuda"] = {"error": "cuda unavailable"}
    except Exception as err:
        stats["cuda"] = {"error": str(err)}

    return stats


@router.get("/memory/models")
def list_loaded_models() -> Dict[str, Any]:
    """Inspect models currently tracked as loaded by the memory manager.

    Returns a lightweight view (no tensors) with identifiers that can be used
    for targeted unload operations.
    """
    try:
        records = list(memory_management.current_loaded_models)
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to inspect loaded models: {err}") from err

    models: List[Dict[str, Any]] = []
    for record in records:
        model_obj = getattr(record, "model", record)
        entry: Dict[str, Any] = {
            "id": id(model_obj),
            "type": type(model_obj).__name__,
            "loader": type(getattr(record, "loader", None)).__name__
            if getattr(record, "loader", None) is not None
            else None,
            "base_module": type(getattr(record, "base_module", None)).__name__
            if getattr(record, "base_module", None) is not None
            else None,
            "load_device": str(getattr(record, "load_device", None)),
            "offload_device": str(getattr(record, "offload_device", None)),
            "storage_dtype": str(getattr(record, "storage_dtype", None)),
            "inclusive_memory": int(getattr(record, "inclusive_memory", 0) or 0),
            "exclusive_memory": int(getattr(record, "exclusive_memory", 0) or 0),
            "accelerated": bool(getattr(record, "model_accelerated", False)),
        }
        models.append(entry)

    return {"count": len(models), "models": models}


@router.post("/memory/unload")
def unload_models(payload: MemoryUnloadRequest = Body(default_factory=MemoryUnloadRequest)) -> Dict[str, Any]:
    """Unload models from GPU/primary device.

    - scope='all'  → unload all tracked models and clear the CUDA cache.
    - scope='ids'  → unload only the models whose id() values are listed.
    """
    if payload.scope == "all":
        try:
            memory_management.unload_all_models()
            memory_management.soft_empty_cache(force=True)
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to unload all models: {err}") from err
        logger.info("memory-unload scope=all")
        return {"scope": "all", "unloaded": "all"}

    ids = set(payload.ids or [])
    if not ids:
        raise HTTPException(
            status_code=400,
            detail="When scope=='ids', 'ids' must be a non-empty list of model ids.",
        )

    records = list(memory_management.current_loaded_models)
    unloaded: List[int] = []
    for record in records:
        model_obj = getattr(record, "model", record)
        mid = id(model_obj)
        if mid in ids:
            try:
                memory_management.unload_model(model_obj)
                unloaded.append(mid)
            except Exception:  # noqa: BLE001
                # Keep going; we'll report which ids we failed to unload.
                continue

    missing = sorted(mid for mid in ids if mid not in unloaded)
    logger.info(
        "memory-unload scope=ids requested=%d unloaded=%d missing=%d",
        len(ids),
        len(unloaded),
        len(missing),
    )
    return {
        "scope": "ids",
        "requested": sorted(ids),
        "unloaded": sorted(unloaded),
        "missing": missing,
    }


# --- Deferred endpoints (return 501 with backlog reference) -----------------


@router.post("/extra-single-image")
def extras_single_image() -> None:
    _raise_not_implemented("extra-single-image postprocessing", "NT-1")


@router.post("/extra-batch-images")
def extras_batch_images() -> None:
    _raise_not_implemented("extra-batch-images postprocessing", "NT-1")


@router.get("/face-restorers")
def face_restorers() -> None:
    _raise_not_implemented("face restorers registry", "NT-1")


@router.get("/upscalers")
def upscalers() -> None:
    _raise_not_implemented("upscalers registry", "NT-2")


@router.get("/latent-upscale-modes")
def latent_upscale_modes() -> None:
    _raise_not_implemented("latent upscale modes", "NT-2")


@router.get("/realesrgan-models")
def realesrgan_models() -> None:
    _raise_not_implemented("RealESRGAN models registry", "NT-2")


@router.post("/refresh-vae")
def refresh_vae() -> None:
    _raise_not_implemented("VAE refresh", "NT-2")


@router.get("/prompt-styles")
def prompt_styles() -> None:
    _raise_not_implemented("prompt styles registry", "NT-3")


@router.get("/embeddings")
def embeddings() -> None:
    _raise_not_implemented("embeddings registry", "NT-3")


@router.post("/refresh-embeddings")
def refresh_embeddings() -> None:
    _raise_not_implemented("embeddings refresh", "NT-3")


@router.post("/interrogate")
def interrogate() -> None:
    _raise_not_implemented("interrogate endpoint", "NT-4")


@router.post("/png-info")
def png_info() -> None:
    _raise_not_implemented("png-info parser", "NT-6")


@router.get("/sd-modules")
def sd_modules() -> None:
    _raise_not_implemented("sd modules listing", "NT-6")


@router.get("/hypernetworks")
def hypernetworks() -> None:
    _raise_not_implemented("hypernetworks registry", "NH-2")


@router.post("/unload-checkpoint")
def unload_checkpoint() -> None:
    _raise_not_implemented("checkpoint unload", "IM-3 follow-up")


@router.post("/reload-checkpoint")
def reload_checkpoint() -> None:
    _raise_not_implemented("checkpoint reload", "IM-3 follow-up")


@router.post("/create/embedding")
def create_embedding() -> None:
    _raise_not_implemented("embedding training", "SP-1")


@router.post("/create/hypernetwork")
def create_hypernetwork() -> None:
    _raise_not_implemented("hypernetwork training", "SP-1")


@router.get("/scripts")
def scripts_list() -> None:
    _raise_not_implemented("scripts listing", "SP-2")


@router.get("/script-info")
def script_info() -> None:
    _raise_not_implemented("script info", "SP-2")


@router.get("/extensions")
def extensions_list() -> None:
    _raise_not_implemented("extensions listing", "SP-2")


@router.post("/server-kill")
def server_kill() -> None:
    _raise_not_implemented("server-kill command", "SP-2")


@router.post("/server-restart")
def server_restart() -> None:
    _raise_not_implemented("server-restart command", "SP-2")


@router.post("/server-stop")
def server_stop() -> None:
    _raise_not_implemented("server-stop command", "SP-2")
