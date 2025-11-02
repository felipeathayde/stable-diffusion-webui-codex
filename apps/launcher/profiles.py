from __future__ import annotations

import json
import logging
import os
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Tuple

LOGGER = logging.getLogger("codex.launcher.profiles")

ENV_PREFIX_AREAS: Dict[str, str] = {
    "CODEX_": "core",
    "WAN_": "wan",
}


def _default_area_env() -> Dict[str, Dict[str, str]]:
    """Compute default environment values partitioned by area."""
    core = {
        "CODEX_ATTENTION_BACKEND": os.getenv("CODEX_ATTENTION_BACKEND", "torch-sdpa"),
        "CODEX_ATTN_CHUNK_SIZE": os.getenv("CODEX_ATTN_CHUNK_SIZE", "0"),
        "CODEX_GGUF_CACHE_POLICY": os.getenv("CODEX_GGUF_CACHE_POLICY", "none"),
        "CODEX_GGUF_CACHE_LIMIT_MB": os.getenv("CODEX_GGUF_CACHE_LIMIT_MB", "0"),
        "CODEX_DIFFUSION_DTYPE": os.getenv("CODEX_DIFFUSION_DTYPE", "fp16"),
        "CODEX_DIFFUSION_DEVICE": os.getenv("CODEX_DIFFUSION_DEVICE", ""),
        "CODEX_VAE_DTYPE": os.getenv("CODEX_VAE_DTYPE", "fp16"),
        "CODEX_VAE_IN_CPU": os.getenv("CODEX_VAE_IN_CPU", "0"),
        "CODEX_VAE_DEVICE": os.getenv("CODEX_VAE_DEVICE", ""),
        "CODEX_SWAP_POLICY": os.getenv("CODEX_SWAP_POLICY", "cpu"),
        "CODEX_SWAP_METHOD": os.getenv("CODEX_SWAP_METHOD", "blocked"),
        "CODEX_GPU_PREFER_CONSTRUCT": os.getenv("CODEX_GPU_PREFER_CONSTRUCT", "0"),
        "CODEX_PIPELINE_DEBUG": os.getenv("CODEX_PIPELINE_DEBUG", "0"),
    }
    wan = {
        "WAN_SDPA_DEBUG": os.getenv("WAN_SDPA_DEBUG", "0"),
        "WAN_SDPA_DEBUG_EVERY": os.getenv("WAN_SDPA_DEBUG_EVERY", "5"),
        "WAN_SDPA_POLICY": os.getenv("WAN_SDPA_POLICY", "mem_efficient"),
        "WAN_I2V_DEBUG_HI_DECODE": os.getenv("WAN_I2V_DEBUG_HI_DECODE", "0"),
        "WAN_I2V_LAT_STATS": os.getenv("WAN_I2V_LAT_STATS", "0"),
        "WAN_I2V_STRICT_VAE": os.getenv("WAN_I2V_STRICT_VAE", "0"),
        "WAN_I2V_CONV32": os.getenv("WAN_I2V_CONV32", "0"),
        "WAN_I2V_ORDER": os.getenv("WAN_I2V_ORDER", "lat_first"),
        "WAN_I2V_DEBUG_CLAMP": os.getenv("WAN_I2V_DEBUG_CLAMP", ""),
        "WAN_I2V_DEBUG_SANITIZE_TOKENS": os.getenv("WAN_I2V_DEBUG_SANITIZE_TOKENS", "0"),
        "WAN_TE_IMPL": os.getenv("WAN_TE_IMPL", "hf"),
        "WAN_TE_DEVICE": os.getenv("WAN_TE_DEVICE", os.getenv("CODEX_TE_DEVICE", "")),
        "WAN_TE_KERNEL_REQUIRED": os.getenv("WAN_TE_KERNEL_REQUIRED", "0"),
        "WAN_GGUF_OFFLOAD_LEVEL": os.getenv("WAN_GGUF_OFFLOAD_LEVEL", "3"),
        "WAN_LOG_INFO": os.getenv("WAN_LOG_INFO", "1"),
        "WAN_LOG_WARN": os.getenv("WAN_LOG_WARN", "1"),
        "WAN_LOG_ERROR": os.getenv("WAN_LOG_ERROR", "1"),
        "WAN_LOG_DEBUG": os.getenv("WAN_LOG_DEBUG", "0"),
    }
    return {"core": core, "wan": wan}


DEFAULT_MODEL_NAME = "default"


@dataclass
class LauncherMeta:
    external_terminal: bool = False
    sdpa_policy: str = "mem_efficient"
    tab_index: int = 0
    active_model: str = DEFAULT_MODEL_NAME


class _EnvironmentView(MutableMapping[str, str]):
    """Mutable mapping that routes env mutations to areas/models."""

    def __init__(self, store: "LauncherProfileStore") -> None:
        self._store = store

    def __getitem__(self, key: str) -> str:
        value = self._store.lookup_env(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: str) -> None:
        value = str(value)
        target_map, target_kind = self._store.resolve_container_for_key(key)
        target_map[key] = value
        if key == "WAN_SDPA_POLICY":
            self._store.meta.sdpa_policy = value
        LOGGER.debug("Set env %s=%s (container=%s)", key, value, target_kind)

    def __delitem__(self, key: str) -> None:
        removed = False
        for area, mapping in self._store.areas.items():
            if key in mapping:
                del mapping[key]
                removed = True
                LOGGER.debug("Deleted env %s from area %s", key, area)
                break
        if not removed:
            model_map = self._store.models.get(self._store.meta.active_model, {})
            if key in model_map:
                del model_map[key]
                removed = True
                LOGGER.debug("Deleted env %s from model %s", key, self._store.meta.active_model)
        if not removed:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.build_env())

    def __len__(self) -> int:
        return len(self._store.build_env())

    def get(self, key: str, default: str | None = None) -> str | None:  # type: ignore[override]
        try:
            return self.__getitem__(key)
        except KeyError:
            return default


@dataclass
class LauncherProfileStore:
    root: Path
    meta: LauncherMeta
    areas: Dict[str, Dict[str, str]] = field(default_factory=dict)
    models: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _env_view: _EnvironmentView | None = field(default=None, init=False, repr=False)

    @classmethod
    def load(cls, root: Path | None = None) -> "LauncherProfileStore":
        root = root or _default_root()
        _ensure_tree(root)
        _maybe_migrate_legacy(root)
        meta = _load_meta(root)
        areas = _load_areas(root)
        models = _load_models(root)
        store = cls(root=root, meta=meta, areas=areas, models=models)
        store._ensure_consistency()
        return store

    @property
    def env(self) -> _EnvironmentView:
        if self._env_view is None:
            self._env_view = _EnvironmentView(self)
        return self._env_view

    def build_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for mapping in self.areas.values():
            env.update(mapping)
        active_model = self.meta.active_model
        env.update(self.models.get(active_model, {}))
        return env

    def lookup_env(self, key: str) -> str | None:
        for prefix, area in ENV_PREFIX_AREAS.items():
            if key.startswith(prefix):
                return self.areas.get(area, {}).get(key)
        active_model = self.meta.active_model
        if key in self.models.get(active_model, {}):
            return self.models[active_model][key]
        # As a safeguard allow lookups in other stored maps
        for mapping in self.areas.values():
            if key in mapping:
                return mapping[key]
        for mapping in self.models.values():
            if key in mapping:
                return mapping[key]
        return None

    def resolve_container_for_key(self, key: str) -> Tuple[Dict[str, str], str]:
        for prefix, area in ENV_PREFIX_AREAS.items():
            if key.startswith(prefix):
                mapping = self.areas.setdefault(area, {})
                return mapping, f"area:{area}"
        model = self.meta.active_model
        mapping = self.models.setdefault(model, {})
        return mapping, f"model:{model}"

    def save(self) -> None:
        LOGGER.debug("Persisting launcher profile to %s", self.root)
        _ensure_tree(self.root)
        _write_meta(self.root, self.meta)
        _write_env_maps(self.root / "areas", self.areas)
        _write_env_maps(self.root / "models", self.models)

    # ------------------------------------------------------------------ internal

    def _ensure_consistency(self) -> None:
        defaults = _default_area_env()
        for area, values in defaults.items():
            current = self.areas.setdefault(area, {})
            for key, default in values.items():
                current.setdefault(key, default)
        if self.meta.active_model not in self.models:
            self.models[self.meta.active_model] = {}

        wan_env = self.areas.get("wan", {})
        if wan_env:
            policy = wan_env.get("WAN_SDPA_POLICY")
            if policy:
                self.meta.sdpa_policy = policy
        self.areas.setdefault("wan", {}).setdefault("WAN_SDPA_POLICY", self.meta.sdpa_policy)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".sangoi" / "launcher"


def _ensure_tree(root: Path) -> None:
    (root / "areas").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)


def _load_meta(root: Path) -> LauncherMeta:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        meta = LauncherMeta()
        _write_meta(root, meta)
        return meta
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read launcher meta {meta_path}: {exc}") from exc
    return LauncherMeta(
        external_terminal=bool(data.get("external_terminal", False)),
        sdpa_policy=str(data.get("sdpa_policy", "mem_efficient")),
        tab_index=int(data.get("tab_index", 0)),
        active_model=str(data.get("active_model", DEFAULT_MODEL_NAME)),
    )


def _write_meta(root: Path, meta: LauncherMeta) -> None:
    meta_path = root / "meta.json"
    payload = {
        "external_terminal": meta.external_terminal,
        "sdpa_policy": meta.sdpa_policy,
        "tab_index": meta.tab_index,
        "active_model": meta.active_model,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_areas(root: Path) -> Dict[str, Dict[str, str]]:
    defaults = _default_area_env()
    areas_dir = root / "areas"
    areas: Dict[str, Dict[str, str]] = {}
    for area, default_env in defaults.items():
        areas[area] = _read_env_file(areas_dir / f"{area}.json", default_env)
    for path in areas_dir.glob("*.json"):
        area = path.stem
        if area in areas:
            continue
        areas[area] = _read_env_file(path)
    return areas


def _load_models(root: Path) -> Dict[str, Dict[str, str]]:
    models_dir = root / "models"
    models: Dict[str, Dict[str, str]] = {}
    default_path = models_dir / f"{DEFAULT_MODEL_NAME}.json"
    models[DEFAULT_MODEL_NAME] = _read_env_file(default_path, {})
    for path in models_dir.glob("*.json"):
        model = path.stem
        if model in models:
            continue
        models[model] = _read_env_file(path)
    return models


def _write_env_maps(base_dir: Path, mapping: Dict[str, Dict[str, str]]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, env in mapping.items():
        out_path = base_dir / f"{name}.json"
        out_path.write_text(json.dumps(_stringify_dict(env), indent=2, sort_keys=True))


def _read_env_file(path: Path, defaults: Dict[str, str] | None = None) -> Dict[str, str]:
    if not path.exists():
        if defaults is None:
            defaults = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_stringify_dict(defaults), indent=2, sort_keys=True))
        return dict(defaults)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read launcher environment file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Launcher environment file {path} must contain a JSON object.")
    result: Dict[str, str] = {}
    for key, value in data.items():
        result[str(key)] = str(value)
    if defaults:
        for key, value in defaults.items():
            result.setdefault(key, value)
    return result


def _stringify_dict(data: Dict[str, str]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in data.items()}


def _maybe_migrate_legacy(root: Path) -> None:
    meta_path = root / "meta.json"
    if meta_path.exists():
        return
    legacy_path = root.parent / "tui-profile.json"
    if not legacy_path.exists():
        return
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read legacy profile {legacy_path}: {exc}") from exc

    LOGGER.info("Migrating legacy launcher profile from %s", legacy_path)
    legacy_env = {str(k): str(v) for k, v in (legacy.get("env") or {}).items()}
    meta = LauncherMeta(
        external_terminal=bool(legacy.get("external_terminal", False)),
        sdpa_policy=str(legacy.get("sdpa_policy", "mem_efficient")),
        tab_index=int(legacy.get("tab_index", 0)),
        active_model=DEFAULT_MODEL_NAME,
    )

    defaults = _default_area_env()
    areas = {area: dict(values) for area, values in defaults.items()}
    models = {DEFAULT_MODEL_NAME: {}}

    for key, value in legacy_env.items():
        container, _ = _resolve_container_static(key, areas, models, meta.active_model)
        container[key] = value
    # ensure consistency and write through
    store = LauncherProfileStore(root=root, meta=meta, areas=areas, models=models)
    store._ensure_consistency()
    store.save()
    backup_path = legacy_path.with_suffix(".legacy-backup")
    legacy_path.rename(backup_path)
    LOGGER.info("Legacy profile migrated; backup saved to %s", backup_path)


def _resolve_container_static(
    key: str,
    areas: Dict[str, Dict[str, str]],
    models: Dict[str, Dict[str, str]],
    active_model: str,
) -> Tuple[Dict[str, str], str]:
    for prefix, area in ENV_PREFIX_AREAS.items():
        if key.startswith(prefix):
            return areas.setdefault(area, {}), f"area:{area}"
    return models.setdefault(active_model, {}), f"model:{active_model}"
