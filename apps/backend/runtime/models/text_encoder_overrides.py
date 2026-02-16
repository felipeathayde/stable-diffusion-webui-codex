"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Text encoder override definitions and resolution helpers.
Holds the override config dataclass, strict validation rules, and filesystem resolution for encoder weights.
Supports explicit WAN22 variant families in override labels (`WAN22_5B`/`WAN22_14B`/`WAN22_ANIMATE`).

Symbols (top-level; keep in sync; no ghosts):
- `TextEncoderOverrideError` (class): Raised when a text encoder override configuration cannot be applied.
- `TextEncoderOverrideConfig` (dataclass): Normalized TE override description (family/label/components; supports explicit path components).
- `_canonical_override_family` (function): Canonicalizes override “family” for loader semantics (so UI/API can use stable labels).
- `resolve_text_encoder_override_paths` (function): Resolves a TE override config into explicit weight paths (including `alias=/abs/path` entries).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.infra.registry.text_encoder_roots import list_text_encoder_roots_by_family
from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature


class TextEncoderOverrideError(RuntimeError):
    """Raised when a text encoder override configuration cannot be applied."""


@dataclass(frozen=True)
class TextEncoderOverrideConfig:
    """Explicit selection of a text encoder root for a given model family.

    family:
        Concrete model family (`ModelFamily.SD15`, `ModelFamily.SDXL`, `ModelFamily.FLUX`, `ModelFamily.WAN22_5B`, ...)
        that this override is valid for.
    root_label:
        A stable label in the form `<family>/<path>` where `<path>` is either repo-relative or absolute and
        originates from the configured `*_tenc` entries in `apps/paths.json` (exposed to the UI via `/api/paths`).
    components:
        Optional subset of logical text encoder aliases (`clip_l`, `clip_g`, `t5xxl`, `umt5xxl`, ...).
        When omitted, all encoders in `ModelSignature.text_encoders` are expected to have weights
        under the selected root or explicit path map.
    explicit_paths:
        Optional mapping from logical alias -> absolute weight path, e.g.
        ``{"clip_l": "/abs/.../clip_l_fp8.safetensors"}``. When provided, the loader
        will bypass root resolution for those aliases and use the explicit paths instead.
    """

    family: ModelFamily
    root_label: str
    components: tuple[str, ...] | None = None
    explicit_paths: Dict[str, str] | None = None


def _canonical_override_family(family: ModelFamily) -> ModelFamily:
    """Map specialised families to their override bucket.

    For example, SDXL refiner shares text encoder family with SDXL.
    """

    if family is ModelFamily.SDXL_REFINER:
        return ModelFamily.SDXL
    return family


def resolve_text_encoder_override_paths(
    *,
    signature: ModelSignature,
    estimated_config: CodexEstimatedConfig,
    override: TextEncoderOverrideConfig | None,
) -> Dict[str, str]:
    """Resolve a text encoder override into concrete component weight paths.

    Returns a mapping from Diffusers component name to absolute weight path,
    e.g. ``{"text_encoder": "/abs/.../clip_l.safetensors"}``.

    This helper is intentionally pure: it validates invariants and inspects
    the filesystem, but leaves loading of the actual state dicts to callers.
    """

    if override is None:
        return {}

    model_family = _canonical_override_family(signature.family)
    if override.family is not model_family:
        raise TextEncoderOverrideError(
            "Text encoder override family=%s is not compatible with model family=%s"
            % (override.family.value, model_family.value)
        )

    allowed_exts = (".safetensors", ".gguf", ".bin", ".pt")

    text_map = dict(getattr(estimated_config, "text_encoder_map", {}) or {})

    # Fast path: explicit alias -> path mapping (e.g. Flux file-level overrides).
    explicit = dict(override.explicit_paths or {})
    if explicit:
        if override.components:
            aliases = tuple(override.components)
        else:
            aliases = tuple(explicit.keys())
        if not aliases:
            raise TextEncoderOverrideError(
                "Text encoder override for family=%s provided explicit paths but no aliases."
                % model_family.value
            )
        missing_aliases = [alias for alias in aliases if alias not in text_map]
        if missing_aliases:
            raise TextEncoderOverrideError(
                "Text encoder override refers to unknown encoder aliases for family=%s: %s"
                % (model_family.value, ", ".join(sorted(missing_aliases)))
            )
        component_paths: Dict[str, str] = {}
        for alias in aliases:
            path = explicit.get(alias)
            if not path:
                raise TextEncoderOverrideError(
                    "Text encoder override missing explicit path for alias %r (family=%s)"
                    % (alias, model_family.value)
                )
            norm = str(path)
            if not os.path.isfile(norm):
                raise TextEncoderOverrideError(
                    "Text encoder override path for alias %r is not a file: %r"
                    % (alias, norm)
                )
            if not norm.lower().endswith(allowed_exts):
                raise TextEncoderOverrideError(
                    "Text encoder override path for alias %r must end with one of: %s"
                    % (alias, ", ".join(allowed_exts))
                )
            component_name = text_map[alias]
            component_paths[component_name] = norm
        return component_paths

    # Root-based path resolution using configured roots (paths.json-backed).
    roots_by_family = list_text_encoder_roots_by_family()
    family_roots = list(roots_by_family.get(model_family.value) or [])
    root_path: str | None = None

    label = str(override.root_label or "").strip()
    label_path_hint = label
    if "/" in label:
        prefix, rest = label.split("/", 1)
        if prefix.strip() == model_family.value:
            label_path_hint = rest
    label_path_hint = label_path_hint.strip()

    resolved_hint: str | None = None
    if label_path_hint:
        try:
            p = Path(os.path.expanduser(label_path_hint))
        except Exception:
            resolved_hint = None
        else:
            if not p.is_absolute():
                p = get_repo_root() / p
            try:
                resolved_hint = str(p.resolve(strict=False))
            except Exception:
                resolved_hint = str(p)

    for entry in family_roots:
        if getattr(entry, "name", None) == label:
            root_path = getattr(entry, "path", None)
            break
        if resolved_hint and getattr(entry, "path", None):
            try:
                entry_path = str(Path(str(getattr(entry, "path"))).resolve(strict=False))
            except Exception:
                entry_path = str(getattr(entry, "path"))
            if entry_path == resolved_hint:
                root_path = getattr(entry, "path", None)
                break

    if root_path is None:
        raise TextEncoderOverrideError(
            "Text encoder override label %r not found for family=%s. "
            "Update apps/paths.json (`%s_tenc`) and choose a valid label."
            % (override.root_label, model_family.value, model_family.value)
        )

    root_path = str(root_path)
    if not os.path.isdir(root_path):
        raise TextEncoderOverrideError(
            "Text encoder override root %r path is not a directory: %r"
            % (override.root_label, root_path)
        )

    # Decide which logical encoders we expect under this root.
    if override.components:
        aliases = tuple(override.components)
    else:
        aliases = tuple(te.name for te in signature.text_encoders)

    if not aliases:
        raise TextEncoderOverrideError(
            "Model family %s declares no text encoders; override cannot be applied."
            % model_family.value
        )

    missing_aliases = [alias for alias in aliases if alias not in text_map]
    if missing_aliases:
        raise TextEncoderOverrideError(
            "Text encoder override refers to unknown encoder aliases for family=%s: %s"
            % (model_family.value, ", ".join(sorted(missing_aliases)))
        )

    # Strict file naming: each alias must have a single weights file named
    # <alias>.<ext> under the selected root. No guessing across families.
    try:
        entries = set(os.listdir(root_path))
    except Exception as exc:  # pragma: no cover - defensive
        raise TextEncoderOverrideError(
            "Failed to list text encoder override root %r: %s" % (root_path, exc)
        ) from exc

    component_paths: Dict[str, str] = {}

    for alias in aliases:
        found = None
        for ext in allowed_exts:
            candidate = alias + ext
            if candidate in entries:
                found = os.path.join(root_path, candidate)
                break
        if not found:
            expected = ", ".join(alias + ext for ext in allowed_exts)
            raise TextEncoderOverrideError(
                "Text encoder override root %r is missing weights for encoder %r. "
                "Expected one of: %s"
                % (override.root_label, alias, expected)
            )
        component_name = text_map[alias]
        component_paths[component_name] = found

    return component_paths
