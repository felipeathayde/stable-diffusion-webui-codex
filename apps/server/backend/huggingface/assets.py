from __future__ import annotations

import os
import shutil
import time
from typing import Iterable


def _has_config(local_path: str) -> bool:
    candidates = (
        "model_index.json",
        "configuration.json",
        "config.json",
    )
    for name in candidates:
        if os.path.isfile(os.path.join(local_path, name)):
            return True
    # WAN repositories place configs under stage directories
    for stage in ("high_noise_model", "low_noise_model"):
        if os.path.isfile(os.path.join(local_path, stage, "config.json")):
            return True
    return False


def _has_tokenizer(local_path: str) -> bool:
    for root, _, files in os.walk(local_path):
        files = set(files)
        if {"tokenizer.json", "tokenizer_config.json"}.issubset(files):
            return True
        if {"spiece.model", "tokenizer_config.json"}.issubset(files):
            return True
        if {"tokenizer.model", "tokenizer_config.json"}.issubset(files):
            return True
        if {"vocab.json", "merges.txt"}.issubset(files):
            return True
    return False


def _has_scheduler(local_path: str) -> bool:
    sch_dir = os.path.join(local_path, "scheduler")
    if not os.path.isdir(sch_dir):
        return False
    for candidate in ("scheduler_config.json", "config.json"):
        if os.path.isfile(os.path.join(sch_dir, candidate)):
            return True
    # fall back to generic *.json detection
    for entry in os.listdir(sch_dir):
        if entry.endswith(".json"):
            return True
    return False


def ensure_repo_minimal_files(
    repo_id: str,
    local_path: str,
    *,
    offline: bool = False,
    include: Iterable[str] | None = None,
) -> None:
    """Ensure minimal diffusers assets (config/tokenizer/scheduler) exist under local_path.

    Downloads only lightweight JSON/TXT/MODEL files from Hugging Face when permitted.
    """
    include = tuple(include) if include is not None else ("config", "tokenizer", "scheduler")

    need: list[str] = []
    if "config" in include and not _has_config(local_path):
        need.append("config")
    if "tokenizer" in include and not _has_tokenizer(local_path):
        need.append("tokenizer")
    if "scheduler" in include and not _has_scheduler(local_path):
        need.append("scheduler")

    if not need:
        return

    if offline:
        missing = ", ".join(need)
        raise RuntimeError(
            (
                f"Missing required assets ({missing}) for repo '{repo_id}' in '{local_path}'.\n"
                "Strict offline mode is enabled; no online downloads were attempted.\n"
                "Populate the directory manually with the expected diffusers files and retry."
            )
        )

    patterns: set[str] = set()
    if "config" in need:
        patterns.update({
            "model_index.json",
            "configuration.json",
            "config.json",
            "high_noise_model/config.json",
            "low_noise_model/config.json",
        })
    if "tokenizer" in need:
        patterns.update(
            {
                "tokenizer/tokenizer.json",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/tokenizer.model",
                "tokenizer/*.json",
                "tokenizer/*.txt",
                "tokenizer/*.model",
                "tokenizer_2/tokenizer.json",
                "tokenizer_2/vocab.json",
                "tokenizer_2/merges.txt",
                "tokenizer_2/tokenizer.model",
                "tokenizer_2/*.json",
                "tokenizer_2/*.txt",
                "tokenizer_2/*.model",
                "google/umt5-xxl/tokenizer.json",
                "google/umt5-xxl/tokenizer_config.json",
                "google/umt5-xxl/vocab.json",
                "google/umt5-xxl/merges.txt",
                "google/umt5-xxl/tokenizer.model",
                "google/umt5-xxl/spiece.model",
                "google/umt5-xxl/*.json",
                "google/umt5-xxl/*.model",
                "google/umt5-xxl/*.txt",
            }
        )
    if "scheduler" in need:
        patterns.update(
            {
                "scheduler/scheduler_config.json",
                "scheduler/config.json",
                "scheduler/*.json",
                "scheduler/*.txt",
            }
        )

    if not patterns:
        return

    os.makedirs(local_path, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        cached_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            allow_patterns=list(patterns),
            local_files_only=False,
            force_download=False,
        )
        cache_dir = os.path.join(local_path, ".cache")
        if os.path.isdir(cache_dir):
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
        # Ensure snapshot_download target isn't nested under local_path unexpectedly
        if cached_dir != local_path and os.path.isdir(cached_dir):
            _copy_selected_files(cached_dir, local_path, patterns)
    except (ImportError, TypeError, OSError):
        _http_list_and_download(repo_id, local_path, patterns)
    except Exception:
        _http_list_and_download(repo_id, local_path, patterns)


def _copy_selected_files(src_root: str, dst_root: str, patterns: Iterable[str]) -> None:
    os.makedirs(dst_root, exist_ok=True)

    def _match(rel_path: str) -> bool:
        rel_path = rel_path.replace("\\", "/")
        for patt in patterns:
            patt = patt.replace("\\", "/")
            if patt.endswith("/*"):
                if rel_path.startswith(patt[:-1]):
                    return True
            elif patt.endswith("*"):
                if rel_path.startswith(patt[:-1]):
                    return True
            else:
                if rel_path == patt:
                    return True
        return False

    for root, _, files in os.walk(src_root):
        rel_dir = os.path.relpath(root, src_root)
        for fname in files:
            rel_path = os.path.normpath(os.path.join(rel_dir, fname)) if rel_dir != os.curdir else fname
            rel_path = rel_path.replace("\\", "/")
            if not _match(rel_path):
                continue
            src = os.path.join(root, fname)
            dst = os.path.join(dst_root, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def _http_list_and_download(repo_id: str, dst_root: str, patterns: Iterable[str]) -> None:
    import httpx

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    max_attempts = 3

    def _should_retry(exc: Exception) -> bool:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (429, 502, 503, 504)

    def _match(path: str) -> bool:
        path = path.replace("\\", "/")
        for patt in patterns:
            patt = patt.replace("\\", "/")
            if patt.endswith("/*"):
                if path.startswith(patt[:-1]):
                    return True
            elif patt.endswith("*"):
                if path.startswith(patt[:-1]):
                    return True
            else:
                if path == patt:
                    return True
        return False

    with httpx.Client(headers=headers, timeout=15.0, follow_redirects=True) as client:
        for attempt in range(max_attempts):
            try:
                resp = client.get(api_url)
                resp.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                if attempt + 1 < max_attempts and _should_retry(exc):
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise

        else:
            raise RuntimeError(f"Failed to query Hugging Face API for repo '{repo_id}'.")

        siblings = (resp.json() or {}).get("siblings") or []
        filepaths = [s.get("rfilename") for s in siblings if s.get("rfilename")]
        wanted = [p for p in filepaths if _match(p)]

        if not wanted:
            raise RuntimeError(f"No files matched required patterns in repo '{repo_id}'. Patterns={patterns}")

        for rel in wanted:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{rel}"
            out = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for attempt in range(max_attempts):
                try:
                    with client.stream("GET", url) as stream:
                        stream.raise_for_status()
                        with open(out, "wb") as f:
                            for chunk in stream.iter_bytes():
                                f.write(chunk)
                    break
                except httpx.HTTPStatusError as exc:
                    if attempt + 1 < max_attempts and _should_retry(exc):
                        time.sleep(0.5 * (2**attempt))
                        continue
                    raise


__all__ = ["ensure_repo_minimal_files"]
