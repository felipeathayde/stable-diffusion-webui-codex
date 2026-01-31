"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime settings tab for the Tk launcher.
Edits bootstrap-critical device defaults and global runtime knobs that must exist before the API starts.

Symbols (top-level; keep in sync; no ghosts):
- `RuntimeTab` (class): Runtime settings tab (devices + GGUF/LoRA + PyTorch alloc conf).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List

from apps.launcher.profiles import DEFAULT_PYTORCH_CUDA_ALLOC_CONF
from apps.launcher.settings import DEVICE_CHOICES, SettingValidationError, normalize_gguf_lora_env

from ..controller import LauncherController
from ..widgets import ScrollableFrame, add_help, add_section_header


class RuntimeTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        canvas_bg: str,
        mark_changed: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._canvas_bg = canvas_bg
        self._mark_changed = mark_changed

        self.frame: ttk.Frame | None = None

        self._var_core_device = tk.StringVar()
        self._var_te_device = tk.StringVar()
        self._var_vae_device = tk.StringVar()

        self._var_lora_apply_mode = tk.StringVar()
        self._var_gguf_exec = tk.StringVar()
        self._var_gguf_dequant_cache = tk.StringVar()
        self._var_lora_online_math = tk.StringVar()
        self._var_pytorch_alloc_conf = tk.StringVar()

        self._lora_math_combo: ttk.Combobox | None = None
        self._gguf_dequant_cache_combo: ttk.Combobox | None = None

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner

        row = 0
        row = add_section_header(body, row, "Device Defaults (bootstrap)")
        row = self._add_choice_combo(
            body,
            row,
            label="Core device (requires API restart):",
            var=self._var_core_device,
            choices=list(DEVICE_CHOICES),
            on_change=lambda: self._set_env_lower("CODEX_CORE_DEVICE", self._var_core_device.get()),
        )
        row = self._add_choice_combo(
            body,
            row,
            label="Text encoder device (requires API restart):",
            var=self._var_te_device,
            choices=list(DEVICE_CHOICES),
            on_change=lambda: self._set_env_lower("CODEX_TE_DEVICE", self._var_te_device.get()),
        )
        row = self._add_choice_combo(
            body,
            row,
            label="VAE device (requires API restart):",
            var=self._var_vae_device,
            choices=list(DEVICE_CHOICES),
            on_change=lambda: self._set_env_lower("CODEX_VAE_DEVICE", self._var_vae_device.get()),
        )
        row = add_help(
            body,
            row,
            "These values are passed as backend CLI flags (`--core-device/--te-device/--vae-device`).\n"
            "They exist so the API can start in non-interactive spawns without prompting or silent fallbacks.",
        )

        row = add_section_header(body, row, "GGUF / LoRA / PyTorch")
        row = self._add_choice_combo(
            body,
            row,
            label="LoRA apply mode (requires API restart):",
            var=self._var_lora_apply_mode,
            choices=["merge", "online"],
            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
            width=12,
        )
        row = add_help(
            body,
            row,
            "merge: rewrites weights once at apply-time (default).\n"
            "online: applies LoRA patches on-the-fly during forward.",
        )
        row = self._add_choice_combo(
            body,
            row,
            label="GGUF exec mode (requires API restart):",
            var=self._var_gguf_exec,
            choices=["dequant_forward", "dequant_upfront"],
            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
            width=18,
        )
        row = add_help(
            body,
            row,
            "dequant_forward: current default (GGUF weights dequantize on-demand during forward).\n"
            "dequant_upfront: dequantize GGUF weights at load time (uses more RAM/VRAM).\n"
            "CodexPack packed GGUFs are auto-detected via `*.codexpack.gguf` (no exec-mode toggle).",
        )
        row = self._add_choice_combo(
            body,
            row,
            label="GGUF dequant cache (requires API restart):",
            var=self._var_gguf_dequant_cache,
            choices=["off", "lvl1", "lvl2"],
            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
            width=10,
            out_combo="_gguf_dequant_cache_combo",
        )
        row = add_help(
            body,
            row,
            "off: no cache.\n"
            "lvl1: cache moved+baked GGUF parameters per sampling run (reuses step-1 bake across steps).\n"
            "lvl2: also cache dequantized float weights per sampling run (more speed, more memory).\n"
            "Only applies to GGUF exec mode 'dequant_forward' and is capped by a heuristic VRAM/RAM budget.",
        )
        row = self._add_choice_combo(
            body,
            row,
            label="LoRA online math (requires API restart):",
            var=self._var_lora_online_math,
            choices=["weight_merge"],
            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
            width=16,
            out_combo="_lora_math_combo",
        )
        row = add_help(
            body,
            row,
            "weight_merge: current online behavior (materializes patched weights per-forward).\n"
            "activation math is reserved for future packed-kernel LoRA support (not exposed in this build).",
        )

        row = self._add_entry(
            body,
            row,
            label="PyTorch CUDA alloc conf (requires API restart):",
            var=self._var_pytorch_alloc_conf,
            width=56,
            on_change=self._on_alloc_conf_changed,
        )
        row = add_help(body, row, "Env var: PYTORCH_CUDA_ALLOC_CONF\n" f"Default: {DEFAULT_PYTORCH_CUDA_ALLOC_CONF}")
        _ = add_help(
            body,
            row,
            "Runtime settings (dtype/attention/cache/offload) are configured via the Web UI.\n"
            "This launcher focuses on bootstrap + global runtime knobs that must exist before the API starts.",
        )

        self.frame = frame
        self.reload()
        return frame

    def reload(self) -> None:
        env = self._controller.store.env

        def _get(key: str, default: str) -> str:
            raw = str(env.get(key, default) or "").strip().lower()
            return raw if raw else default

        self._var_core_device.set(_get("CODEX_CORE_DEVICE", "auto"))
        self._var_te_device.set(_get("CODEX_TE_DEVICE", "auto"))
        self._var_vae_device.set(_get("CODEX_VAE_DEVICE", "auto"))

        self._var_lora_apply_mode.set(_get("CODEX_LORA_APPLY_MODE", "merge"))
        self._var_gguf_exec.set(_get("CODEX_GGUF_EXEC", "dequant_forward"))
        self._var_gguf_dequant_cache.set(_get("CODEX_GGUF_DEQUANT_CACHE", "off"))
        self._var_lora_online_math.set(_get("CODEX_LORA_ONLINE_MATH", "weight_merge"))

        alloc = str(self._controller.store.build_env().get("PYTORCH_CUDA_ALLOC_CONF") or DEFAULT_PYTORCH_CUDA_ALLOC_CONF).strip()
        self._var_pytorch_alloc_conf.set(alloc)

        self._sync_runtime_deps(mark_changed=False)

    # ------------------------------------------------------------------ widgets

    def _add_choice_combo(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        choices: List[str],
        on_change: Callable[[], None],
        width: int = 18,
        out_combo: str | None = None,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        combo = ttk.Combobox(parent, textvariable=var, values=choices, state="readonly", width=width)
        combo.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)
        combo.bind("<<ComboboxSelected>>", lambda _e: on_change())
        if out_combo:
            setattr(self, out_combo, combo)
            if out_combo == "_lora_math_combo":
                self._lora_math_combo = combo
            elif out_combo == "_gguf_dequant_cache_combo":
                self._gguf_dequant_cache_combo = combo
        return row + 1

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        width: int,
        on_change: Callable[[], None],
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)
        entry.bind("<KeyRelease>", lambda _e: on_change())
        return row + 1

    # ------------------------------------------------------------------ env helpers

    def _set_env_lower(self, key: str, value: str) -> None:
        self._controller.store.env[key] = str(value).strip().lower()
        self._mark_changed()

    def _on_alloc_conf_changed(self) -> None:
        key = "PYTORCH_CUDA_ALLOC_CONF"
        value = str(self._var_pytorch_alloc_conf.get() or "").strip()
        if not value:
            try:
                del self._controller.store.env[key]
            except KeyError:
                pass
        else:
            self._controller.store.env[key] = value
        self._mark_changed()

    # ------------------------------------------------------------------ dependency logic

    def _sync_runtime_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        env["CODEX_GGUF_EXEC"] = str(self._var_gguf_exec.get() or "").strip().lower() or "dequant_forward"
        env["CODEX_GGUF_DEQUANT_CACHE"] = str(self._var_gguf_dequant_cache.get() or "").strip().lower() or "off"
        env["CODEX_LORA_APPLY_MODE"] = str(self._var_lora_apply_mode.get() or "").strip().lower() or "merge"
        env["CODEX_LORA_ONLINE_MATH"] = str(self._var_lora_online_math.get() or "").strip().lower() or "weight_merge"
        try:
            gguf, gguf_cache, lora_apply, lora_math = normalize_gguf_lora_env(env)
        except SettingValidationError as exc:
            env["CODEX_GGUF_EXEC"] = "dequant_forward"
            env["CODEX_GGUF_DEQUANT_CACHE"] = "off"
            env["CODEX_LORA_APPLY_MODE"] = "merge"
            env["CODEX_LORA_ONLINE_MATH"] = "weight_merge"
            gguf, gguf_cache, lora_apply, lora_math = normalize_gguf_lora_env(env)
            messagebox.showerror("Invalid runtime setting", str(exc))
            mark_changed = True

        self._var_gguf_exec.set(gguf)
        self._var_gguf_dequant_cache.set(gguf_cache)
        self._var_lora_apply_mode.set(lora_apply)
        self._var_lora_online_math.set(lora_math)

        if self._gguf_dequant_cache_combo is not None:
            self._gguf_dequant_cache_combo.configure(state="readonly" if gguf == "dequant_forward" else "disabled")

        if self._lora_math_combo is not None:
            self._lora_math_combo.configure(state="readonly" if lora_apply == "online" else "disabled")

        if mark_changed:
            self._mark_changed()
