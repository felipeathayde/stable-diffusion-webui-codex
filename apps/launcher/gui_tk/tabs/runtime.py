"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime settings tab for the Tk launcher.
Edits bootstrap-critical device defaults and global runtime/task knobs that must exist before the API starts (devices, GGUF/LoRA, task single-flight,
task cancel default mode, task SSE buffer caps, upscaler safeweights).

Symbols (top-level; keep in sync; no ghosts):
- `RuntimeTab` (class): Runtime settings tab (device defaults + attention mode + GGUF/LoRA + `PYTORCH_ALLOC_CONF`/cuda-malloc toggles).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List

from apps.launcher.profiles import (
    CODEX_CUDA_MALLOC_KEY,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
)
from apps.launcher.settings import (
    BoolSetting,
    DEVICE_CHOICES,
    IntSetting,
    WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES,
    SettingValidationError,
    TASK_CANCEL_DEFAULT_MODE_CHOICES,
    TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
    TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
    attention_mode_to_backend_policy,
    backend_policy_to_attention_mode,
    normalize_gguf_lora_env,
    normalize_attention_env,
    normalize_task_runtime_env,
)

from ..controller import LauncherController
from ..widgets import ScrollableFrame, add_help, add_section_header


_ATTENTION_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("SDPA - Auto", "sdpa_auto"),
    ("SDPA - Flash", "sdpa_flash"),
    ("SDPA - Mem Efficient", "sdpa_mem_efficient"),
    ("SDPA - Math", "sdpa_math"),
    ("xFormers", "xformers"),
    ("Split (Chunked)", "split"),
    ("Quad (Sub-Quadratic)", "quad"),
)
_ATTENTION_LABEL_TO_MODE = {label: mode for label, mode in _ATTENTION_MODE_OPTIONS}
_ATTENTION_MODE_TO_LABEL = {mode: label for label, mode in _ATTENTION_MODE_OPTIONS}


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
        self._var_attention_mode = tk.StringVar()

        self._var_lora_apply_mode = tk.StringVar()
        self._var_gguf_exec = tk.StringVar()
        self._var_gguf_dequant_cache = tk.StringVar()
        self._var_gguf_dequant_cache_ratio = tk.StringVar()
        self._var_wan_chunk_buffer_mode = tk.StringVar()
        self._var_lora_online_math = tk.StringVar()
        self._var_pytorch_alloc_conf = tk.StringVar()
        self._var_default_alloc_conf_enabled = tk.BooleanVar()
        self._var_cuda_malloc = tk.BooleanVar()

        self._var_single_flight = tk.BooleanVar()
        self._var_safeweights = tk.BooleanVar()
        self._var_task_cancel_default_mode = tk.StringVar()
        self._var_task_buffer_max_events = tk.StringVar()
        self._var_task_buffer_max_mb = tk.StringVar()

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
        row = self._add_choice_combo(
            body,
            row,
            label="Attention mode (requires API restart):",
            var=self._var_attention_mode,
            choices=[label for label, _mode in _ATTENTION_MODE_OPTIONS],
            on_change=self._on_attention_mode_changed,
            width=20,
        )
        row = add_help(
            body,
            row,
            "sdpa_auto lets PyTorch pick the best SDPA kernel.\n"
            "sdpa_flash/sdpa_mem_efficient/sdpa_math force a specific SDPA policy.\n"
            "xformers/split/quad select non-SDPA attention backends.",
        )
        row = add_help(
            body,
            row,
            "These values are passed as backend CLI flags (`--core-device/--te-device/--vae-device`).\n"
            "Attention mode is passed via `--attention-backend` + optional `--attention-sdpa-policy`.\n"
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
        row = self._add_entry(
            body,
            row,
            label="GGUF dequant cache ratio (optional):",
            var=self._var_gguf_dequant_cache_ratio,
            width=12,
            on_change=self._on_gguf_dequant_cache_ratio_changed,
        )
        row = add_help(
            body,
            row,
            "Env var: CODEX_GGUF_DEQUANT_CACHE_RATIO (float, >0 and <=1).\n"
            "Used only when GGUF dequant cache is enabled and no explicit LIMIT_MB is set.\n"
            "Example: 0.30 reserves ~30% of free VRAM/RAM at sampling start.",
        )
        row = self._add_choice_combo(
            body,
            row,
            label="WAN img2vid chunk buffer mode (requires API restart):",
            var=self._var_wan_chunk_buffer_mode,
            choices=list(WAN22_IMG2VID_CHUNK_BUFFER_MODE_CHOICES),
            on_change=lambda: self._sync_runtime_deps(mark_changed=True),
            width=10,
        )
        row = add_help(
            body,
            row,
            "Env var: CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE\n"
            "hybrid: auto-select RAM or RAM+disk by chunk memory estimate.\n"
            "ram: keep chunk buffers only in RAM.\n"
            "ram+hd: spool chunk buffers to RAM+disk (bounded RAM).",
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
        row = self._add_check(
            body,
            row,
            label="Apply default PyTorch alloc conf when unset (requires API restart):",
            var=self._var_default_alloc_conf_enabled,
            on_change=self._on_default_alloc_conf_toggle_changed,
        )
        row = add_help(
            body,
            row,
            "Env var: PYTORCH_ALLOC_CONF\n"
            f"Default value: {DEFAULT_PYTORCH_CUDA_ALLOC_CONF}\n"
            f"Default toggle env: {ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY}",
        )
        row = self._add_check(
            body,
            row,
            label="Enable cudaMallocAsync backend (requires API restart):",
            var=self._var_cuda_malloc,
            on_change=self._on_cuda_malloc_changed,
        )
        row = add_help(
            body,
            row,
            f"Env var: {CODEX_CUDA_MALLOC_KEY}\n"
            "When enabled, launcher forwards backend flag '--cuda-malloc'.",
        )
        _ = add_help(
            body,
            row,
            "Runtime settings (dtype/cache/offload) are configured via the Web UI.\n"
            "This launcher focuses on bootstrap + global runtime knobs that must exist before the API starts.",
        )

        row = add_section_header(body, row, "Tasks / Safety")
        row = self._add_check(
            body,
            row,
            label="Single-flight inference (requires API restart):",
            var=self._var_single_flight,
            on_change=lambda: self._sync_task_deps(mark_changed=True),
        )
        row = add_help(
            body,
            row,
            "Env var: CODEX_SINGLE_FLIGHT\n"
            "When enabled (default), GPU-heavy tasks (generation/video/upscale/SUPIR) are serialized to avoid global-state races.",
        )
        row = self._add_choice_combo(
            body,
            row,
            label="Task cancel default mode (requires API restart):",
            var=self._var_task_cancel_default_mode,
            choices=list(TASK_CANCEL_DEFAULT_MODE_CHOICES),
            on_change=lambda: self._sync_task_deps(mark_changed=True),
            width=14,
        )
        row = add_help(
            body,
            row,
            "Env var: CODEX_TASK_CANCEL_DEFAULT_MODE\n"
            "immediate: cancels in-flight generation now.\n"
            "after_current: finish current image job (1st pass + hires/decode/cleanup) before stopping.",
        )
        row = self._add_entry_commit_int(
            body,
            row,
            label="Task SSE buffer max events (requires API restart):",
            var=self._var_task_buffer_max_events,
            key="CODEX_TASK_EVENT_BUFFER_MAX_EVENTS",
            default=TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT,
            minimum=1,
        )
        row = self._add_entry_commit_int(
            body,
            row,
            label="Task SSE buffer max MB (requires API restart):",
            var=self._var_task_buffer_max_mb,
            key="CODEX_TASK_EVENT_BUFFER_MAX_MB",
            default=TASK_EVENT_BUFFER_MAX_MB_DEFAULT,
            minimum=1,
        )
        row = add_help(
            body,
            row,
            "These caps bound in-memory task replay buffers (per task) used for reconnect/resume.\n"
            "Env vars: CODEX_TASK_EVENT_BUFFER_MAX_EVENTS, CODEX_TASK_EVENT_BUFFER_MAX_MB",
        )
        row = self._add_check(
            body,
            row,
            label="Upscalers safeweights mode (requires API restart):",
            var=self._var_safeweights,
            on_change=lambda: self._sync_task_deps(mark_changed=True),
        )
        _ = add_help(
            body,
            row,
            "Env var: CODEX_SAFE_WEIGHTS\n"
            "When enabled, upscaler weights must be .safetensors (blocks .pt/.pth at discovery, download, and load-time).",
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
        try:
            attn_backend, attn_sdpa_policy = normalize_attention_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_ATTENTION_BACKEND"] = "pytorch"
            self._controller.store.env["CODEX_ATTENTION_SDPA_POLICY"] = "mem_efficient"
            attn_backend, attn_sdpa_policy = normalize_attention_env(self._controller.store.env)
            messagebox.showerror("Invalid runtime setting", str(exc))
        attention_mode = backend_policy_to_attention_mode(attn_backend, attn_sdpa_policy)
        self._var_attention_mode.set(_ATTENTION_MODE_TO_LABEL.get(attention_mode, "SDPA - Mem Efficient"))

        self._var_lora_apply_mode.set(_get("CODEX_LORA_APPLY_MODE", "merge"))
        self._var_gguf_exec.set(_get("CODEX_GGUF_EXEC", "dequant_forward"))
        self._var_gguf_dequant_cache.set(_get("CODEX_GGUF_DEQUANT_CACHE", "off"))
        self._var_gguf_dequant_cache_ratio.set(str(env.get("CODEX_GGUF_DEQUANT_CACHE_RATIO", "") or "").strip())
        self._var_wan_chunk_buffer_mode.set(_get("CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE", "hybrid"))
        self._var_lora_online_math.set(_get("CODEX_LORA_ONLINE_MATH", "weight_merge"))
        try:
            default_alloc_enabled = BoolSetting(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, default=True).get(env)
        except SettingValidationError:
            default_alloc_enabled = True
            BoolSetting(ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY, default=True).set(env, default_alloc_enabled)
        self._var_default_alloc_conf_enabled.set(bool(default_alloc_enabled))

        try:
            cuda_malloc_enabled = BoolSetting(CODEX_CUDA_MALLOC_KEY, default=False).get(env)
        except SettingValidationError:
            cuda_malloc_enabled = False
            BoolSetting(CODEX_CUDA_MALLOC_KEY, default=False).set(env, cuda_malloc_enabled)
        self._var_cuda_malloc.set(bool(cuda_malloc_enabled))

        alloc = str(env.get("PYTORCH_ALLOC_CONF", "") or "").strip()
        if not alloc and default_alloc_enabled:
            alloc = DEFAULT_PYTORCH_CUDA_ALLOC_CONF
        self._var_pytorch_alloc_conf.set(alloc)

        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            self._controller.store.env["CODEX_SINGLE_FLIGHT"] = "1"
            self._controller.store.env["CODEX_SAFE_WEIGHTS"] = "0"
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT)
            self._controller.store.env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(TASK_EVENT_BUFFER_MAX_MB_DEFAULT)
            self._controller.store.env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = "immediate"
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))

        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))

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

    def _add_check(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.BooleanVar,
        on_change: Callable[[], None],
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        cb = ttk.Checkbutton(parent, variable=var, command=on_change)
        cb.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)
        return row + 1

    def _add_entry_commit_int(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        key: str,
        default: int,
        minimum: int,
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        entry = ttk.Entry(parent, textvariable=var, width=12)
        entry.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)

        def commit() -> None:
            env = self._controller.store.env
            setting = IntSetting(key, default=default, minimum=minimum)
            try:
                value = setting.parse(str(var.get() or ""))
            except SettingValidationError as exc:
                messagebox.showerror("Invalid task setting", str(exc))
                value = int(default)
            setting.set(env, value)
            var.set(str(value))
            self._mark_changed()

        entry.bind("<FocusOut>", lambda _e: commit())
        entry.bind("<Return>", lambda _e: commit())
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

    def _on_attention_mode_changed(self) -> None:
        env = self._controller.store.env
        try:
            raw_mode = str(self._var_attention_mode.get() or "").strip()
            attention_mode = _ATTENTION_LABEL_TO_MODE.get(raw_mode, raw_mode)
            backend, sdpa_policy = attention_mode_to_backend_policy(attention_mode)
        except SettingValidationError as exc:
            messagebox.showerror("Invalid runtime setting", str(exc))
            backend, sdpa_policy = "pytorch", "mem_efficient"
        env["CODEX_ATTENTION_BACKEND"] = backend
        env["CODEX_ATTENTION_SDPA_POLICY"] = sdpa_policy
        attention_mode = backend_policy_to_attention_mode(backend, sdpa_policy)
        self._var_attention_mode.set(_ATTENTION_MODE_TO_LABEL.get(attention_mode, "SDPA - Mem Efficient"))
        self._mark_changed()

    def _on_alloc_conf_changed(self) -> None:
        key = "PYTORCH_ALLOC_CONF"
        value = str(self._var_pytorch_alloc_conf.get() or "").strip()
        if not value:
            try:
                del self._controller.store.env[key]
            except KeyError:
                pass
        else:
            self._controller.store.env[key] = value
        self._mark_changed()

    def _on_default_alloc_conf_toggle_changed(self) -> None:
        env = self._controller.store.env
        enabled = bool(self._var_default_alloc_conf_enabled.get())
        BoolSetting(
            ENABLE_DEFAULT_PYTORCH_CUDA_ALLOC_CONF_KEY,
            default=True,
        ).set(env, enabled)
        if not enabled and "PYTORCH_ALLOC_CONF" not in env:
            self._var_pytorch_alloc_conf.set("")
        if enabled and "PYTORCH_ALLOC_CONF" not in env and not str(self._var_pytorch_alloc_conf.get() or "").strip():
            self._var_pytorch_alloc_conf.set(DEFAULT_PYTORCH_CUDA_ALLOC_CONF)
        self._mark_changed()

    def _on_cuda_malloc_changed(self) -> None:
        BoolSetting(
            CODEX_CUDA_MALLOC_KEY,
            default=False,
        ).set(self._controller.store.env, bool(self._var_cuda_malloc.get()))
        self._mark_changed()

    def _on_gguf_dequant_cache_ratio_changed(self) -> None:
        key = "CODEX_GGUF_DEQUANT_CACHE_RATIO"
        value = str(self._var_gguf_dequant_cache_ratio.get() or "").strip()
        if not value:
            self._controller.store.env.pop(key, None)
        else:
            self._controller.store.env[key] = value
        self._mark_changed()

    def _sync_task_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env

        BoolSetting("CODEX_SINGLE_FLIGHT", default=True).set(env, bool(self._var_single_flight.get()))
        BoolSetting("CODEX_SAFE_WEIGHTS", default=False).set(env, bool(self._var_safeweights.get()))
        env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = str(self._var_task_cancel_default_mode.get() or "").strip().lower()
        env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(self._var_task_buffer_max_events.get() or "").strip()
        env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(self._var_task_buffer_max_mb.get() or "").strip()

        try:
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
        except SettingValidationError as exc:
            env["CODEX_SINGLE_FLIGHT"] = "1"
            env["CODEX_SAFE_WEIGHTS"] = "0"
            env["CODEX_TASK_EVENT_BUFFER_MAX_EVENTS"] = str(TASK_EVENT_BUFFER_MAX_EVENTS_DEFAULT)
            env["CODEX_TASK_EVENT_BUFFER_MAX_MB"] = str(TASK_EVENT_BUFFER_MAX_MB_DEFAULT)
            env["CODEX_TASK_CANCEL_DEFAULT_MODE"] = "immediate"
            single_flight, safeweights, max_events, max_mb, cancel_default_mode = normalize_task_runtime_env(env)
            messagebox.showerror("Invalid task setting", str(exc))
            mark_changed = True

        self._var_single_flight.set(bool(single_flight))
        self._var_safeweights.set(bool(safeweights))
        self._var_task_cancel_default_mode.set(str(cancel_default_mode))
        self._var_task_buffer_max_events.set(str(int(max_events)))
        self._var_task_buffer_max_mb.set(str(int(max_mb)))

        if mark_changed:
            self._mark_changed()

    # ------------------------------------------------------------------ dependency logic

    def _sync_runtime_deps(self, *, mark_changed: bool) -> None:
        env = self._controller.store.env
        env["CODEX_GGUF_EXEC"] = str(self._var_gguf_exec.get() or "").strip().lower() or "dequant_forward"
        env["CODEX_GGUF_DEQUANT_CACHE"] = str(self._var_gguf_dequant_cache.get() or "").strip().lower() or "off"
        env["CODEX_LORA_APPLY_MODE"] = str(self._var_lora_apply_mode.get() or "").strip().lower() or "merge"
        env["CODEX_LORA_ONLINE_MATH"] = str(self._var_lora_online_math.get() or "").strip().lower() or "weight_merge"
        env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = (
            str(self._var_wan_chunk_buffer_mode.get() or "").strip().lower() or "hybrid"
        )
        try:
            gguf, gguf_cache, lora_apply, lora_math, chunk_buffer_mode = normalize_gguf_lora_env(env)
        except SettingValidationError as exc:
            env["CODEX_GGUF_EXEC"] = "dequant_forward"
            env["CODEX_GGUF_DEQUANT_CACHE"] = "off"
            env.pop("CODEX_GGUF_DEQUANT_CACHE_RATIO", None)
            env["CODEX_LORA_APPLY_MODE"] = "merge"
            env["CODEX_LORA_ONLINE_MATH"] = "weight_merge"
            env["CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE"] = "hybrid"
            gguf, gguf_cache, lora_apply, lora_math, chunk_buffer_mode = normalize_gguf_lora_env(env)
            messagebox.showerror("Invalid runtime setting", str(exc))
            mark_changed = True

        self._var_gguf_exec.set(gguf)
        self._var_gguf_dequant_cache.set(gguf_cache)
        self._var_lora_apply_mode.set(lora_apply)
        self._var_gguf_dequant_cache_ratio.set(str(env.get("CODEX_GGUF_DEQUANT_CACHE_RATIO", "") or "").strip())
        self._var_lora_online_math.set(lora_math)
        self._var_wan_chunk_buffer_mode.set(chunk_buffer_mode)

        if self._gguf_dequant_cache_combo is not None:
            self._gguf_dequant_cache_combo.configure(state="readonly" if gguf == "dequant_forward" else "disabled")

        if self._lora_math_combo is not None:
            self._lora_math_combo.configure(state="readonly" if lora_apply == "online" else "disabled")

        if mark_changed:
            self._mark_changed()
