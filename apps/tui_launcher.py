"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: BIOS-style curses TUI (stdlib-only) for managing Codex services and launcher options.
Settings persist under `.sangoi/launcher/` and the UI controls API/UI service processes plus runtime flags (including WAN controls).

Symbols (top-level; keep in sync; no ghosts):
- `BIOSApp` (class): Curses app implementing the tabbed launcher UI (render loop + key handling); includes nested helpers for panes,
  item lists per tab (`_items_main/_items_runtime/_items_wan/_items_debug/_items_logging`), popup selection, and applying changes to the profile/env.
- `main` (function): Curses entrypoint (used with `curses.wrapper`); creates `BIOSApp` and runs the UI loop.
"""
from __future__ import annotations

import curses
import locale
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Root: CODEX_ROOT must be set by .bat launcher
_codex_root = os.environ.get("CODEX_ROOT")
if not _codex_root:
    raise EnvironmentError("CODEX_ROOT not set. Use run-tui.bat to launch.")
_root_path = Path(_codex_root)
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

from apps.backend.infra.config.repo_root import get_repo_root

CODEX_ROOT = get_repo_root()
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from apps.backend.infra.config.args import TRACE_DEBUG_DEFAULT
from apps.launcher import (
    CodexLaunchCheck,
    CodexLogBuffer,
    CodexServiceHandle,
    LauncherProfileStore,
    ServiceStatus,
    default_services,
    run_launch_checks,
)
class BIOSApp:
    TABS = ["Main", "Runtime", "WAN", "DEBUG", "Logging", "Logs", "Exit"]

    def __init__(self, stdscr) -> None:
        self.stdscr = stdscr
        self.log_buffer = CodexLogBuffer(capacity=4000)
        self.services: Dict[str, CodexServiceHandle] = default_services(log_buffer=self.log_buffer)
        self.store = LauncherProfileStore.load()
        self.meta = self.store.meta
        self.env = self.store.env
        self.tab_index: int = max(0, min(len(self.TABS) - 1, self.meta.tab_index))
        self.sel_index: int = 0
        self.log_scroll: int = 0
        self.message: str = ""
        self.popup_active: bool = False
        self.popup_selection: Optional[Tuple[str, List[str], int, str]] = None  # (title, options, index, action)
        self.launch_checks: List[CodexLaunchCheck] = run_launch_checks()

        # colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLUE)    # top/bottom bar
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)    # active tab
        curses.init_pair(3, curses.COLOR_WHITE, -1)                   # normal text
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)   # selection
        curses.init_pair(5, curses.COLOR_YELLOW, -1)                  # hint
        curses.init_pair(6, curses.COLOR_CYAN, -1)                    # label
        self.C_BAR = curses.color_pair(1)
        self.C_TAB = curses.color_pair(2)
        self.C_SEL = curses.color_pair(4)
        self.C_NRM = curses.color_pair(3)
        self.C_LBL = curses.color_pair(6)

    # --------------- rendering helpers ---------------------------------
    def _prompt(self, prompt: str) -> Optional[str]:
        try:
            h, w = self.stdscr.getmaxyx()
            y = h - 2
            msg = (prompt + " " * w)[:w-1]
            curses.echo()
            try:
                curses.curs_set(1)
            except Exception:
                pass
            self.stdscr.addstr(y, 1, msg)
            self.stdscr.refresh()
            win = curses.newwin(1, max(10, w - len(prompt) - 2), y, len(prompt) + 2)
            val = win.getstr().decode(errors='ignore')
            return val.strip()
        except Exception:
            return None
        finally:
            try:
                curses.noecho()
                curses.curs_set(0)
            except Exception:
                pass
    def _draw_bars(self, h: int, w: int) -> None:
        title = " PhoenixBIOS-like Setup Utility "
        try:
            self.stdscr.attron(self.C_BAR)
            # Top bar
            self.stdscr.addstr(0, 0, (" " * w)[:w])
            self.stdscr.addstr(0, max(0, (w - len(title)) // 2), title[:max(0, w)])
            self.stdscr.attroff(self.C_BAR)
        except Exception:
            pass

        # tabs line
        x = 0
        for i, name in enumerate(self.TABS):
            label = f" {name} "
            attr = self.C_TAB if i == self.tab_index else self.C_NRM
            self.stdscr.attron(attr)
            self.stdscr.addstr(1, x, label)
            self.stdscr.attroff(attr)
            x += len(label) + 1

        # bottom bar
        keys = "F1 Help   Esc Exit   Enter Select   +/- Change   F2/F3 API Start/Kill   F4/F5 UI Start/Kill   F10 Save and Exit"
        try:
            self.stdscr.attron(self.C_BAR)
            y = max(0, h - 1)
            self.stdscr.addstr(y, 0, (keys + " " * w)[:w])
            self.stdscr.attroff(self.C_BAR)
        except Exception:
            pass

    def _pane_geometry(self) -> Tuple[int, int, int, int]:
        h, w = self.stdscr.getmaxyx()
        top = 2
        left_w = max(52, w * 2 // 3)
        right_w = max(20, w - left_w - 1)
        height = max(3, h - top - 2)
        return top, left_w, right_w, height

    def _draw_help(self, top: int, left_w: int, right_w: int, height: int, help_lines: List[str]) -> None:
        x = left_w + 1
        try:
            self.stdscr.attron(self.C_LBL)
            self.stdscr.addstr(top, x, "Item Specific Help")
            self.stdscr.attroff(self.C_LBL)
        except Exception:
            pass
        for i in range(1, height):
            line = help_lines[i - 1] if i - 1 < len(help_lines) else ""
            try:
                self.stdscr.addstr(top + i, x, (line + " " * right_w)[:right_w])
            except Exception:
                pass

    def _persist(self) -> None:
        self.meta.tab_index = max(0, self.tab_index)
        self.store.save()

    # --------------- content per tab -----------------------------------
    def _ensure_log_file(self) -> str:
        logs_dir = CODEX_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"codex-{stamp}.log"
        return str(log_path)


    def _items_main(self) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        for name in ("API", "UI"):
            svc = self.services[name]
            status = svc.status.value
            pid = svc.pid or 0
            up = "-"
            if svc.started_at and svc.status == ServiceStatus.RUNNING:
                up = f"{int(time.time() - svc.started_at)}s"
            out.append((f"{name} Status", f"{status} (pid={pid}, up={up})", "status"))
            out.append((f"Start {name}", "", "start"))
            out.append((f"Restart {name}", "", "restart"))
            out.append((f"Kill {name}", "", "kill"))
        return out

    def _items_runtime(self) -> List[Tuple[str, str, str]]:
        env = self.env
        ext = "Enabled" if self.meta.external_terminal else "Disabled"
        pol = self.meta.sdpa_policy
        attn_backend = env.get("CODEX_ATTENTION_BACKEND", "torch-sdpa")
        attn_chunk = env.get("CODEX_ATTN_CHUNK_SIZE", "0")
        gguf_pol = env.get("CODEX_GGUF_CACHE_POLICY", "none")
        gguf_lim = env.get("CODEX_GGUF_CACHE_LIMIT_MB", "0")
        _unet_dtype_options = {"auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"}
        _te_dtype_options = {"auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"}
        _vae_dtype_options = {"auto", "fp16", "bf16", "fp32"}

        unet_dtype = (env.get("CODEX_DIFFUSION_DTYPE", "auto") or "auto").strip().lower()
        if unet_dtype not in _unet_dtype_options:
            unet_dtype = "auto"
        env["CODEX_DIFFUSION_DTYPE"] = unet_dtype
        diff_device_raw = env.get("CODEX_DIFFUSION_DEVICE", "").strip().lower()
        if diff_device_raw == "gpu":
            diff_device_raw = "cuda"
            env["CODEX_DIFFUSION_DEVICE"] = "cuda"
        if diff_device_raw == "cpu" and unet_dtype != "fp32":
            unet_dtype = "fp32"
            env["CODEX_DIFFUSION_DTYPE"] = "fp32"
        te_dtype = (env.get("CODEX_TE_DTYPE", "auto") or "auto").strip().lower()
        if te_dtype not in _te_dtype_options:
            te_dtype = "auto"
        env["CODEX_TE_DTYPE"] = te_dtype
        te_device_raw = env.get("CODEX_TE_DEVICE", "").strip().lower()
        if te_device_raw == "gpu":
            te_device_raw = "cuda"
            env["CODEX_TE_DEVICE"] = "cuda"
        if te_device_raw == "cpu" and te_dtype != "fp32":
            te_dtype = "fp32"
            env["CODEX_TE_DTYPE"] = "fp32"
        vae_dtype = (env.get("CODEX_VAE_DTYPE", "auto") or "auto").strip().lower()
        if vae_dtype not in _vae_dtype_options:
            vae_dtype = "auto"
        env["CODEX_VAE_DTYPE"] = vae_dtype
        vae_device_raw = env.get("CODEX_VAE_DEVICE", "").strip().lower()
        if vae_device_raw == "gpu":
            vae_device_raw = "cuda"
            env["CODEX_VAE_DEVICE"] = "cuda"
        if vae_device_raw == "cpu" and vae_dtype != "fp32":
            vae_dtype = "fp32"
            env["CODEX_VAE_DTYPE"] = "fp32"

        def _device_label(raw: str) -> str:
            mapping = {
                "cpu": "CPU",
                "cuda": "GPU",
                "mps": "MPS",
                "xpu": "XPU",
                "directml": "DirectML",
                "dml": "DirectML",
            }
            return mapping.get(raw, "Auto")

        diff_dev_label = _device_label(diff_device_raw)
        te_dev_label = _device_label(te_device_raw)
        vae_dev_label = _device_label(vae_device_raw)
        swap_pol = env.get("CODEX_SWAP_POLICY", "cpu")
        swap_mth = env.get("CODEX_SWAP_METHOD", "blocked")
        smart_offload = env.get("CODEX_SMART_OFFLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
        pin_shared = env.get("CODEX_PIN_SHARED_MEMORY", "0").strip().lower() in {"1", "true", "yes", "on"}
        return [
            ("Services in new terminal", f"[{ext}]", "toggle_extterm"),
            ("SDPA Policy", f"[{pol}]", "cycle_sdpa"),
            ("Attention Backend", f"[{attn_backend}]", "cycle_attn_backend"),
            ("Attn Chunk Size", f"[{attn_chunk}]", "edit_attn_chunk"),
            ("GGUF Cache Policy", f"[{gguf_pol}]", "cycle_gguf_pol"),
            ("GGUF Cache Limit (MB)", f"[{gguf_lim}]", "edit_gguf_lim"),
            ("Diffusion Device", f"[{diff_dev_label}]", "select_diff_device"),
            ("DiT/UNet DType", f"[{unet_dtype}]", "cycle_unet_dtype"),
            ("Text Encoder Device", f"[{te_dev_label}]", "select_te_device"),
            ("Text Encoder DType", f"[{te_dtype}]", "cycle_te_dtype"),
            ("VAE DType", f"[{vae_dtype}]", "cycle_vae_dtype"),
            ("VAE Device", f"[{vae_dev_label}]", "select_vae_dev"),
            ("Swap Policy", f"[{swap_pol}]", "cycle_swap_pol"),
            ("Swap Method", f"[{swap_mth}]", "cycle_swap_mth"),
            ("Smart Offload", f"[{'Enabled' if smart_offload else 'Disabled'}]", "toggle_smart_offload"),
            ("Pin Shared Memory", f"[{'Enabled' if pin_shared else 'Disabled'}]", "toggle_pin_shared"),
        ]

    def _items_wan(self) -> List[Tuple[str, str, str]]:
        env = self.env
        order = env.get("WAN_I2V_ORDER", "lat_first")
        offload_lvl = env.get("WAN_GGUF_OFFLOAD_LEVEL", "3")
        te_impl = env.get("WAN_TE_IMPL", "hf")
        te_fp8_label = "Enabled" if te_impl == "cuda_fp8" else "Disabled"
        sdpa_dbg = env.get("WAN_SDPA_DEBUG", "0").lower() in ("1", "true", "yes", "on")
        dbg_hi = env.get("WAN_I2V_DEBUG_HI_DECODE", "0").lower() in ("1", "true", "yes", "on")
        lat_stats = env.get("WAN_I2V_LAT_STATS", "0").lower() in ("1", "true", "yes", "on")
        conv32 = env.get("WAN_I2V_CONV32", "0").lower() in ("1", "true", "yes", "on")
        sanitize = env.get("WAN_I2V_DEBUG_SANITIZE_TOKENS", "0").lower() in ("1", "true", "yes", "on")
        clamp = env.get("WAN_I2V_DEBUG_CLAMP", "")
        clamp_label = clamp if clamp else "Disabled"
        return [
            ("I2V Concat Order", f"[{order}]", "cycle_i2v_order"),
            ("GGUF Offload Level", f"[{offload_lvl}]", "cycle_offload_lvl"),
            ("Use cuda fp8 (TE)", f"[{te_fp8_label}]", "toggle_te_fp8"),
            ("WAN_SDPA_DEBUG", f"[{'Enabled' if sdpa_dbg else 'Disabled'}]", "toggle_sdpa_debug"),
            ("WAN_I2V_DEBUG_HI_DECODE", f"[{'Enabled' if dbg_hi else 'Disabled'}]", "toggle_hi_decode_debug"),
            ("WAN_I2V_LAT_STATS", f"[{'Enabled' if lat_stats else 'Disabled'}]", "toggle_lat_stats"),
            ("WAN_I2V_CONV32", f"[{'Enabled' if conv32 else 'Disabled'}]", "toggle_conv32"),
            ("WAN_I2V_DEBUG_SANITIZE_TOKENS", f"[{'Enabled' if sanitize else 'Disabled'}]", "toggle_sanitize_tokens"),
            ("WAN_I2V_DEBUG_CLAMP", f"[{clamp_label}]", "edit_debug_clamp"),
        ]

    def _items_debug(self) -> List[Tuple[str, str, str]]:
        env = self.env

        def _enabled(key: str) -> bool:
            return env.get(key, "0").strip().lower() in {"1", "true", "yes", "on"}

        cond = _enabled("CODEX_DEBUG_COND")
        sampler = _enabled("CODEX_LOG_SAMPLER")
        cfg_delta = _enabled("CODEX_LOG_CFG_DELTA")
        sigmas = _enabled("CODEX_LOG_SIGMAS")
        force_native = _enabled("CODEX_SAMPLER_FORCE_NATIVE")
        trace = _enabled("CODEX_TRACE_DEBUG")
        pipeline = _enabled("CODEX_PIPELINE_DEBUG")
        dump_latents = _enabled("CODEX_DUMP_LATENTS")
        dump_path = env.get("CODEX_DUMP_LATENTS_PATH", "").strip()
        if not dump_path:
            dump_path = "<logs/diagnostics>"
        cfg_delta_n = env.get("CODEX_LOG_CFG_DELTA_N", "2").strip() or "2"
        trace_max = env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC", str(TRACE_DEBUG_DEFAULT))
        return [
            ("Conditioning Debug", f"[{'Enabled' if cond else 'Disabled'}]", "toggle_cond_debug"),
            ("Sampler Verbose Logs", f"[{'Enabled' if sampler else 'Disabled'}]", "toggle_sampler_logs"),
            ("CFG Delta Logs", f"[{'Enabled' if cfg_delta else 'Disabled'}]", "toggle_cfg_delta_logs"),
            ("CFG Delta Steps (N)", f"[{cfg_delta_n}]", "edit_cfg_delta_n"),
            ("Sigma Ladder Logs", f"[{'Enabled' if sigmas else 'Disabled'}]", "toggle_sigma_logs"),
            ("Force Native Sampler", f"[{'Enabled' if force_native else 'Disabled'}]", "toggle_force_native_sampler"),
            ("Trace Debug", f"[{'ON' if trace else 'OFF'}]", "toggle_trace_debug"),
            ("Trace Max Per Func", f"[{trace_max}]", "edit_trace_max"),
            ("Pipeline Debug", f"[{'ON' if pipeline else 'OFF'}]", "toggle_pipeline_debug"),
            ("Dump Latents", f"[{'Enabled' if dump_latents else 'Disabled'}]", "toggle_dump_latents"),
            ("Dump Latents Path", f"[{dump_path}]", "edit_dump_latents_path"),
        ]

    def _items_logging(self) -> List[Tuple[str, str, str]]:
        env = self.env
        log_file = env.get("CODEX_LOG_FILE", "")
        if log_file:
            file_label = f"Enabled ({Path(log_file).name})"
        else:
            file_label = "Disabled"
        
        # Helper to check if a log level flag is enabled
        def _flag_label(key: str, default: str = "1") -> str:
            val = env.get(key, default).strip().lower()
            return "ON" if val in ("1", "true", "yes", "on") else "OFF"
        
        return [
            # Codex Log Levels (individual toggles)
            ("Codex DEBUG", f"[{_flag_label('CODEX_LOG_DEBUG', '0')}]", "toggle_codex_debug"),
            ("Codex INFO", f"[{_flag_label('CODEX_LOG_INFO', '1')}]", "toggle_codex_info"),
            ("Codex WARNING", f"[{_flag_label('CODEX_LOG_WARNING', '1')}]", "toggle_codex_warning"),
            ("Codex ERROR", f"[{_flag_label('CODEX_LOG_ERROR', '1')}]", "toggle_codex_error"),
            ("Write Codex Log File", f"[{file_label}]", "toggle_log_file"),
            # WAN Log Levels
            ("WAN DEBUG", f"[{_flag_label('WAN_LOG_DEBUG', '0')}]", "toggle_log_debug"),
            ("WAN INFO", f"[{_flag_label('WAN_LOG_INFO', '1')}]", "toggle_log_info"),
            ("WAN WARNING", f"[{_flag_label('WAN_LOG_WARN', '1')}]", "toggle_log_warn"),
            ("WAN ERROR", f"[{_flag_label('WAN_LOG_ERROR', '1')}]", "toggle_log_error"),
        ]

    def _help_for(self, tab: str, key: str) -> List[str]:
        d: Dict[str, List[str]] = {
            "Services in new terminal": [
                "Launch API/UI in a separate Windows console window.",
                "On Linux/macOS this is ignored (attached mode only).",
            ],
            "SDPA Policy": [
                "Select SDPA kernel preference: flash | mem_efficient | math.",
                "Applied via WAN_SDPA_POLICY env when starting services.",
            ],
            "Attention Backend": [
                "Global attention backend: torch-sdpa | xformers | sage.",
                "Applied via CODEX_ATTENTION_BACKEND.",
            ],
            "Use cuda fp8 (TE)": [
                "Toggle Text Encoder FP8 (CUDA) path.",
                "On = use custom CUDA kernels (FP8) for TE; Off = use HF implementation.",
                "Requires PyTorch >= 2.1 and compatible GPU; may reduce VRAM with slight precision loss.",
                "Applied via WAN_TE_IMPL=(cuda_fp8|hf).",
            ],
            "Attn Chunk Size": [
                "Split attention into chunks to cap peak VRAM during SDPA.",
                "0 disables chunking. Use when hitting OOM: try 2048, then 1024.",
                "Trade-off: smaller chunk → lower peak memory, but slower total time.",
                "Applies to self/cross attention in DiT/UNet blocks.",
                "Applied via CODEX_ATTN_CHUNK_SIZE.",
            ],
            "GGUF Cache Policy": [
                "Control CPU-side cache of dequantized GGUF tensors (weights).",
                "none: no cache (lowest RAM, more CPU dequant work).",
                "cpu_lru: LRU cache in host RAM up to limit; reduces CPU work between steps.",
                "Useful when models don't fit fully in VRAM and you reuse the same model.",
                "Applied via CODEX_GGUF_CACHE_POLICY.",
            ],
            "GGUF Cache Limit (MB)": [
                "Max RAM (in MB) used by cpu_lru GGUF cache.",
                "Set to ~50% of free system RAM for safety. 0 disables cache.",
                "Only applies when GGUF Cache Policy = cpu_lru.",
                "Applied via CODEX_GGUF_CACHE_LIMIT_MB.",
            ],
            "Diffusion Device": [
                "Where to run the Diffusion core (UNet/DiT): Auto | GPU | CPU.",
                "CPU forces fp32 precision automatically.",
                "Applied via CODEX_DIFFUSION_DEVICE=(cuda|cpu|auto).",
            ],
            "DiT/UNet DType": [
                "Numerical dtype for the main denoiser (DiT/UNet).",
                "fp16/bf16 recommended. fp32 enforced when device=CPU.",
                "Applied via CODEX_DIFFUSION_DTYPE.",
            ],
            "Text Encoder Device": [
                "Run Text Encoder on GPU | CPU | Auto.",
                "CPU forces fp32 precision; GPU unlocks fp16/bf16/fp8.",
                "Applied via CODEX_TE_DEVICE.",
            ],
            "Text Encoder DType": [
                "Precision for Text Encoder (fp16/bf16/fp8/fp32).",
                "Forced to fp32 when device=CPU.",
                "Applied via CODEX_TE_DTYPE.",
            ],
            "VAE DType": [
                "VAE precision (fp16/bf16/fp32).",
                "Applied via CODEX_VAE_DTYPE.",
            ],
            "VAE Device": [
                "Where to run the VAE: Auto | GPU | CPU.",
                "CPU forces fp32 precision automatically.",
                "Applied via CODEX_VAE_DEVICE.",
            ],
            "Swap Policy": [
                "Swap/offload policy: never | cpu | shared (pinned).",
                "Applied via CODEX_SWAP_POLICY.",
            ],
            "Swap Method": [
                "Transfer method: blocked | async (CUDA streams).",
                "Applied via CODEX_SWAP_METHOD.",
            ],
            "Smart Offload": [
                "Stage-wise VRAM management: load only the components needed for the current stage.",
                "Enabled: TE → UNet → VAE are moved between CPU/GPU automatically.",
                "Helps GPUs with limited VRAM avoid OOM at the cost of extra transfers.",
                "Applied via CODEX_SMART_OFFLOAD or --smart-offload.",
            ],
            "Pin Shared Memory": [
                "Keep CPU tensors in pinned host memory after offloading (faster GPU reloads).",
                "Enable only when Smart Offload is on and you need faster transfers; uses extra RAM.",
                "Applied via CODEX_PIN_SHARED_MEMORY or --pin-shared-memory.",
            ],
            
            "I2V Concat Order": [
                "Order for I2V channels when assembling 36ch volume:",
                "lat_first (Comfy: latents first, then mask+img) or lat_last.",
                "Applied via WAN_I2V_ORDER.",
            ],
            "WAN_SDPA_DEBUG": [
                "Verbose log of SDPA backend selection (flash/mem/math).",
                "Effective when running WAN22 pipelines.",
            ],
            "WAN_I2V_DEBUG_HI_DECODE": [
                "Debug: decodificar tokens do High antes do handoff.",
                "Isola NaNs entre High/Low (usa 16 canais base).",
            ],
            "WAN_I2V_LAT_STATS": [
                "Print min/max/mean/std of latents before VAE decode.",
                "Useful when preview shows non-finite outputs.",
            ],
            "WAN_I2V_STRICT_VAE": [
                "If latents or decoded images contain NaN/Inf, raise",
                "a RuntimeError instead of sanitizing output.",
            ],
            "WAN_I2V_DEBUG_CLAMP": [
                "Clamp latents to ±value in preview decode only.",
                "Empty disables. Does not affect final decode.",
            ],
            "WAN_I2V_CONV32": [
                "Compute patch embed/unembed in float32 to reduce overflow.",
                "May reduce performance; recommended only for debugging.",
            ],
            "WAN_I2V_DEBUG_SANITIZE_TOKENS": [
                "Preview-only: replace NaN with 0, +Inf with +1, -Inf with -1",
                "before unembedding to latents. Helps visualize bad tokens",
                "without touching the actual pipeline output.",
            ],
            "Codex DEBUG": [
                "Enable/disable DEBUG level logs from Codex backend.",
                "Very verbose; includes internal tracing and diagnostics.",
                "Applied via CODEX_LOG_DEBUG.",
            ],
            "Codex INFO": [
                "Enable/disable INFO level logs from Codex backend.",
                "Standard operational messages.",
                "Applied via CODEX_LOG_INFO.",
            ],
            "Codex WARNING": [
                "Enable/disable WARNING level logs from Codex backend.",
                "Potential issues that don't prevent operation.",
                "Applied via CODEX_LOG_WARNING.",
            ],
            "Codex ERROR": [
                "Enable/disable ERROR level logs from Codex backend.",
                "Errors that may affect operation.",
                "Applied via CODEX_LOG_ERROR.",
            ],
            "Write Codex Log File": [
                "Toggle writing backend logs to logs/codex-<timestamp>.log.",
                "Enabled state sets CODEX_LOG_FILE automatically; disable to stop logging to file.",
            ],
            "Pipeline Debug": [
                "Toggle SDXL/txt2img pipeline trace logs (entrou/saiu).",
                "Applies via CODEX_PIPELINE_DEBUG=1 before starting the API.",
            ],
            "Trace Debug": [
                "Enable global function-call tracing (very verbose).",
                "Applies via CODEX_TRACE_DEBUG=1 / --trace-debug; restart API after toggling.",
            ],
            "Trace Max Per Func": [
                "Limit how many entries per function the call tracer records.",
                f"Set to 0 for unlimited; default is {TRACE_DEBUG_DEFAULT}.",
                "Applies via CODEX_TRACE_DEBUG_MAX_PER_FUNC.",
            ],
            "Conditioning Debug": [
                "Dump CLIP conditioning tensor norms during SDXL runs.",
                "Intended for diagnostics; adds extra logging noise.",
                "Applies via CODEX_DEBUG_COND or --debug-conditioning.",
            ],
            "Sampler Verbose Logs": [
                "Emit per-step sampler diagnostics (sigma schedule, timings).",
                "Applies via CODEX_LOG_SAMPLER.",
            ],
            "CFG Delta Logs": [
                "Log the cond/uncond delta inside CFG for the first N steps.",
                "Requires CODEX_LOG_SAMPLER=1; enabling it also enables sampler logs.",
                "Applies via CODEX_LOG_CFG_DELTA and CODEX_LOG_CFG_DELTA_N.",
            ],
            "CFG Delta Steps (N)": [
                "How many initial steps to log cond/uncond delta for (0 disables).",
                "Applies via CODEX_LOG_CFG_DELTA_N (default 2).",
            ],
            "Sigma Ladder Logs": [
                "Dump full sigma ladder (first/last and compact summary).",
                "Applies via CODEX_LOG_SIGMAS.",
            ],
            "Force Native Sampler": [
                "Force Codex to use the native sampler loop instead of k-diffusion where available.",
                "Useful for isolating sampler bugs or comparing Codex vs legacy behavior.",
                "Applies via CODEX_SAMPLER_FORCE_NATIVE=1.",
            ],
            "Dump Latents": [
                "Save final latent tensor after each sampling run.",
                "Applies via CODEX_DUMP_LATENTS; files stored under logs/diagnostics by default.",
            ],
            "Dump Latents Path": [
                "Optional file or directory to receive latent dumps.",
                "Directory → auto filenames; file path → overwrite each run.",
                "Applies via CODEX_DUMP_LATENTS_PATH.",
            ],
            "WAN DEBUG": [
                "Enable/disable DEBUG level logs from WAN runtime (very verbose).",
                "Applied via WAN_LOG_DEBUG.",
            ],
            "WAN INFO": [
                "Enable/disable INFO level logs from WAN runtime.",
                "Applied via WAN_LOG_INFO.",
            ],
            "WAN WARNING": [
                "Enable/disable WARNING level logs from WAN runtime.",
                "Applied via WAN_LOG_WARN.",
            ],
            "WAN ERROR": [
                "Enable/disable ERROR level logs from WAN runtime.",
                "Applied via WAN_LOG_ERROR.",
            ],
            "Start API": ["Start backend API server."],
            "Restart API": ["Restart backend API server (stop + start)."],
            "Kill API": ["Force kill backend API server."],
            "Start UI": ["Start UI dev server (npm dev)."],
            "Restart UI": ["Restart UI dev server."],
            "Kill UI": ["Force kill UI dev server."],
            "Logs": [
                "Combined logs. Up/Down scroll, g/G bottom/top, c clear.",
            ],
            "Save and Exit": ["Write settings to .sangoi/launcher/ and quit."],
            "Exit Without Saving": ["Quit without writing settings."],
        }
        return d.get(key, ["<Tab>, <Shift-Tab>, or <Enter> selects field."])

    # --------------- event handling ------------------------------------
    def _act_main(self, idx: int) -> None:
        items = self._items_main()
        if not (0 <= idx < len(items)):
            return
        label, _val, action = items[idx]
        name = "API" if "API" in label else ("UI" if "UI" in label else None)
        if not name:
            return
        svc = self.services[name]
        env = self.store.build_env()
        env["WAN_SDPA_POLICY"] = self.meta.sdpa_policy
        if action == "start":
            svc.start(env, external_terminal=self.meta.external_terminal)
        elif action == "restart":
            svc.restart(env, external_terminal=self.meta.external_terminal)
        elif action == "kill":
            svc.stop()  # graceful first
            time.sleep(0.2)
            svc.kill() if hasattr(svc, 'kill') else None

    def _act_runtime(self, idx: int) -> None:
        items = self._items_runtime()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env
        if action == "toggle_extterm":
            self.meta.external_terminal = not self.meta.external_terminal
        elif action == "cycle_sdpa":
            order = ["flash", "mem_efficient", "math"]
            cur = self.meta.sdpa_policy
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            self.meta.sdpa_policy = order[i]
            env["WAN_SDPA_POLICY"] = self.meta.sdpa_policy
        elif action == "cycle_attn_backend":
            order = ["torch-sdpa", "xformers", "sage"]
            cur = env.get("CODEX_ATTENTION_BACKEND", "torch-sdpa")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_ATTENTION_BACKEND"] = order[i]
        elif action == "edit_attn_chunk":
            val = self._prompt("Attn chunk size (0 disables): ")
            if val is not None and val.isdigit():
                env["CODEX_ATTN_CHUNK_SIZE"] = val
        elif action == "cycle_gguf_pol":
            order = ["none", "cpu_lru"]
            cur = env.get("CODEX_GGUF_CACHE_POLICY", "none")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_GGUF_CACHE_POLICY"] = order[i]
        elif action == "edit_gguf_lim":
            val = self._prompt("GGUF cache limit MB (0 disables): ")
            if val is not None and val.isdigit():
                env["CODEX_GGUF_CACHE_LIMIT_MB"] = val
        elif action == "cycle_unet_dtype":
            order = ["auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"]
            cur = (env.get("CODEX_DIFFUSION_DTYPE", "auto") or "auto").strip().lower()
            if cur not in order:
                cur = "auto"
            if env.get("CODEX_DIFFUSION_DEVICE", "").strip().lower() == "cpu":
                env["CODEX_DIFFUSION_DTYPE"] = "fp32"
                self.message = "Diffusion dtype locked to fp32 on CPU."
                return
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_DIFFUSION_DTYPE"] = order[i]
        elif action == "cycle_te_dtype":
            order = ["auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"]
            cur = (env.get("CODEX_TE_DTYPE", "auto") or "auto").strip().lower()
            if cur not in order:
                cur = "auto"
            if env.get("CODEX_TE_DEVICE", "").strip().lower() == "cpu":
                env["CODEX_TE_DTYPE"] = "fp32"
                self.message = "Text Encoder dtype locked to fp32 on CPU."
                return
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_TE_DTYPE"] = order[i]
        elif action == "cycle_vae_dtype":
            order = ["auto", "fp16", "bf16", "fp32"]
            cur = (env.get("CODEX_VAE_DTYPE", "auto") or "auto").strip().lower()
            if cur not in order:
                cur = "auto"
            if env.get("CODEX_VAE_DEVICE", "").strip().lower() == "cpu":
                env["CODEX_VAE_DTYPE"] = "fp32"
                self.message = "VAE dtype locked to fp32 on CPU."
                return
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_VAE_DTYPE"] = order[i]
        elif action == "select_vae_dev":
            order = ["", "cuda", "cpu"]  # Auto, GPU, CPU
            cur = env.get("CODEX_VAE_DEVICE", "").strip().lower()
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            nxt = order[i]
            env["CODEX_VAE_DEVICE"] = nxt
            if nxt == "cpu":
                env["CODEX_VAE_DTYPE"] = "fp32"
        elif action == "select_diff_device":
            order = ["", "cuda", "cpu"]  # Auto, GPU, CPU
            cur = env.get("CODEX_DIFFUSION_DEVICE", "").strip().lower()
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            nxt = order[i]
            env["CODEX_DIFFUSION_DEVICE"] = nxt
            if nxt == "cpu":
                env["CODEX_DIFFUSION_DTYPE"] = "fp32"
        elif action == "select_te_device":
            order = ["", "cuda", "cpu"]  # Auto, GPU, CPU
            cur = env.get("CODEX_TE_DEVICE", "").strip().lower()
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            nxt = order[i]
            env["CODEX_TE_DEVICE"] = nxt
            if nxt == "cpu":
                env["CODEX_TE_DTYPE"] = "fp32"
        elif action == "cycle_swap_pol":
            order = ["never", "cpu", "shared"]
            cur = env.get("CODEX_SWAP_POLICY", "cpu")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_SWAP_POLICY"] = order[i]
        elif action == "cycle_swap_mth":
            order = ["blocked", "async"]
            cur = env.get("CODEX_SWAP_METHOD", "blocked")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_SWAP_METHOD"] = order[i]
        elif action == "toggle_smart_offload":
            enabled = env.get("CODEX_SMART_OFFLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}
            if enabled:
                env.pop("CODEX_SMART_OFFLOAD", None)
                self.message = "Smart Offload disabled."
            else:
                env["CODEX_SMART_OFFLOAD"] = "1"
                self.message = "Smart Offload enabled (stage-wise VRAM loads)."
        elif action == "toggle_pin_shared":
            enabled = env.get("CODEX_PIN_SHARED_MEMORY", "0").strip().lower() in {"1", "true", "yes", "on"}
            if enabled:
                env.pop("CODEX_PIN_SHARED_MEMORY", None)
                self.message = "Pinned shared memory disabled."
            else:
                env["CODEX_PIN_SHARED_MEMORY"] = "1"
                self.message = "Pinned shared memory enabled for offloaded models."

    def _act_wan(self, idx: int) -> None:
        items = self._items_wan()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env
        if action == "cycle_i2v_order":
            order = ["lat_first", "lat_last"]
            cur = env.get("WAN_I2V_ORDER", "lat_first")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_I2V_ORDER"] = order[i]
        elif action == "cycle_offload_lvl":
            order = ["0", "1", "2", "3"]
            cur = env.get("WAN_GGUF_OFFLOAD_LEVEL", "3")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_GGUF_OFFLOAD_LEVEL"] = order[i]
        elif action == "toggle_te_fp8":
            cur = env.get("WAN_TE_IMPL", "hf").strip().lower()
            env["WAN_TE_IMPL"] = "cuda_fp8" if cur != "cuda_fp8" else "hf"
        elif action == "toggle_sdpa_debug":
            cur = env.get("WAN_SDPA_DEBUG", "0").strip().lower()
            env["WAN_SDPA_DEBUG"] = "1" if cur in ("0", "", "false", "no") else "0"
        elif action == "toggle_hi_decode_debug":
            cur = env.get("WAN_I2V_DEBUG_HI_DECODE", "0").strip().lower()
            env["WAN_I2V_DEBUG_HI_DECODE"] = "1" if cur in ("0", "", "false", "no") else "0"
        elif action == "toggle_lat_stats":
            cur = env.get("WAN_I2V_LAT_STATS", "0").strip().lower()
            env["WAN_I2V_LAT_STATS"] = "1" if cur in ("0", "", "false", "no") else "0"
        elif action == "toggle_conv32":
            cur = env.get("WAN_I2V_CONV32", "0").strip().lower()
            env["WAN_I2V_CONV32"] = "1" if cur in ("0", "", "false", "no") else "0"
        elif action == "toggle_sanitize_tokens":
            cur = env.get("WAN_I2V_DEBUG_SANITIZE_TOKENS", "0").strip().lower()
            env["WAN_I2V_DEBUG_SANITIZE_TOKENS"] = "1" if cur in ("0", "", "false", "no") else "0"
        elif action == "edit_debug_clamp":
            val = self._prompt("Debug clamp (abs, empty to disable): ")
            if val is None:
                return
            val = val.strip()
            if not val:
                env["WAN_I2V_DEBUG_CLAMP"] = ""
                return
            try:
                float(val)
            except Exception:
                self.message = "Invalid clamp value; expect float or empty"
            else:
                env["WAN_I2V_DEBUG_CLAMP"] = val
    def _act_logging(self, idx: int) -> None:
        items = self._items_logging()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env
        
        # Helper to toggle a boolean flag
        def _toggle_flag(key: str, default: str = "1") -> None:
            cur = env.get(key, default).strip().lower()
            env[key] = "0" if cur in ("1", "true", "yes", "on") else "1"
        
        # Codex log level toggles
        if action == "toggle_codex_debug":
            _toggle_flag("CODEX_LOG_DEBUG", "0")
        elif action == "toggle_codex_info":
            _toggle_flag("CODEX_LOG_INFO", "1")
        elif action == "toggle_codex_warning":
            _toggle_flag("CODEX_LOG_WARNING", "1")
        elif action == "toggle_codex_error":
            _toggle_flag("CODEX_LOG_ERROR", "1")
        elif action == "toggle_log_file":
            current = env.get("CODEX_LOG_FILE", "").strip()
            if current:
                env.pop("CODEX_LOG_FILE", None)
                self.message = "File logging disabled"
            else:
                log_path = self._ensure_log_file()
                env["CODEX_LOG_FILE"] = log_path
                self.message = f"Logging to {Path(log_path).name}"
        # WAN log level toggles
        elif action == "toggle_log_debug":
            _toggle_flag("WAN_LOG_DEBUG", "0")
        elif action == "toggle_log_info":
            _toggle_flag("WAN_LOG_INFO", "1")
        elif action == "toggle_log_warn":
            _toggle_flag("WAN_LOG_WARN", "1")
        elif action == "toggle_log_error":
            _toggle_flag("WAN_LOG_ERROR", "1")

    def _act_debug(self, idx: int) -> None:
        items = self._items_debug()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env

        def _toggle_flag(key: str, *, message_on: str | None = None, message_off: str | None = None) -> None:
            cur = env.get(key, "0").strip().lower()
            if cur in {"1", "true", "yes", "on"}:
                env.pop(key, None)
                if message_off:
                    self.message = message_off
            else:
                env[key] = "1"
                if message_on:
                    self.message = message_on

        if action == "toggle_cond_debug":
            _toggle_flag("CODEX_DEBUG_COND")
        elif action == "toggle_sampler_logs":
            _toggle_flag(
                "CODEX_LOG_SAMPLER",
                message_on="Sampler logs enabled (restart API to apply).",
                message_off="Sampler logs disabled.",
            )
            # CFG delta logs require sampler logs; if sampler logs are disabled, also disable delta logs.
            if env.get("CODEX_LOG_SAMPLER", "0").strip().lower() not in {"1", "true", "yes", "on"}:
                env.pop("CODEX_LOG_CFG_DELTA", None)
        elif action == "toggle_cfg_delta_logs":
            cur = env.get("CODEX_LOG_CFG_DELTA", "0").strip().lower()
            if cur in {"1", "true", "yes", "on"}:
                env.pop("CODEX_LOG_CFG_DELTA", None)
                self.message = "CFG delta logs disabled."
            else:
                env["CODEX_LOG_CFG_DELTA"] = "1"
                # Requires sampler logs to be enabled; turn them on automatically.
                env["CODEX_LOG_SAMPLER"] = "1"
                self.message = "CFG delta logs enabled (sampler logs enabled too). Restart API to apply."
        elif action == "edit_cfg_delta_n":
            val = self._prompt("CFG delta steps N (>=0): ")
            if val is None:
                return
            val = val.strip()
            if not val:
                env.pop("CODEX_LOG_CFG_DELTA_N", None)
                self.message = "CFG delta steps reset to default (2)."
                return
            try:
                numeric = max(0, int(val))
            except ValueError:
                self.message = "Enter a valid integer."
                return
            env["CODEX_LOG_CFG_DELTA_N"] = str(numeric)
            self.message = f"CFG delta steps set to {numeric}."
        elif action == "toggle_sigma_logs":
            _toggle_flag(
                "CODEX_LOG_SIGMAS",
                message_on="Sigma ladder logs enabled (restart API to apply).",
                message_off="Sigma ladder logs disabled.",
            )
        elif action == "toggle_force_native_sampler":
            _toggle_flag(
                "CODEX_SAMPLER_FORCE_NATIVE",
                message_on="Native sampler forced (k-diffusion disabled). Restart API to apply.",
                message_off="Native sampler automatic (k-diffusion allowed). Restart API to apply.",
            )
        elif action == "toggle_trace_debug":
            _toggle_flag(
                "CODEX_TRACE_DEBUG",
                message_on="Trace Debug enabled. Restart API to attach call tracing.",
                message_off="Trace Debug disabled. Restart API to stop call tracing.",
            )
        elif action == "toggle_pipeline_debug":
            _toggle_flag("CODEX_PIPELINE_DEBUG")
        elif action == "toggle_dump_latents":
            _toggle_flag(
                "CODEX_DUMP_LATENTS",
                message_on="Latent dumps enabled. Results will be written after sampling.",
                message_off="Latent dumps disabled.",
            )
        elif action == "edit_dump_latents_path":
            val = self._prompt("Dump path (file or directory): ")
            if val is None:
                return
            val = val.strip()
            if val:
                env["CODEX_DUMP_LATENTS_PATH"] = val
                self.message = f"Latent dump path set to {val}"
            else:
                env.pop("CODEX_DUMP_LATENTS_PATH", None)
                self.message = "Latent dump path reset to default."
        elif action == "edit_trace_max":
            val = self._prompt("Trace max per func (>=0): ")
            if val is None:
                return
            try:
                numeric = max(0, int(val))
            except ValueError:
                self.message = "Enter a valid integer."
                return
            env["CODEX_TRACE_DEBUG_MAX_PER_FUNC"] = str(numeric)
            self.message = f"Trace limit set to {numeric}."


    def _log_lines(self) -> List[str]:
        return self.log_buffer.snapshot()

    # --------------- main render ---------------------------------------
    def render(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        self._draw_bars(h, w)
        top, left_w, right_w, height = self._pane_geometry()

        # Left content box
        if self.TABS[self.tab_index] == "Logs":
            # logs view uses full left area
            logs = self._log_lines()
            total = len(logs)
            start = max(0, total - height - self.log_scroll)
            lines = logs[start:start + height]
            for i in range(height):
                ln = lines[i] if i < len(lines) else ""
                try:
                    self.stdscr.addstr(top + i, 1, (ln + " " * (left_w - 2))[:left_w - 2])
                except Exception:
                    pass
            help_lines = self._help_for("Logs", "Logs")
            self._draw_help(top, left_w, right_w, height, help_lines)
        elif self.TABS[self.tab_index] == "Exit":
            options = ["Save and Exit", "Exit Without Saving"]
            for i, opt in enumerate(options):
                attr = self.C_SEL if i == self.sel_index else self.C_NRM
                self.stdscr.attron(attr)
                self.stdscr.addstr(top + i, 1, f"{opt}" + " " * max(0, left_w - 2 - len(opt)))
                self.stdscr.attroff(attr)
            self._draw_help(top, left_w, right_w, height, self._help_for("Exit", options[self.sel_index]))
        else:
            tab_name = self.TABS[self.tab_index]
            if tab_name == "Main":
                items = self._items_main()
            elif tab_name == "Runtime":
                items = self._items_runtime()
            elif tab_name == "WAN":
                items = self._items_wan()
            elif tab_name == "DEBUG":
                items = self._items_debug()
            elif tab_name == "Logging":
                items = self._items_logging()
            else:
                items = []
            if self.sel_index >= len(items):
                self.sel_index = max(0, len(items) - 1)
            # draw items (name left, value right)
            label_w = 28  # widened label column
            if tab_name == "Main":
                # Two-row layout: first the statuses (even indices 0 and 4), then actions
                status_lines = [items[0], items[4]] if len(items) >= 8 else items[:2]
                action_lines = [it for it in items if it[2] in ("start", "restart", "kill")]
                y = top
                # Status row (bold labels)
                for key, val, _ in status_lines:
                    try:
                        self.stdscr.addstr(y, 1, f" {key:<{label_w}}  ", self.C_LBL | curses.A_BOLD)
                        self.stdscr.addstr(y, 2 + label_w, f"{val}"[:left_w - 4 - label_w], self.C_NRM | curses.A_BOLD)
                    except Exception:
                        pass
                    y += 1
                # Separator
                try:
                    self.stdscr.hline(y, 1, ord(' '), left_w - 2)
                except Exception:
                    pass
                y += 1
                # Actions grid
                for idx, (key, val, _action) in enumerate(action_lines):
                    sel_idx = len(status_lines) + 1 + idx
                    attr = self.C_SEL if sel_idx == self.sel_index else self.C_NRM
                    try:
                        self.stdscr.attron(attr)
                        self.stdscr.addstr(y, 1, f" {key:<{label_w}}  {val}"[:left_w - 2])
                        self.stdscr.attroff(attr)
                    except Exception:
                        pass
                    y += 1
            else:
                for i, (key, val, _action) in enumerate(items):
                    attr = self.C_SEL if i == self.sel_index else self.C_NRM
                    self.stdscr.attron(attr)
                    line = f" {key:<{label_w}}  {val:>}"[:left_w - 2]
                    self.stdscr.addstr(top + i, 1, line)
                    self.stdscr.attroff(attr)
            # help
            key = items[self.sel_index][0] if items else ""
            help_lines = self._help_for(tab_name, key)
            if tab_name == "Runtime" and self.launch_checks:
                help_lines = help_lines + ["", "Launch Checks:"]
                for chk in self.launch_checks:
                    status = "OK" if chk.ok else "FAIL"
                    help_lines.append(f"- {chk.name}: {status}")
                    help_lines.append(f"  {chk.detail}")
            self._draw_help(top, left_w, right_w, height, help_lines)

        # transient message (below tabs)
        if self.message:
            try:
                self.stdscr.addstr(top - 1, 1, (self.message + " " * (w - 2))[:w - 2], self.C_LBL)
            except Exception:
                pass
        self.stdscr.refresh()

        # Keep popup painted on top each frame to avoid flicker/overdraw
        if self.popup_active and self.popup_selection is not None:
            try:
                title, options, idx, _action = self.popup_selection
                self._draw_select_popup(title, options, idx)
            except Exception:
                pass

    def handle_key(self, ch: int) -> None:
        tab = self.TABS[self.tab_index]
        # If a popup is active, handle navigation/confirm/cancel here
        if self.popup_active and self.popup_selection is not None:
            title, options, idx, action = self.popup_selection
            # Navigate within popup
            if ch in (curses.KEY_UP, ord('k')):
                idx = (idx - 1) % len(options)
                self.popup_selection = (title, options, idx, action)
                self._draw_select_popup(title, options, idx)
                return
            if ch in (curses.KEY_DOWN, ord('j')):
                idx = (idx + 1) % len(options)
                self.popup_selection = (title, options, idx, action)
                self._draw_select_popup(title, options, idx)
                return
            if ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                return
            if ch in (curses.KEY_ENTER, ord('\n'), ord('\r'), 10, 13, 343):
                self._apply_popup_selection(action, options[idx])
                self.popup_active = False
                self.popup_selection = None
                return
            if ch == 27:  # ESC cancels popup
                self.popup_active = False
                self.popup_selection = None
                return
        if ch in (curses.KEY_LEFT, ord('\t') - 256):
            self.tab_index = (self.tab_index - 1) % len(self.TABS)
            self.sel_index = 0
            self.meta.tab_index = self.tab_index
            if self.TABS[self.tab_index] == "Runtime":
                self.launch_checks = run_launch_checks()
            return
        if ch in (curses.KEY_RIGHT,):
            self.tab_index = (self.tab_index + 1) % len(self.TABS)
            self.sel_index = 0
            self.meta.tab_index = self.tab_index
            if self.TABS[self.tab_index] == "Runtime":
                self.launch_checks = run_launch_checks()
            return
        if ch in (curses.KEY_UP, ord('k')):
            self.sel_index = max(0, self.sel_index - 1)
            return
        if ch in (curses.KEY_DOWN, ord('j')):
            self.sel_index += 1
            return
        if ch in (curses.KEY_ENTER, ord('\n'), ord('\r'), 10, 13, 343):
            if tab == "Main":
                self._act_main(self.sel_index)
            elif tab == "Runtime":
                # Open BIOS-style popup for multi-choice fields
                self._act_runtime_popup_or_apply(self.sel_index)
            elif tab == "WAN":
                self._act_wan_popup_or_apply(self.sel_index)
            elif tab == "DEBUG":
                self._act_debug(self.sel_index)
            elif tab == "Logging":
                self._act_logging(self.sel_index)
            elif tab == "Exit":
                if self.sel_index % 2 == 0:
                    self._persist()
                raise SystemExit
            return
        if ch in (ord('+'), ord('-')):
            if tab == "Runtime":
                self._act_runtime(self.sel_index)
            elif tab == "WAN":
                self._act_wan(self.sel_index)
            elif tab == "DEBUG":
                self._act_debug(self.sel_index)
            elif tab == "Logging":
                self._act_logging(self.sel_index)
            return
        if tab == "Logs":
            logs = self._log_lines()
            if ch in (ord('g'),):
                self.log_scroll = 0
                return
            if ch in (ord('G'),):
                self.log_scroll = max(0, len(logs) - 1)
                return
            if ch in (ord('c'), ord('C')):
                self.log_buffer.clear()
                self.log_scroll = 0
                return
        # Function keys shortcuts
        if ch == curses.KEY_F2:
            env = self.store.build_env()
            env["WAN_SDPA_POLICY"] = self.meta.sdpa_policy
            self.services["API"].start(env, external_terminal=self.meta.external_terminal)
            return
        if ch == curses.KEY_F3:
            # fast kill API
            self.services["API"].kill()
            return
        if ch == curses.KEY_F4:
            env = self.store.build_env()
            env["WAN_SDPA_POLICY"] = self.meta.sdpa_policy
            self.services["UI"].start(env, external_terminal=self.meta.external_terminal)
            return
        if ch == curses.KEY_F5:
            self.services["UI"].kill()
            return
        if ch == curses.KEY_F1:
            self._show_help_popup()
            return
        if ch == 27:  # ESC
            if self.popup_active:
                self.popup_active = False
                self.popup_selection = None
                return
            raise SystemExit
        if ch == curses.KEY_F10:
            self._persist()
            raise SystemExit

    # ---------------- popup helpers ------------------------------------
    def _show_help_popup(self) -> None:
        lines = [
            "Keyboard Shortcuts:",
            "- Left/Right: Change Tab",
            "- Up/Down: Move Selection",
            "- Enter: Select / Open popup",
            "- +/-: Change value inline",
            "- F2 / F3: Start / Kill API",
            "- F4 / F5: Start / Kill UI",
            "- F10: Save and Exit",
            "- Esc: Exit (or close popup)",
        ]
        self._draw_popup("Help", lines)

    def _draw_popup(self, title: str, lines: List[str], *, selectable: bool = False, options: Optional[List[str]] = None, index: int = 0) -> None:
        h, w = self.stdscr.getmaxyx()
        ph = min(len(lines) + 4, max(8, len(lines) + 4))
        pw = min(max(len(title) + 8, max((len(l) for l in lines), default=30) + 4), w - 4)
        y0 = max(2, (h - ph) // 2)
        x0 = max(2, (w - pw) // 2)
        win = curses.newwin(ph, pw, y0, x0)
        win.bkgd(' ', self.C_NRM)
        win.box()
        try:
            win.addstr(0, 2, f" {title} ", self.C_TAB)
        except Exception:
            pass
        for i, ln in enumerate(lines):
            try:
                win.addstr(2 + i, 2, ln[: pw - 4])
            except Exception:
                pass
        win.refresh()
        self.popup_active = True
        # Wait for any key to close if not selectable
        if not selectable:
            self.stdscr.nodelay(False)
            win.getch()
            self.stdscr.nodelay(True)
            self.popup_active = False
            return

    def _act_runtime_popup_or_apply(self, idx: int) -> None:
        items = self._items_runtime()
        if not (0 <= idx < len(items)):
            return
        key, _val, action = items[idx]
        env = self.env
        choices_map = {
            'cycle_sdpa': ["flash", "mem_efficient", "math"],
            'cycle_attn_backend': ["torch-sdpa", "xformers", "sage"],
            'cycle_gguf_pol': ["none", "cpu_lru"],
            'cycle_unet_dtype': ["auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"],
            'cycle_te_dtype': ["auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"],
            'cycle_vae_dtype': ["auto", "fp16", "bf16", "fp32"],
            'cycle_swap_pol': ["never", "cpu", "shared"],
            'cycle_swap_mth': ["blocked", "async"],
            'select_diff_device': ["Auto", "GPU", "CPU"],
            'select_te_device': ["Auto", "GPU", "CPU"],
            'select_vae_dev': ["Auto", "GPU", "CPU"],
        }
        if action not in choices_map:
            return self._act_runtime(idx)
        options = choices_map[action]

        def _device_option_label(key_name: str) -> str:
            raw = env.get(key_name, "").strip().lower()
            if raw == "gpu":
                env[key_name] = "cuda"
                raw = "cuda"
            if raw == "cuda":
                return "GPU"
            if raw == "cpu":
                return "CPU"
            return "Auto"

        cur_val = {
            'cycle_sdpa': self.meta.sdpa_policy,
            'cycle_attn_backend': env.get('CODEX_ATTENTION_BACKEND', 'torch-sdpa'),
            'cycle_gguf_pol': env.get('CODEX_GGUF_CACHE_POLICY', 'none'),
            'cycle_unet_dtype': (env.get('CODEX_DIFFUSION_DTYPE', 'auto') or 'auto').strip().lower(),
            'cycle_te_dtype': (env.get('CODEX_TE_DTYPE', 'auto') or 'auto').strip().lower(),
            'cycle_vae_dtype': (env.get('CODEX_VAE_DTYPE', 'auto') or 'auto').strip().lower(),
            'cycle_swap_pol': env.get('CODEX_SWAP_POLICY', 'cpu'),
            'cycle_swap_mth': env.get('CODEX_SWAP_METHOD', 'blocked'),
            'select_diff_device': _device_option_label('CODEX_DIFFUSION_DEVICE'),
            'select_te_device': _device_option_label('CODEX_TE_DEVICE'),
            'select_vae_dev': _device_option_label('CODEX_VAE_DEVICE'),
        }[action]
        try:
            sel = options.index(cur_val)
        except ValueError:
            sel = 0
        self.popup_selection = (key, options, sel, action)
        self._draw_select_popup(key, options, sel)
    def _act_wan_popup_or_apply(self, idx: int) -> None:
        items = self._items_wan()
        if not (0 <= idx < len(items)):
            return
        key, _val, action = items[idx]
        env = self.env
        choices_map = {
            'cycle_i2v_order': ["lat_first", "lat_last"],
            'cycle_offload_lvl': ["0", "1", "2", "3"],
        }
        if action not in choices_map:
            return self._act_wan(idx)
        options = choices_map[action]
        cur_val = {
            'cycle_i2v_order': env.get('WAN_I2V_ORDER', 'lat_first'),
            'cycle_offload_lvl': env.get('WAN_GGUF_OFFLOAD_LEVEL', '3'),
        }[action]
        try:
            sel = options.index(cur_val)
        except ValueError:
            sel = 0
        self.popup_selection = (key, options, sel, action)
        self._draw_select_popup(key, options, sel)


    def _draw_select_popup(self, title: str, options: List[str], index: int) -> None:
        self.popup_active = True
        self.stdscr.nodelay(True)
        h, w = self.stdscr.getmaxyx()
        ph = min(len(options) + 4, max(8, len(options) + 4))
        pw = min(max(len(title) + 8, max((len(o) for o in options), default=20) + 6), w - 4)
        y0 = max(2, (h - ph) // 2)
        x0 = max(2, (w - pw) // 2)
        win = curses.newwin(ph, pw, y0, x0)
        win.bkgd(' ', self.C_NRM)
        win.box()
        try:
            win.addstr(0, 2, f" {title} ", self.C_TAB)
        except Exception:
            pass
        for i, opt in enumerate(options):
            attr = self.C_SEL if i == index else self.C_NRM
            try:
                win.attron(attr)
                win.addstr(2 + i, 2, opt.ljust(pw - 4)[: pw - 4])
                win.attroff(attr)
            except Exception:
                pass
        win.refresh()
        # update current popup selection tuple if exists (keep action and options)
        if self.popup_selection is not None:
            title0, opts0, _idx0, action0 = self.popup_selection
            # if options list changed, keep as is; otherwise set new index
            if opts0 == options and title0 == title:
                self.popup_selection = (title, options, index, action0)

    def _apply_popup_selection(self, action: str, value: str) -> None:
        env = self.env
        if action == 'cycle_sdpa':
            self.meta.sdpa_policy = value
            env['WAN_SDPA_POLICY'] = value
        elif action == 'cycle_offload_lvl':
            env['WAN_GGUF_OFFLOAD_LEVEL'] = value
        elif action == 'cycle_attn_backend':
            env['CODEX_ATTENTION_BACKEND'] = value
        elif action == 'cycle_gguf_pol':
            env['CODEX_GGUF_CACHE_POLICY'] = value
        elif action == 'cycle_unet_dtype':
            if env.get('CODEX_DIFFUSION_DEVICE', '').strip().lower() == 'cpu' and value != 'fp32':
                env['CODEX_DIFFUSION_DTYPE'] = 'fp32'
                self.message = "Diffusion dtype locked to fp32 on CPU."
            else:
                env['CODEX_DIFFUSION_DTYPE'] = value
        elif action == 'cycle_te_dtype':
            if env.get('CODEX_TE_DEVICE', '').strip().lower() == 'cpu' and value != 'fp32':
                env['CODEX_TE_DTYPE'] = 'fp32'
                self.message = "Text Encoder dtype locked to fp32 on CPU."
            else:
                env['CODEX_TE_DTYPE'] = value
        elif action == 'cycle_vae_dtype':
            if env.get('CODEX_VAE_DEVICE', '').strip().lower() == 'cpu' and value != 'fp32':
                env['CODEX_VAE_DTYPE'] = 'fp32'
                self.message = "VAE dtype locked to fp32 on CPU."
            else:
                env['CODEX_VAE_DTYPE'] = value
        elif action == 'cycle_swap_pol':
            env['CODEX_SWAP_POLICY'] = value
        elif action == 'cycle_swap_mth':
            env['CODEX_SWAP_METHOD'] = value
        elif action == 'cycle_i2v_order':
            env['WAN_I2V_ORDER'] = value
        elif action == 'select_diff_device':
            v = value.strip().lower()
            if v == 'gpu':
                env['CODEX_DIFFUSION_DEVICE'] = 'cuda'
            elif v == 'cpu':
                env['CODEX_DIFFUSION_DEVICE'] = 'cpu'
                env['CODEX_DIFFUSION_DTYPE'] = 'fp32'
            else:
                env['CODEX_DIFFUSION_DEVICE'] = ''
        elif action == 'select_te_device':
            v = value.strip().lower()
            if v == 'gpu':
                env['CODEX_TE_DEVICE'] = 'cuda'
            elif v == 'cpu':
                env['CODEX_TE_DEVICE'] = 'cpu'
                env['CODEX_TE_DTYPE'] = 'fp32'
            else:
                env['CODEX_TE_DEVICE'] = ''
        elif action == 'select_vae_dev':
            v = value.strip().lower()
            if v == 'gpu':
                env['CODEX_VAE_DEVICE'] = 'cuda'
            elif v == 'cpu':
                env['CODEX_VAE_DEVICE'] = 'cpu'
                env['CODEX_VAE_DTYPE'] = 'fp32'
            else:
                env['CODEX_VAE_DEVICE'] = ''



def main(stdscr) -> None:
    locale.setlocale(locale.LC_ALL, '')
    try:
        curses.curs_set(0)
    except Exception:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(100)
    app = BIOSApp(stdscr)
    while True:
        app.render()
        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            break
        if ch == -1:
            continue
        try:
            app.handle_key(ch)
        except SystemExit:
            break
        except Exception as exc:
            app.message = f"Error: {exc}"


if __name__ == "__main__":
    curses.wrapper(main)
