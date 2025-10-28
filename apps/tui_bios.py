#!/usr/bin/env python3
"""BIOS-style curses TUI (pure stdlib) for managing services and options.

Settings persist under ``.sangoi/launcher/`` segmented in meta/areas/models.

Tabs
- Main: status + actions (Start/Restart/Kill API/UI)
- Advanced: runtime options (Services in new terminal)
- Security: environment toggles (WAN_SDPA_DEBUG)
- Logs: combined logs with scrolling
- Exit: Save and Exit / Exit Without Saving

Controls
- Left/Right: change tab
- Up/Down: move selection
- Enter: select / toggle
- +/-: change value (where applicable)
- g / G: bottom/top in Logs
- c: clear logs (Logs tab)
- F10: Save and Exit
- Esc: Exit without saving

Notes
- External terminal launch is supported only on Windows (new console window).
- WAN_SDPA_DEBUG toggles verbose SDPA backend logs in wan22 runtime.
"""
from __future__ import annotations

import curses
import locale
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    TABS = ["Main", "Advanced", "Security", "Logs", "Exit"]

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

    def _items_advanced(self) -> List[Tuple[str, str, str]]:
        env = self.env
        ext = "Enabled" if self.meta.external_terminal else "Disabled"
        pol = self.meta.sdpa_policy
        attn_backend = env.get("CODEX_ATTENTION_BACKEND", "torch-sdpa")
        attn_chunk = env.get("CODEX_ATTN_CHUNK_SIZE", "0")
        gguf_pol = env.get("CODEX_GGUF_CACHE_POLICY", "none")
        gguf_lim = env.get("CODEX_GGUF_CACHE_LIMIT_MB", "0")
        offload_lvl = env.get("WAN_GGUF_OFFLOAD_LEVEL", "3")
        unet_dtype = env.get("CODEX_UNET_DTYPE", "fp16")
        vae_dtype = env.get("CODEX_VAE_DTYPE", "fp16")
        # VAE device label: Auto/GPU/CPU (prefer CODEX_VAE_DEVICE if set)
        _vae_dev = env.get("CODEX_VAE_DEVICE", "").strip().lower()
        if _vae_dev == "cpu":
            vae_dev_label = "CPU"
        elif _vae_dev in ("cuda", "gpu"):
            vae_dev_label = "GPU"
        else:
            vae_cpu_flag = env.get("CODEX_VAE_IN_CPU", "0").lower() in ("1","true","yes","on")
            vae_dev_label = "CPU" if vae_cpu_flag else "Auto"
        all_fp32 = "Enabled" if env.get("CODEX_ALL_IN_FP32", "0").lower() in ("1","true","yes","on") else "Disabled"
        swap_pol = env.get("CODEX_SWAP_POLICY", "cpu")
        swap_mth = env.get("CODEX_SWAP_METHOD", "blocked")
        te_impl = env.get("WAN_TE_IMPL", "hf")
        te_fp8_label = "Enabled" if te_impl == "cuda_fp8" else "Disabled"
        te_dev_env = env.get("WAN_TE_DEVICE", "").strip().lower()
        te_dev_label = "CPU" if te_dev_env == "cpu" else ("GPU" if te_dev_env in ("cuda", "gpu") else "Auto")
        return [
            ("Services in new terminal", f"[{ext}]", "toggle_extterm"),
            ("I2V Concat Order", f"[{env.get('WAN_I2V_ORDER','lat_first')}]", "cycle_i2v_order"),
            ("GGUF Offload Level", f"[{offload_lvl}]", "cycle_offload_lvl"),
            ("SDPA Policy", f"[{pol}]", "cycle_sdpa"),
            ("Use cuda fp8 (TE)", f"[{te_fp8_label}]", "toggle_te_fp8"),
            ("TE Device", f"[{te_dev_label}]", "select_te_dev"),
            # Removed: TE CUDA Required — derived from WAN_TE_IMPL
            ("Attention Backend", f"[{attn_backend}]", "cycle_attn_backend"),
            ("Attn Chunk Size", f"[{attn_chunk}]", "edit_attn_chunk"),
            ("GGUF Cache Policy", f"[{gguf_pol}]", "cycle_gguf_pol"),
            ("GGUF Cache Limit (MB)", f"[{gguf_lim}]", "edit_gguf_lim"),
            ("DiT/UNet DType", f"[{unet_dtype}]", "cycle_unet_dtype"),
            ("VAE DType", f"[{vae_dtype}]", "cycle_vae_dtype"),
            ("VAE device", f"[{vae_dev_label}]", "select_vae_dev"),
            ("All in FP32", f"[{all_fp32}]", "toggle_all_fp32"),
            ("Swap Policy", f"[{swap_pol}]", "cycle_swap_pol"),
            ("Swap Method", f"[{swap_mth}]", "cycle_swap_mth"),
        ]

    def _items_security(self) -> List[Tuple[str, str, str]]:
        v = self.env.get("WAN_SDPA_DEBUG", "0")
        label = "Enabled" if v.lower() in ("1", "true", "yes", "on") else "Disabled"
        dbg_hi = self.env.get("WAN_I2V_DEBUG_HI_DECODE", "0")
        label_hi = "Enabled" if dbg_hi.lower() in ("1", "true", "yes", "on") else "Disabled"
        lat_stats = self.env.get("WAN_I2V_LAT_STATS", "0")
        label_lat = "Enabled" if lat_stats.lower() in ("1", "true", "yes", "on") else "Disabled"
        strict_vae = self.env.get("WAN_I2V_STRICT_VAE", "0")
        label_strict = "Enabled" if strict_vae.lower() in ("1", "true", "yes", "on") else "Disabled"
        conv32 = self.env.get("WAN_I2V_CONV32", "0")
        label_conv32 = "Enabled" if conv32.lower() in ("1", "true", "yes", "on") else "Disabled"
        return [
            ("WAN_SDPA_DEBUG", f"[{label}]", "toggle_sdpa_debug"),
            ("WAN_I2V_DEBUG_HI_DECODE", f"[{label_hi}]", "toggle_hi_decode_debug"),
            ("WAN_I2V_LAT_STATS", f"[{label_lat}]", "toggle_lat_stats"),
            ("WAN_I2V_CONV32", f"[{label_conv32}]", "toggle_conv32"),
            ("WAN_LOG_INFO", f"[{self.env.get('WAN_LOG_INFO','1')}]", "toggle_log_info"),
            ("WAN_LOG_WARN", f"[{self.env.get('WAN_LOG_WARN','1')}]", "toggle_log_warn"),
            ("WAN_LOG_ERROR", f"[{self.env.get('WAN_LOG_ERROR','1')}]", "toggle_log_error"),
            ("WAN_LOG_DEBUG", f"[{self.env.get('WAN_LOG_DEBUG','0')}]", "toggle_log_debug"),
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
            "TE Device": [
                "Run Text Encoder on GPU or CPU.",
                "GPU: faster, higher VRAM usage; CPU: slower, frees VRAM.",
                "If 'Use cuda fp8 (TE)' is ON, device must be GPU.",
                "Applied via WAN_TE_DEVICE=(cuda|cpu).",
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
            "DiT/UNet DType": [
                "Numerical dtype for the main denoiser (DiT/UNet).",
                "fp16/bf16 recommended. fp32 only for diagnostics (slow/high VRAM).",
                "Applied via CODEX_UNET_DTYPE.",
            ],
            "VAE DType": [
                "VAE precision (fp16/bf16/fp32).",
                "Applied via CODEX_VAE_DTYPE.",
            ],
            "VAE device": [
                "Where to run the VAE: Auto | GPU | CPU.",
                "Auto: runtime decides; GPU: faster, uses VRAM; CPU: frees VRAM (slower).",
                "Applied via CODEX_VAE_DEVICE=(cuda|cpu) and mirrors CODEX_VAE_IN_CPU.",
            ],
            "All in FP32": [
                "Force fp32 globally (CODEX_ALL_IN_FP32=1).",
            ],
            "Swap Policy": [
                "Swap/offload policy: never | cpu | shared (pinned).",
                "Applied via CODEX_SWAP_POLICY.",
            ],
            "Swap Method": [
                "Transfer method: blocked | async (CUDA streams).",
                "Applied via CODEX_SWAP_METHOD.",
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
            "WAN_LOG_INFO": [
                "Enable/disable info-level logs from WAN runtime.",
            ],
            "WAN_LOG_WARN": [
                "Enable/disable warn-level logs from WAN runtime.",
            ],
            "WAN_LOG_ERROR": [
                "Enable/disable error-level logs from WAN runtime.",
            ],
            "WAN_LOG_DEBUG": [
                "Enable/disable debug-level logs (very verbose).",
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

    def _act_advanced(self, idx: int) -> None:
        items = self._items_advanced()
        if not (0 <= idx < len(items)):
            return
        key, _val, action = items[idx]
        env = self.env
        if action == "toggle_extterm":
            self.meta.external_terminal = not self.meta.external_terminal
        elif action == "toggle_te_fp8":
            # Binary toggle: hf <-> cuda_fp8
            cur = env.get("WAN_TE_IMPL", "hf").strip().lower()
            env["WAN_TE_IMPL"] = "cuda_fp8" if cur != "cuda_fp8" else "hf"
        elif action == "toggle_te_dev":
            cur = env.get("WAN_TE_DEVICE", "").strip().lower()
            # Default to GPU if unset; toggle CPU<->GPU
            env["WAN_TE_DEVICE"] = "cpu" if cur not in ("cpu",) else "cuda"
        elif action == "cycle_sdpa":
            order = ["flash", "mem_efficient", "math"]
            cur = self.meta.sdpa_policy
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            self.meta.sdpa_policy = order[i]
            env["WAN_SDPA_POLICY"] = self.meta.sdpa_policy
        elif action == "cycle_i2v_order":
            order = ["lat_first", "lat_last"]
            cur = env.get("WAN_I2V_ORDER", "lat_first")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_I2V_ORDER"] = order[i]
        elif action == "cycle_te_impl":
            order = ["hf", "cuda_fp8"]
            cur = env.get("WAN_TE_IMPL", "hf")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_TE_IMPL"] = order[i]
        elif action == "cycle_offload_lvl":
            order = ["0", "1", "2", "3"]
            cur = env.get("WAN_GGUF_OFFLOAD_LEVEL", "3")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_GGUF_OFFLOAD_LEVEL"] = order[i]
        # Removed: toggle_te_required — implied by TE implementation
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
            order = ["fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"]
            cur = env.get("CODEX_UNET_DTYPE", "fp16")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_UNET_DTYPE"] = order[i]
        elif action == "cycle_vae_dtype":
            order = ["fp16", "bf16", "fp32"]
            cur = env.get("CODEX_VAE_DTYPE", "fp16")
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["CODEX_VAE_DTYPE"] = order[i]
        elif action == "toggle_vae_cpu":
            cur = env.get("CODEX_VAE_IN_CPU", "0").strip().lower()
            env["CODEX_VAE_IN_CPU"] = "0" if cur in ("1","true","yes","on") else "1"
        elif action == "toggle_all_fp32":
            cur = env.get("CODEX_ALL_IN_FP32", "0").strip().lower()
            env["CODEX_ALL_IN_FP32"] = "0" if cur in ("1","true","yes","on") else "1"
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
        elif action == "toggle_gpu_pref":
            cur = env.get("CODEX_GPU_PREFER_CONSTRUCT", "0").strip().lower()
            env["CODEX_GPU_PREFER_CONSTRUCT"] = "0" if cur in ("1","true","yes","on") else "1"
        elif action == "select_te_dev":
            # Cycle TE device with +/- as a fallback to popup
            order = ["", "cuda", "cpu"]  # Auto, GPU, CPU
            cur = env.get("WAN_TE_DEVICE", "").strip().lower()
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            env["WAN_TE_DEVICE"] = order[i]
        elif action == "select_vae_dev":
            # Cycle VAE device with +/- as a fallback to popup
            order = ["", "cuda", "cpu"]  # Auto, GPU, CPU
            cur = env.get("CODEX_VAE_DEVICE", "").strip().lower()
            try:
                i = (order.index(cur) + 1) % len(order)
            except ValueError:
                i = 0
            nxt = order[i]
            env["CODEX_VAE_DEVICE"] = nxt
            # Keep mirror for legacy toggle
            env["CODEX_VAE_IN_CPU"] = "1" if nxt == "cpu" else "0"

    def _act_security(self, idx: int) -> None:
        items = self._items_security()
        if not (0 <= idx < len(items)):
            return
        key, _val, action = items[idx]
        if action == "toggle_sdpa_debug":
            cur = self.env.get("WAN_SDPA_DEBUG", "0").strip().lower()
            nxt = "1" if cur in ("0", "", "false", "no") else "0"
            self.env["WAN_SDPA_DEBUG"] = nxt
        elif action == "toggle_hi_decode_debug":
            cur = self.env.get("WAN_I2V_DEBUG_HI_DECODE", "0").strip().lower()
            nxt = "1" if cur in ("0", "", "false", "no") else "0"
            self.env["WAN_I2V_DEBUG_HI_DECODE"] = nxt
        elif action == "toggle_lat_stats":
            cur = self.env.get("WAN_I2V_LAT_STATS", "0").strip().lower()
            nxt = "1" if cur in ("0", "", "false", "no") else "0"
            self.env["WAN_I2V_LAT_STATS"] = nxt
        # removed toggle_strict_vae (now hard error on non-finite)
        elif action == "edit_debug_clamp":
            val = self._prompt("Debug clamp (abs, empty to disable): ")
            if val is None:
                return
            val = val.strip()
            if val == "":
                self.env["WAN_I2V_DEBUG_CLAMP"] = ""
                return
            try:
                float(val)
                self.env["WAN_I2V_DEBUG_CLAMP"] = val
            except Exception:
                self.message = "Invalid clamp value; expect float or empty"
        elif action == "toggle_conv32":
            cur = self.env.get("WAN_I2V_CONV32", "0").strip().lower()
            nxt = "1" if cur in ("0", "", "false", "no") else "0"
            self.env["WAN_I2V_CONV32"] = nxt
        elif action == "toggle_sanitize_tokens":
            cur = self.env.get("WAN_I2V_DEBUG_SANITIZE_TOKENS", "0").strip().lower()
            nxt = "1" if cur in ("0", "", "false", "no") else "0"
            self.env["WAN_I2V_DEBUG_SANITIZE_TOKENS"] = nxt
        elif action == "toggle_log_info":
            cur = self.env.get("WAN_LOG_INFO", "1").strip().lower()
            self.env["WAN_LOG_INFO"] = "0" if cur in ("1","true","yes","on") else "1"
        elif action == "toggle_log_warn":
            cur = self.env.get("WAN_LOG_WARN", "1").strip().lower()
            self.env["WAN_LOG_WARN"] = "0" if cur in ("1","true","yes","on") else "1"
        elif action == "toggle_log_error":
            cur = self.env.get("WAN_LOG_ERROR", "1").strip().lower()
            self.env["WAN_LOG_ERROR"] = "0" if cur in ("1","true","yes","on") else "1"
        elif action == "toggle_log_debug":
            cur = self.env.get("WAN_LOG_DEBUG", "0").strip().lower()
            self.env["WAN_LOG_DEBUG"] = "0" if cur in ("1","true","yes","on") else "1"

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
            if self.TABS[self.tab_index] == "Main":
                items = self._items_main()
            elif self.TABS[self.tab_index] == "Advanced":
                items = self._items_advanced()
            else:
                items = self._items_security()
            # draw items (name left, value right)
            label_w = 28  # widened label column
            if self.TABS[self.tab_index] == "Main":
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
            help_lines = self._help_for(self.TABS[self.tab_index], key)
            if self.TABS[self.tab_index] == "Advanced" and self.launch_checks:
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
            if self.TABS[self.tab_index] == "Advanced":
                self.launch_checks = run_launch_checks()
            return
        if ch in (curses.KEY_RIGHT,):
            self.tab_index = (self.tab_index + 1) % len(self.TABS)
            self.sel_index = 0
            self.meta.tab_index = self.tab_index
            if self.TABS[self.tab_index] == "Advanced":
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
            elif tab == "Advanced":
                # Open BIOS-style popup for multi-choice fields
                self._act_advanced_popup_or_apply(self.sel_index)
            elif tab == "Security":
                self._act_security(self.sel_index)
            elif tab == "Exit":
                if self.sel_index % 2 == 0:
                    self._persist()
                raise SystemExit
            return
        if ch in (ord('+'), ord('-')):
            if tab == "Advanced":
                self._act_advanced(self.sel_index)
            elif tab == "Security":
                self._act_security(self.sel_index)
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

    def _act_advanced_popup_or_apply(self, idx: int) -> None:
        items = self._items_advanced()
        if not (0 <= idx < len(items)):
            return
        key, val, action = items[idx]
        env = self.env
        # Define options per action
        choices_map = {
            'cycle_sdpa': ["flash", "mem_efficient", "math"],
            'cycle_te_impl': ["hf", "cuda_fp8"],
            'cycle_offload_lvl': ["0", "1", "2", "3"],
            'cycle_attn_backend': ["torch-sdpa", "xformers", "sage"],
            'cycle_gguf_pol': ["none", "cpu_lru"],
            'cycle_unet_dtype': ["fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "fp32"],
            'cycle_vae_dtype': ["fp16", "bf16", "fp32"],
            'cycle_swap_pol': ["never", "cpu", "shared"],
            'cycle_swap_mth': ["blocked", "async"],
            'cycle_i2v_order': ["lat_first", "lat_last"],
            'select_te_dev': ["Auto", "GPU", "CPU"],
            'select_vae_dev': ["Auto", "GPU", "CPU"],
        }
        if action not in choices_map:
            # Default: execute normal action
            return self._act_advanced(idx)
        options = choices_map[action]
        # Determine current index based on env
        cur_val = {
            'cycle_sdpa': self.meta.sdpa_policy,
            'cycle_te_impl': env.get('WAN_TE_IMPL', 'hf'),
            'cycle_offload_lvl': env.get('WAN_GGUF_OFFLOAD_LEVEL', '3'),
            'cycle_attn_backend': env.get('CODEX_ATTENTION_BACKEND', 'torch-sdpa'),
            'cycle_gguf_pol': env.get('CODEX_GGUF_CACHE_POLICY', 'none'),
            'cycle_unet_dtype': env.get('CODEX_UNET_DTYPE', 'fp16'),
            'cycle_vae_dtype': env.get('CODEX_VAE_DTYPE', 'fp16'),
            'cycle_swap_pol': env.get('CODEX_SWAP_POLICY', 'cpu'),
            'cycle_swap_mth': env.get('CODEX_SWAP_METHOD', 'blocked'),
            'cycle_i2v_order': env.get('WAN_I2V_ORDER', 'lat_first'),
            'select_te_dev': (lambda v: 'GPU' if v in ('cuda','gpu') else ('CPU' if v=='cpu' else 'Auto'))(env.get('WAN_TE_DEVICE','').strip().lower()),
            'select_vae_dev': (lambda v, c: ('GPU' if v in ('cuda','gpu') else ('CPU' if v=='cpu' else ('CPU' if c else 'Auto'))))(env.get('CODEX_VAE_DEVICE','').strip().lower(), env.get('CODEX_VAE_IN_CPU','0').strip().lower() in ('1','true','yes','on')),
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
        elif action == 'cycle_te_impl':
            env['WAN_TE_IMPL'] = value
        elif action == 'cycle_offload_lvl':
            env['WAN_GGUF_OFFLOAD_LEVEL'] = value
        elif action == 'cycle_attn_backend':
            env['CODEX_ATTENTION_BACKEND'] = value
        elif action == 'cycle_gguf_pol':
            env['CODEX_GGUF_CACHE_POLICY'] = value
        elif action == 'cycle_unet_dtype':
            env['CODEX_UNET_DTYPE'] = value
        elif action == 'cycle_vae_dtype':
            env['CODEX_VAE_DTYPE'] = value
        elif action == 'cycle_swap_pol':
            env['CODEX_SWAP_POLICY'] = value
        elif action == 'cycle_swap_mth':
            env['CODEX_SWAP_METHOD'] = value
        elif action == 'cycle_i2v_order':
            env['WAN_I2V_ORDER'] = value
        elif action == 'select_te_dev':
            # Map display value to env
            v = value.strip().lower()
            if v == 'gpu':
                env['WAN_TE_DEVICE'] = 'cuda'
            elif v == 'cpu':
                env['WAN_TE_DEVICE'] = 'cpu'
            else:
                env['WAN_TE_DEVICE'] = ''  # Auto
        elif action == 'select_vae_dev':
            v = value.strip().lower()
            if v == 'gpu':
                env['CODEX_VAE_DEVICE'] = 'cuda'
                env['CODEX_VAE_IN_CPU'] = '0'
            elif v == 'cpu':
                env['CODEX_VAE_DEVICE'] = 'cpu'
                env['CODEX_VAE_IN_CPU'] = '1'
            else:
                env['CODEX_VAE_DEVICE'] = ''
                env['CODEX_VAE_IN_CPU'] = '0'



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
