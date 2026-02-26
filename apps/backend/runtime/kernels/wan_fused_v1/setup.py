"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Build script for `wan_fused_v1_cuda` CUDA extension (WAN fused attention V1 ops).

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.runtime.kernels.wan_fused_v1.setup` (module): Setup script configuring `CUDAExtension` sources/flags for `wan_fused_v1_cuda`.
"""

from __future__ import annotations

import os
import shutil

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(__file__)

if torch.version.cuda is None:
    raise RuntimeError(
        "Cannot build `wan_fused_v1_cuda`: PyTorch is CPU-only in this environment "
        f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda}). "
        "Install a CUDA-enabled PyTorch build and retry."
    )

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot build `wan_fused_v1_cuda`: CUDA toolkit not detected (CUDA_HOME is None). "
        "Install the CUDA toolkit and ensure `nvcc` is available, then retry."
    )

_nvcc = shutil.which("nvcc")
if _nvcc is None:
    nvcc_candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
    if os.path.isfile(nvcc_candidate):
        _nvcc = nvcc_candidate
if _nvcc is None:
    raise RuntimeError(
        "Cannot build `wan_fused_v1_cuda`: `nvcc` not found. "
        f"CUDA_HOME={CUDA_HOME!r}. Ensure `{os.path.join(CUDA_HOME, 'bin')}` is on PATH, then retry."
    )

_cuda_arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST")
_cuda_arch_list_selected = (
    _cuda_arch_list_env.strip() if _cuda_arch_list_env and _cuda_arch_list_env.strip() else "<torch-default>"
)
_cuda_arch_list_source = "env" if _cuda_arch_list_selected != "<torch-default>" else "torch_default"
print(
    "[wan_fused_v1.build] "
    f"cuda_arch_list={_cuda_arch_list_selected} cuda_arch_source={_cuda_arch_list_source}"
)
print(
    "[wan_fused_v1.build] "
    "attn_core_default_force=cuda_experimental attn_core_default_other=aten "
    "attn_core_env=CODEX_WAN_FUSED_V1_ATTN_CORE"
)

sources = [
    os.path.join(this_dir, "wan_fused_v1_binding.cpp"),
    os.path.join(this_dir, "wan_fused_v1_kernels.cu"),
]

ext_modules = [
    CUDAExtension(
        name="wan_fused_v1_cuda",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-DUSE_CUDA"],
        },
    )
]

setup(
    name="wan_fused_v1_cuda",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
