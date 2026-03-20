"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Build script for the generic `attention_sram_v1_cuda` CUDA extension.
Provides the local `CUDAExtension` build used by `runtime/attention/sram` loader paths.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.runtime.kernels.attention_sram_v1.setup` (module): Setup script configuring `CUDAExtension` sources and compile flags for `attention_sram_v1_cuda`.
"""

from __future__ import annotations

import os
import shutil

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

THIS_DIR = os.path.dirname(__file__)
MODULE_NAME = "attention_sram_v1_cuda"

if torch.version.cuda is None:
    raise RuntimeError(
        f"Cannot build `{MODULE_NAME}`: PyTorch is CPU-only in this environment "
        f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda}). "
        "Install a CUDA-enabled PyTorch build and retry."
    )

if CUDA_HOME is None:
    raise RuntimeError(
        f"Cannot build `{MODULE_NAME}`: CUDA toolkit not detected (CUDA_HOME is None). "
        "Install the CUDA toolkit and ensure `nvcc` is available, then retry."
    )

nvcc_path = shutil.which("nvcc")
if nvcc_path is None:
    nvcc_candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
    if os.path.isfile(nvcc_candidate):
        nvcc_path = nvcc_candidate
if nvcc_path is None:
    raise RuntimeError(
        f"Cannot build `{MODULE_NAME}`: `nvcc` not found. "
        f"CUDA_HOME={CUDA_HOME!r}. Ensure `{os.path.join(CUDA_HOME, 'bin')}` is on PATH, then retry."
    )

cuda_arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST")
cuda_arch_list_selected = cuda_arch_list_env.strip() if cuda_arch_list_env and cuda_arch_list_env.strip() else "<torch-default>"
cuda_arch_list_source = "env" if cuda_arch_list_selected != "<torch-default>" else "torch_default"
print(
    "[attention_sram_v1.build] "
    f"cuda_arch_list={cuda_arch_list_selected} cuda_arch_source={cuda_arch_list_source}"
)

sources = [
    os.path.join(THIS_DIR, "attention_sram_v1_binding.cpp"),
    os.path.join(THIS_DIR, "attention_sram_v1_kernels.cu"),
]

ext_modules = [
    CUDAExtension(
        name=MODULE_NAME,
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
        },
    )
]

setup(
    name=MODULE_NAME,
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
