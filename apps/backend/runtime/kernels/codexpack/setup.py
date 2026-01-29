"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Build script for the `codexpack_cuda` CUDA extension (CodexPack packed GGUF kernels).
Builds and installs a local CUDA extension that registers `torch.ops.codexpack.q4k_tilepack_linear(...)`.

Symbols (top-level; keep in sync; no ghosts):
- `apps.backend.runtime.kernels.codexpack.setup` (module): Setup script configuring `CUDAExtension` sources/flags for `codexpack_cuda`.
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
        "Cannot build `codexpack_cuda`: PyTorch is CPU-only in this environment "
        f"(torch={torch.__version__}, torch.version.cuda={torch.version.cuda}). "
        "Install a CUDA-enabled PyTorch build and retry."
    )

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot build `codexpack_cuda`: CUDA toolkit not detected (CUDA_HOME is None). "
        "Install the CUDA toolkit and ensure `nvcc` is available, then retry."
    )

_nvcc = shutil.which("nvcc")
if _nvcc is None:
    nvcc_candidate = os.path.join(CUDA_HOME, "bin", "nvcc")
    if os.path.isfile(nvcc_candidate):
        _nvcc = nvcc_candidate
if _nvcc is None:
    raise RuntimeError(
        "Cannot build `codexpack_cuda`: `nvcc` not found. "
        f"CUDA_HOME={CUDA_HOME!r}. Ensure `{os.path.join(CUDA_HOME, 'bin')}` is on PATH, then retry."
    )

sources = [
    os.path.join(this_dir, "codexpack_binding.cpp"),
    os.path.join(this_dir, "q4k_tilepack_linear.cu"),
]

ext_modules = [
    CUDAExtension(
        name="codexpack_cuda",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
    )
]

setup(
    name="codexpack_cuda",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
