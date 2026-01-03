"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Build script for the `wan_te_cuda` CUDA extension (WAN T5 FP8 encoder kernels).

Symbols (top-level; keep in sync; no ghosts):
- (none): Setup script configuring `CUDAExtension` sources/flags for `wan_te_cuda`.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

this_dir = os.path.dirname(__file__)

sources = [
    os.path.join(this_dir, 'te_binding.cpp'),
    os.path.join(this_dir, 'te_attention_fp8.cu'),
    os.path.join(this_dir, 'te_attention_fp8_kernel.cu'),
    os.path.join(this_dir, 'te_linear_fp8.cu'),
]

ext_modules = [
    CUDAExtension(
        name='wan_te_cuda',
        sources=sources,
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math']
        }
    )
]

setup(
    name='wan_te_cuda',
    version='0.1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
