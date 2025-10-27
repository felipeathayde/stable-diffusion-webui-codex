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
