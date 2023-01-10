import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = []

if torch.cuda.is_available():
    extension = CUDAExtension(
        'tensorlist_cu_extension.tensorlist', 
        [ 'tensorlist.cu'],
        extra_compile_args={'nvcc': ['-O2']}
    )
    ext_modules.append(extension)

setup(
    name='tensorlist_cu_extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=1)}
)