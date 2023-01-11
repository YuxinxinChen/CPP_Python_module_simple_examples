from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='hello',
    ext_modules=[
        CUDAExtension('hello_cuda', ['hello.cpp', 'hello_cuda.cu',])
    ],
    cmdclass={'build_ext': BuildExtension}
)
