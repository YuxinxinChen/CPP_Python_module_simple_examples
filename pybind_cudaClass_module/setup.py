from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_cpp_class',
    ext_modules=[
        CUDAExtension('my_cpp_class', [
            'my_cpp_class.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

