from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='hello_cpp_extension',
    ext_modules=[
        cpp_extension.CppExtension(
            'hello_cpp_extension.hello',
            ['hello.cpp']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=1)}
)