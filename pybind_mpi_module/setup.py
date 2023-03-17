from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

MPI_HOME = os.getenv('MPI_HOME')
mpi_include_path = MPI_HOME+'/include'

setup(
    name='mpi_util',
    ext_modules=[
        CUDAExtension('mpi', ['mpi_util.cu',],
        include_dirs=[mpi_include_path],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
