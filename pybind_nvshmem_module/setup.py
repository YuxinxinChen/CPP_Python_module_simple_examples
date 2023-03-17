from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

MPI_HOME = os.getenv('MPI_HOME')
mpi_include_path = MPI_HOME+'/include'
NVSHMEM_HOME = os.getenv('NVSHMEM_HOME')
nvshmem_include_path = NVSHMEM_HOME+'/include'
nvshmem_lib_path = NVSHMEM_HOME+'/lib'

setup(
    name='nvshmem',
    ext_modules=[
        CUDAExtension('nvshm', ['nvshmem_util.cu',],
        include_dirs=[mpi_include_path, nvshmem_include_path],
        libraries=['nvshmem', 'nvidia-ml', 'cudart', 'cuda',],
        library_dirs=[nvshmem_lib_path],
        extra_compile_args={'nvcc': ['-rdc=true','-gencode=arch=compute_70,code=sm_70','-ccbin', 'g++']},
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
