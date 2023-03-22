// #include <ATen/ATen.h>
// #include <ATen/core/Tensor.h>
// #include <ATen/Utils.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "mpi.h"

#define CUDA_CHECK(call) {                                    \
          cudaError_t err =                                                         call;                                                    \
          if( cudaSuccess != err) {                                                 \
                       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",         \
                                                __FILE__, __LINE__, cudaGetErrorString( err) );               \
                       exit(EXIT_FAILURE);                                                   \
                   } }

__device__ int clockrate;
void nvshmem_mpi_init()
{
    int rank;
    int nranks;
    //MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    CUDA_CHECK(cudaSetDevice(rank));
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    assert(my_pe == rank);
    assert(n_pes == nranks);

    cudaDeviceProp prop;
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count)); 
    CUDA_CHECK(cudaSetDevice(my_pe%dev_count)); 
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_pe%dev_count))
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate,sizeof(int), 0, cudaMemcpyHostToDevice));

    printf("[%d of %d] has GPUs %d, set on %d, device name %s, bus id %d, clockrate %d\n", my_pe, n_pes, dev_count, my_pe%dev_count, prop.name, prop.pciBusID, prop.clockRate);
}

void nvshm_mpi_finalize()
{
    nvshmem_barrier_all();
    nvshmem_finalize();
    //MPI_Finalize();
}

using namespace at;


int64_t integer_round(int64_t num, int64_t denom){
  return (num + denom - 1) / denom;
}


template<class T>
__global__ void add_one_kernel(const T *const input, T *const output, const int64_t N){
  // Grid-strided loop
  for(int i=blockDim.x*blockIdx.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x){
    output[i] = input[i] + 1;
  }
}


Tensor add_one(const Tensor &input){
  auto output = torch::zeros_like(input);

  AT_DISPATCH_ALL_TYPES(
    input.scalar_type(), "add_one_cuda", [&](){
      const auto block_size = 128;
      const auto num_blocks = std::min(65535L, integer_round(input.numel(), block_size));
      add_one_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        input.numel()
      );
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  );

  return output;
}


TORCH_LIBRARY(nvshmem_example, m) {
  m.def("add_one(Tensor input) -> Tensor");
  m.impl("add_one", c10::DispatchKey::CUDA, TORCH_FN(add_one));

  m.def("nvshmem_mpi_init", &nvshmem_mpi_init);
  m.def("nvshmem_mpi_finalize", &nvshm_mpi_finalize);
}
