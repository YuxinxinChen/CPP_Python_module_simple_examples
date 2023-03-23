// #include <ATen/ATen.h>
// #include <ATen/core/Tensor.h>
// #include <ATen/Utils.h>
#include <tuple>

#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>

//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAEvent.h>
//#include <c10/core/Event.h>
//#include <c10/core/impl/InlineEvent.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <c10/cuda/impl/CUDAGuardImpl.h>
//#include <c10/util/irange.h>

#include <cuda_runtime.h>

#include "nvshmem.h"
#include "nvshmemx.h"
#include "mpi.h"

//#include <functional>
//#include <future>
//#include <thread>
//#include <unordered_set>

#define CUDA_CHECK(call) {                                    \
          cudaError_t err =                                                         call;                                                    \
          if( cudaSuccess != err) {                                                 \
                       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",         \
                                                __FILE__, __LINE__, cudaGetErrorString( err) );               \
                       exit(EXIT_FAILURE);                                                   \
                   } }

__global__ void simple_shift(int *destination) {
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int peer = (mype + 1) % npes;
    printf("in kernel my_pe %d, npes %d, peer %d\n", mype, npes, peer);

    nvshmem_int_p(destination, mype, peer);
}

__global__ void spin(int *flag) {
  volatile int * flagg = (volatile int *)flag;
  while((*flagg) == 1) {}
}

__global__ void signal(int *flag, int peer) {
  int mype = nvshmem_my_pe();
  printf("pe %d about to signal %d to stop\n", mype, peer);
  nvshmem_int_p(flag, 0, peer);
}

__device__ int clockrate;
std::tuple<int64_t, int64_t> nvshmem_mpi_init()
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

    int64_t my_pe = nvshmem_my_pe();
    int64_t n_pes = nvshmem_n_pes();
    assert(my_pe == rank);
    assert(n_pes == nranks);

    cudaDeviceProp prop;
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count)); 
    CUDA_CHECK(cudaSetDevice(my_pe%dev_count)); 
    CUDA_CHECK(cudaGetDeviceProperties(&prop, my_pe%dev_count))
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate,sizeof(int), 0, cudaMemcpyHostToDevice));

    printf("[%d of %d] has GPUs %d, set on %d, device name %s, bus id %d, clockrate %d\n", int(my_pe), int(n_pes), dev_count, int(my_pe)%dev_count, prop.name, prop.pciBusID, prop.clockRate);

    

    

    return std::make_tuple(my_pe, n_pes);
}

torch::Tensor nvshmem_alloc()
{
  int my_pe = nvshmem_my_pe();
  int check_device = -1;
  CUDA_CHECK(cudaGetDevice(&check_device));
  assert(check_device == my_pe);
  torch::Device device(torch::kCUDA, my_pe);
  int * test_ptr = (int *)nvshmem_malloc(sizeof(int));
  int test_value = 2;
  if(my_pe == 0) test_value = 12;
  CUDA_CHECK(cudaMemcpy(test_ptr, &test_value, sizeof(int), cudaMemcpyHostToDevice));
  int check = 1;
  CUDA_CHECK(cudaMemcpy(&check, test_ptr, sizeof(int), cudaMemcpyDeviceToHost));
  printf("pe %d, check alloc %d\n", int(my_pe), check);

  auto options = torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(device);
  torch::Tensor tensor = torch::from_blob(test_ptr, {1, 1}, options);
  printf("pe %d, test_ptr %p\n", int(my_pe), test_ptr);
  return tensor;
}

void nvshmem_check(torch::Tensor input)
{
  int my_pe = nvshmem_my_pe();
  int * test_ptr = input.data_ptr<int>();
  printf("pe %d, test_ptr %p\n", int(my_pe), test_ptr);
  int check = 1;
  CUDA_CHECK(cudaMemcpy(&check, test_ptr, sizeof(int), cudaMemcpyDeviceToHost));

  nvshmem_barrier_all();
  printf("pe %d, check check alloc %d\n", int(my_pe), check);
  nvshmem_barrier_all();

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  simple_shift<<<1, 1, 0, stream>>>(test_ptr);
  nvshmemx_barrier_all_on_stream(stream);
  cudaMemcpyAsync(&check, test_ptr, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  printf("pe %d, check alloc again %d\n", int(my_pe), check);
}

void nvshmem_spin(torch::Tensor input, cudaStream_t stream)
{
  //cudaStream_t stream = std::reinterpret_cast<cudaStream_t>(stream_ptr);
  //cudaStream_t stream; // = cudaStream_t(py_stream);
  int my_pe = nvshmem_my_pe();
  int * test_ptr = input.data_ptr<int>();
  printf("pe %d, test_ptr %p\n", int(my_pe), test_ptr);  
  printf("pe %d about to spin\n", my_pe);

  spin<<<1,1,0, stream>>>(test_ptr);
  //CUDA_CHECK(cudaStreamSynchronize(stream));
  //printf("pe %d, spin exit\n", my_pe);
}

void nvshmem_signal(torch::Tensor input, cudaStream_t stream)
{
  //cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  //cudaStream_t stream; // = cudaStream_t(py_stream);
  //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int my_pe = nvshmem_my_pe();
  int * test_ptr = input.data_ptr<int>();
  printf("pe %d, test_ptr %p\n", int(my_pe), test_ptr);  
  printf("pe %d, about to signal spin to stop\n", my_pe);
  signal<<<1,1,0, stream>>>(test_ptr, 0);
  //CUDA_CHECK(cudaStreamSynchronize(stream));
  //printf("pe %d, signaled\n", my_pe);
}

void nvshmem_spin_signal(torch::Tensor input)
{
  cudaStream_t stream_spin;
  cudaStream_t stream_signal;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_spin, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_signal, cudaStreamNonBlocking));
  int my_pe = nvshmem_my_pe();
  if(my_pe == 0 ) nvshmem_spin(input, stream_spin);
  else if(my_pe == 1) nvshmem_signal(input, stream_signal); 
  
  if(my_pe == 1) {
    CUDA_CHECK(cudaStreamSynchronize(stream_signal));  
    printf("pe %d, signaled\n", my_pe);
  }
  if(my_pe == 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream_spin));
    printf("pe %d, spin exit\n", my_pe);
  }
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
  m.def("nvshmem_alloc", &nvshmem_alloc);
  m.def("nvshmem_check", & nvshmem_check);
  m.def("nvshmem_spin_signal", &nvshmem_spin_signal);
}
