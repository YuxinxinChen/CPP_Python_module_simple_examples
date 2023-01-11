#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
//__global__ void hello_cuda_kernel(const scalar_t * __restrict__ input)
__global__ void hello_cuda_kernel(
  const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> input) {
  if(threadIdx.x == 0) {
    printf("hello from tensor:\n");
    for(int i=0; i<input.size(0); i++)
      printf("%d, ", input[i]);
    printf("finish!\n");
  }
}

void hello_cuda(torch::Tensor input) {
   const int threads=32;
   const int blocks=1;

   int device_id = input.get_device();
   printf("device_id %d\n", device_id);
   cudaSetDevice(device_id);
   AT_DISPATCH_ALL_TYPES(input.type(), "hello_cuda", 
     ([&] {
        hello_cuda_kernel<scalar_t><<<blocks, threads>>>
	//(input.data<scalar_t>());
	(input.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
   }));
}
