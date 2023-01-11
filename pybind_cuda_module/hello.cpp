#include <torch/extension.h>

void hello_cuda(torch::Tensor input);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void hello(torch::Tensor input) {
   CHECK_INPUT(input);

   return hello_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("hello", &hello);
}
