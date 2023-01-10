#include <torch/extension.h>
#include <iostream>

bool print(torch::TensorList x) {
    std::cout << " TensorList" << std::endl;
    return true;
}

TORCH_LIBRARY(tensorlist, m) {
    m.def("print", &print);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}