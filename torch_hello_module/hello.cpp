#include <torch/extension.h>
#include <iostream>

bool print() {
    std::cout << "Hello" << std::endl;
    return true;
}

TORCH_LIBRARY(hello, m) {
    m.def("print", &print);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}