#include <torch/extension.h>
#include <iostream>

bool print() {
    std::cout << "Hello" << std::endl;
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("print", &print);
}