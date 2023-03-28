#include <iostream>
#include <torch/extension.h>

class MyClass {
public:
    MyClass() : data(0) {}

    void set_data(int value) {
        data = value;
    }

    int get_data() {
        return data;
    }

private:
    int data;
};

// PyTorch binding functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("set_data", &MyClass::set_data)
        .def("get_data", &MyClass::get_data);
}

