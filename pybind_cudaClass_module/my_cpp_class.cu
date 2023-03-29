#include <iostream>
#include <torch/extension.h>

// CUDA kernel function
__global__ void add_one_kernel(int *data) {
    int idx = threadIdx.x;
    data[idx] += 1;
}

class MyClass {
public:
    MyClass():host_data(10){
        cudaMalloc(&data, sizeof(int) * 10);
        cudaMemset(data, 0, sizeof(int) * 10);
    }

    ~MyClass() {
        cudaFree(data);
    }

    void launch_add_one_kernel() {
        add_one_kernel<<<1, 10>>>(data);
        cudaDeviceSynchronize();
    }

    at::Tensor get_data() {
        cudaMemcpy(host_data.data(), data, sizeof(int) * 10, cudaMemcpyDeviceToHost);
        for(int i=0; i<host_data.size(); i++)
            printf("%d ", host_data[i]);
        printf("\n");
        printf("host data ptr %p\n", host_data.data());
        return torch::from_blob(host_data.data(), {10}, torch::kInt);
    }

    void print_ptr(at::Tensor tensor) {
        printf("print ptr from tensor: %p\n", tensor.data_ptr<int>());
    }

private:
    int *data;
    std::vector<int> host_data;
};

// PyTorch binding functions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<>())
        .def("launch_add_one_kernel", &MyClass::launch_add_one_kernel)
        .def("get_data", &MyClass::get_data)
        .def("print_ptr", &MyClass::print_ptr);
}

