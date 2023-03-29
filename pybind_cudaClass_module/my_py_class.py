import sys
import os
import ctypes
import torch
import numpy as np
from torch.utils.cpp_extension import load

build_directory = os.path.join(".", "build", "lib.linux-x86_64-cpython-310")
sys.path.insert(0, build_directory)
# Compile and load C++ extension
my_cpp_class_module = load(name='my_cpp_class', sources=['my_cpp_class.cu'], is_python_module=True)

class MyPyClass:
    def __init__(self):
        self._cpp_obj = my_cpp_class_module.MyClass()

    def launch_add_one_kernel(self):
        self._cpp_obj.launch_add_one_kernel()

    def get_data(self):
        return self._cpp_obj.get_data() 

    def print_ptr(self, tensor):
        return self._cpp_obj.print_ptr(tensor)

