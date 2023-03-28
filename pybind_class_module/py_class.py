import torch
from torch.utils.cpp_extension import load

# Compile and load C++ extension
cpp_class_module = load(name='cpp_class', sources=['cpp_class.cpp'])

class MyPyClass:
    def __init__(self):
        self._cpp_obj = cpp_class_module.MyClass()

    def set_data(self, value):
        self._cpp_obj.set_data(value)

    def get_data(self):
        return self._cpp_obj.get_data()

