from my_py_class import MyPyClass

# Create Python object wrapping the C++ object
obj = MyPyClass()

# Launch the CUDA kernel through the Python wrapper
obj.launch_add_one_kernel()

# Get data from the device and print it
data = obj.get_data()
print("Data:", data)

obj.print_ptr(data)

