from py_class import MyPyClass

# Create Python object wrapping the C++ object
obj = MyPyClass()

# Use the C++ object through the Python wrapper
obj.set_data(42)
print("Data:", obj.get_data())

