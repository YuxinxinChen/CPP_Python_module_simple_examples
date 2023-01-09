### To compile the CPP module
```
python setup.py build --build-base=/home/yuxin420/tmp/pybind_hello_module/build
```

### To use the compiled module
```
cd build/lib.linux-x86_64-cpython-310/
python
>>> import torch
>>> import hello_cpp_extension
>>> hello_cpp_extension.print()
Hello
True
>>> 
```