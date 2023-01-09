### To compile the C++ module
```
python setup.py build --build-base=/home/yuxin420/tmp/torch_hello_module/build
// --build-base pointing to the build path
```

To test the compile module, go the lib folder
```
>>> import torch
>>> import hello_cpp_extension.hello
>>> torch.ops.hello.print()
Hello
True
```