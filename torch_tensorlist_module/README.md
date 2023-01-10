### To compile the .cu file

```
python setup.py build --build-base=/home/yuxin420/tmp/torch_tensorlist_module/build
```

### To use the module

```
cd build/lib.linux-x86_64-cpython-310/
python
>>> import torch
>>> import tensorlist_cu_extension.tensorlist
>>> a=[]
>>> a.append(torch.tensor([0,1]))
>>> a.append(torch.tensor([2,3]))
>>> torch.ops.tensorlist.print(a)
 TensorList
True
```