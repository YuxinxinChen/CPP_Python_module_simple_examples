### To compile the C++ custom operator
```
mkdir build
cd build
cmake ..
make -j
```

### To use the custom operator
```
cd build
python
>>> import torch
>>> torch.ops.load_library("libwarp_perspective.so")
>>> torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3))
tensor([[-0.2021,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0503, -0.0956, -0.1095,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0159, -0.0269, -0.0501, -0.1842, -1.0084,  0.0000,  0.0000,  0.0000],
        [ 0.0000, -0.0055, -0.0112, -0.0456, -0.1551, -0.6661, -0.3043,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.0091, -0.0285, -0.0981, -0.3194, -0.7269],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.0525, -0.1528],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000]])
```