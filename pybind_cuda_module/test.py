import torch
import sys

sys.path.append("./build/lib.linux-x86_64-cpython-310")
import hello_cuda

cuda_device = torch.device("cuda")
a = torch.tensor([1,2], device=cuda_device)
hello_cuda.hello(a)
torch.cuda.synchronize()
print("exit python file")
