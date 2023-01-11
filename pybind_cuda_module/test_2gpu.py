import torch
import sys

sys.path.append("./build/lib.linux-x86_64-cpython-310")
import hello_cuda

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')

a = torch.tensor([1,2])
b = torch.tensor([4,5])

a_cuda = a.to(cuda0)
b_cuda = b.to(cuda1)
torch.cuda.synchronize()
hello_cuda.hello(a_cuda)
torch.cuda.synchronize()
hello_cuda.hello(b_cuda)
torch.cuda.synchronize()
print("exit python file")
