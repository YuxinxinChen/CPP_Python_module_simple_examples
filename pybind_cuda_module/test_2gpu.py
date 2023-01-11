import torch
import sys

sys.path.append("./build/lib.linux-x86_64-cpython-310")
import hello_cuda

num_of_device = torch.cuda.device_count()
print("number of devices: %d" % num_of_device)
cudaDevice = []
for i in range(num_of_device):
    cudaDevice.append(torch.device(i))

a = torch.tensor([1,2])
b = torch.tensor([4,5])

a_cuda = a.to(cudaDevice[0])
b_cuda = b.to(cudaDevice[1])
torch.cuda.synchronize()
hello_cuda.hello(a_cuda)
torch.cuda.synchronize()
hello_cuda.hello(b_cuda)
torch.cuda.synchronize()
print("exit python file")
