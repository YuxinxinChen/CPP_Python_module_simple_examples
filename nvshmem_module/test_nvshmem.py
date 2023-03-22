import torch
import sys
import os
import socket

import torch.distributed as dist
torch.ops.load_library("build/libnvshmem_example.so")

shape = (3,3,3)
a = torch.randint(0, 10, shape, dtype=torch.float).cuda()
a_plus_one = torch.ops.nvshmem_example.add_one(a)
print(a_plus_one)

global world_size
global world_rank
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
hostname = socket.gethostname()
dist.init_process_group('mpi', rank=world_rank, world_size=world_size)
torch.ops.nvshmem_example.nvshmem_mpi_init()
torch.ops.nvshmem_example.nvshmem_mpi_finalize()
