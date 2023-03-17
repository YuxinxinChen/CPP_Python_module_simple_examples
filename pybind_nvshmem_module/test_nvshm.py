import torch
import sys
import os
import socket

import torch.distributed as dist

sys.path.append("./build/lib.linux-x86_64-cpython-310")
import nvshm

global world_size
global world_rank
world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
hostname = socket.gethostname()
dist.init_process_group('mpi', rank=world_rank, world_size=world_size)

nvshm.init();
