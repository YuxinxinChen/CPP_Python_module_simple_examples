#include <torch/extension.h>
#include "nvshmem.h"
#include "nvshmemx.h"
#include "mpi.h"

static std::tuple<MPI_Comm*, int, int> mpi_comm() {
  static std::once_flag once;
  static MPI_Comm world_comm;
  static int world_size;
  static int world_rank;

  std::call_once(once, [&] {
    {
      printf("MPI_COMM CALLED\n");
      auto op = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
      AT_ASSERT(op == MPI_SUCCESS);
    }
    {
      auto op = MPI_Comm_size(world_comm, &world_size);
      AT_ASSERT(op == MPI_SUCCESS);
    }
    {
      auto op = MPI_Comm_rank(world_comm, &world_rank);
      AT_ASSERT(op == MPI_SUCCESS);
    }
  });
  return std::make_tuple(&world_comm, world_size, world_rank);
}

void nvshmem_mpi_init()
{
    std::tuple<MPI_Comm*, int, int> meta = mpi_comm();
    int rank = std::get<2>(meta);
    int nranks = std::get<1>(meta);
    printf("rank %d, nranks %d\n", rank, nranks);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int my_pe = nvshmem_my_pe();
    int n_pes = nvshmem_n_pes();
    assert(my_pe == rank);
    assert(n_pes == nranks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("init", &nvshmem_mpi_init);
}

