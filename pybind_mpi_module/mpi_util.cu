#include <torch/extension.h>
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

void mpi_init()
{
    std::tuple<MPI_Comm*, int, int> meta = mpi_comm();
    int rank = std::get<2>(meta);
    int nranks = std::get<1>(meta);
    printf("rank %d, nranks %d\n", rank, nranks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("init", &mpi_init);
}

