#include "nvshmem.h"
#include "nvshmemx.h"
#include "mpi.h"

void nvshmem_mpi_init(int *argc, char ***argv)
{
    int rank;
    int nranks;
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
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

int main(int argc, char *argv[])
{
    nvshmem_mpi_init(&argc, &argv);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
