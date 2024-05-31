#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>

#include "logger.h"
#include "threading/thread_stream_nccl.h"

namespace dllm::test {
ncclUniqueId &getNcclUniqueId() {
  static ncclUniqueId id = [] {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    return id;
  }();
  return id;
}

ThreadStreamNccl *getNcclStream() {
  struct StreamWrapper {
    MPI_Comm comm;

    ThreadStreamNccl *stream;
    StreamWrapper() {
      int processesPerNode;
      int rank;
      CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                    MPI_INFO_NULL, &comm));
      CHECK_MPI(MPI_Comm_size(comm, &processesPerNode));
      CHECK_MPI(MPI_Comm_rank(comm, &rank));
      ncclUniqueId id;
      if (rank == 0) {
        id = getNcclUniqueId();
      }
      CHECK_MPI(MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, comm));
      stream = new dllm::ThreadStreamNccl{id, processesPerNode, rank, 0};
    }

    ~StreamWrapper() {
      delete stream;
      CHECK_MPI(MPI_Comm_free(&comm));
    }
  };
  static StreamWrapper stream_wrapper{};
  return stream_wrapper.stream;
}
}  // namespace dllm::test

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  ::testing::InitGoogleTest(&argc, argv);
  const auto error = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return error;
}
