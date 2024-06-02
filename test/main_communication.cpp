#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <netdb.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

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

auto &getSingleton() {
  struct StreamWrapper {
    const MPI_Comm comm = MPI_COMM_WORLD;

    ThreadStreamNccl *stream;
    StreamWrapper() { stream = new ThreadStreamNccl{comm, 0}; }

    ~StreamWrapper() { delete stream; }
  };
  static auto stream_wrapper = new StreamWrapper;
  return stream_wrapper;
}

ThreadStreamNccl *getNcclStream() { return getSingleton()->stream; }
}  // namespace dllm::test

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  dllm::test::getSingleton();
  ::testing::InitGoogleTest(&argc, argv);
  const auto error = RUN_ALL_TESTS();
  delete dllm::test::getSingleton();
  dllm::test::getSingleton() = nullptr;
  CHECK_MPI(MPI_Finalize());
  return error;
}
