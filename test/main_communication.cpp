#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>

#include "logger.h"

namespace dllm::test {
ncclUniqueId &getUniqueNcclId() {
  static ncclUniqueId id = [] {
    ncclUniqueId id;
    CHECK_NCCL(ncclGetUniqueId(&id));
    return id;
  }();
  return id;
}
}  // namespace dllm::test

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  ::testing::InitGoogleTest(&argc, argv);
  const auto error = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return error;
}
