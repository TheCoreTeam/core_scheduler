#include <gtest/gtest.h>
#include <mpi.h>

#include "logger.h"

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  ::testing::InitGoogleTest(&argc, argv);
  const auto error = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return error;
}
