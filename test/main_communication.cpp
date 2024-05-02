#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  RUN_ALL_TESTS();
  MPI_Finalize();
}
