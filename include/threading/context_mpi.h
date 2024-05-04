#pragma once
#include <mpi.h>

namespace dllm {
struct ContextMpi {
  int mpiRank;
  MPI_Comm mpiComm;
};
}  // namespace dllm