#pragma once
#include <mpi.h>

namespace dllm {
struct ContextMpi {
  int mpiRank;
  int commSize;
  MPI_Comm mpiComm;
};
}  // namespace dllm