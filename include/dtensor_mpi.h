#pragma once
#include <mpi.h>

#include "dtensor.h"

namespace dllm {
template <int N>
struct DTensor<N, communication::MPI> : public Tensor<N> {
  int rank, lrank;
  MPI_Comm comm;
};

namespace communication {
constexpr MPI_Op toMpiOp(Operation operation) {
  switch (operation) {
    case SUM:
      return MPI_SUM;
    default:
      return 0;
  }
}
}  // namespace communication
}  // namespace dllm
