#pragma once
#include <mpi.h>

#include "dtensor.h"

namespace dllm {
template <int N>
struct DTensor<N, communication::MPI> : public Tensor<N> {
  using Base = Tensor<N>;
  int rank;
  MPI_Comm comm;

  DTensor(int rank, MPI_Comm comm, const Base &tensor)
      : Base{tensor}, rank{rank}, comm{comm} {}
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
