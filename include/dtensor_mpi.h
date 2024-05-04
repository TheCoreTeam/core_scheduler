#pragma once
#include <mpi.h>

#include "communication.h"

namespace dllm::util {
constexpr MPI_Op toMpiOp(communication::Operation operation) {
  switch (operation) {
    case communication::SUM:
      return MPI_SUM;
    default:
      return 0;
  }
}
}  // namespace dllm::util
