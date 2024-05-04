#pragma once
#include "communication/all_reduce.h"
#include "dtensor.h"
#include "threading/task_mpi.h"

namespace dllm::communication {
template <>
struct AllReduce<MPI> {
  static TaskMpi run(const std::shared_ptr<const DTensor1D<MPI>> &tensorSend,
                     const std::shared_ptr<DTensor1D<MPI>> &tensorReceive,
                     Operation operation);

  static TaskMpi runInplace(const std::shared_ptr<DTensor1D<MPI>> &tensor,
                            Operation operation);
};
}  // namespace dllm::communication
