#pragma once
#include "communication/all_reduce.h"
#include "tensor.h"
#include "threading/task_mpi.h"

namespace dllm::communication {
template <>
struct AllReduce<MPI> {
  static TaskMpi run(const std::shared_ptr<const Tensor1D> &tensorSend,
                     const std::shared_ptr<Tensor1D> &tensorReceive,
                     Operation operation);

  static TaskMpi runInplace(const std::shared_ptr<Tensor1D> &tensor,
                            Operation operation);
};
}  // namespace dllm::communication
