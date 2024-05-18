#pragma once
#include "communication/all_gather.h"
#include "tensor.h"
#include "threading/task_mpi.h"

namespace dllm::communication {
template <>
struct AllGather<MPI> {
  static TaskMpi run(const std::shared_ptr<const Tensor1D> &tensorSend,
                     const std::shared_ptr<Tensor1D> &tensorReceive);

  static TaskMpi runInplace(const std::shared_ptr<Tensor1D> &tensor);
};
}  // namespace dllm::communication
