#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/task_mpi.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <Backend backend>
struct AllReduce;

template <>
struct AllReduce<NCCL> {
  static TaskNccl runInplace(const std::shared_ptr<Tensor> &tensor,
                             Operation operation);

  static TaskNccl runInplace(
      const std::vector<std::shared_ptr<Tensor>> &tensors,
      Operation operation);
};
}  // namespace dllm::communication
