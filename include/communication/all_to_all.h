#pragma once
#include "tensor.h"
#include "threading/task_mpi.h"

namespace dllm::communication {
struct AllToAll {
  static TaskMpi run(const std::shared_ptr<Tensor> &tensorReceive,
                     const std::shared_ptr<const ReadOnlyTensor> &tensorSend);

  static TaskMpi runInplace(const std::shared_ptr<Tensor> &tensor);
};
}  // namespace dllm::communication
