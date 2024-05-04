#pragma once
#include "communication/all_reduce.h"
#include "tensor.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <>
struct AllReduce<NCCL> {
  static TaskNccl run(const std::shared_ptr<const Tensor1D> &tensorSend,
                      const std::shared_ptr<Tensor1D> &tensorReceive,
                      Operation operation);

  static TaskNccl runInplace(const std::shared_ptr<Tensor1D> &tensor,
                             Operation operation);
};
}  // namespace dllm::communication
