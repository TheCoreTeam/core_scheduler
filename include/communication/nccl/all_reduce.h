#pragma once
#include "communication/all_reduce.h"
#include "tensor.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <>
struct AllReduce<NCCL> {
  static TaskNccl run(const std::shared_ptr<Tensor> &tensorReceive,
                      const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                      Operation operation);

  static TaskNccl runInplace(const std::shared_ptr<Tensor> &tensor,
                             Operation operation);
};
}  // namespace dllm::communication
