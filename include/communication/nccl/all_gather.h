#pragma once
#include "communication/all_gather.h"
#include "tensor.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <>
struct AllGather<NCCL> {
  static TaskNccl run(const std::shared_ptr<Tensor> &tensorReceive,
                      const std::shared_ptr<const ReadOnlyTensor> &tensorSend,
                      int64_t receiveCount);

  static TaskNccl runInplace(const std::shared_ptr<Tensor> &tensor);
};
}  // namespace dllm::communication
