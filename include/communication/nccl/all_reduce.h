#pragma once
#include "communication/all_reduce.h"
#include "dtensor.h"
#include "threading/task_nccl.h"

namespace dllm::communication {
template <>
struct AllReduce<NCCL> {
  static TaskNccl run(const std::shared_ptr<const DTensor1D<NCCL>> &tensorSend,
                      const std::shared_ptr<DTensor1D<NCCL>> &tensorReceive,
                      Operation operation);

  static TaskNccl runInplace(const std::shared_ptr<DTensor1D<NCCL>> &tensor,
                             Operation operation);
};
}  // namespace dllm::communication
