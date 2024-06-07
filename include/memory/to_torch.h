#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::memory {
// dst is not ready immediately! you should wait for src
void toTorch(const Scheduler &scheduler, at::Tensor &dst,
             const std::shared_ptr<const ReadOnlyTensor> &src);
}  // namespace dllm::memory
