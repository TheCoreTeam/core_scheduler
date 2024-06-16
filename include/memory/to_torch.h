#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::memory {
// dst is not ready immediately! you should wait for src
at::Tensor toTorch(const Scheduler &scheduler, const ReadOnlyTensor &src);
}  // namespace dllm::memory
