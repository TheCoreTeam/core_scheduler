#pragma once
#include "tensor.h"
#include "threading/task_cudart.h"

namespace dllm::memory {
// dst is not ready immediately! you should wait for src
TaskCudart toTorch(at::Tensor &dst,
                   const std::shared_ptr<const ReadOnlyTensor> &src);
}  // namespace dllm::memory
