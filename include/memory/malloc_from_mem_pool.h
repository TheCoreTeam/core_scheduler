#pragma once
#include "logger.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::memory {
void mallocFromMemPool(const std::shared_ptr<Tensor1D> &x,
                       cudaStream_t stream = nullptr);
}  // namespace dllm::alloc
