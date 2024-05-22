#pragma once
#include "logger.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::memory {
Tensor1D::DataPtr mallocFromMemPool(size_t size, Dtype dtype,
                                    const ContextCompute *context);
}  // namespace dllm::memory
