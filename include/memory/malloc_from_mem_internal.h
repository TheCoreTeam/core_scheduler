#pragma once
#include "logger.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::memory {
std::shared_ptr<void> mallocFromMemPool(size_t size, Dtype dtype,
                                        const ContextCompute *context);
}  // namespace dllm::alloc
