#pragma once
#include <future>

#include "threading/context_cudart.h"
#include "threading/task_future.h"

namespace dllm {
using TaskCudart = std::packaged_task<void(const ContextCudart *)>;
}  // namespace dllm
