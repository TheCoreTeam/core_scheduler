#pragma once
#include <future>

#include "threading/context_compute.h"
#include "threading/task_future.h"

namespace dllm {
using TaskCompute = std::packaged_task<void(const ContextCompute *)>;
}  // namespace dllm
