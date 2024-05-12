#pragma once
#include <future>

#include "threading/context_compute.h"

namespace dllm {
using TaskCompute = std::packaged_task<void(const ContextCompute *)>;
using TaskFuture = std::shared_future<void>;
}  // namespace dllm
