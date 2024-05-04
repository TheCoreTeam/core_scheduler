#pragma once
#include <future>

#include "threading/context_compute.h"

namespace dllm {
using TaskCompute = std::packaged_task<void(const ContextCompute *)>;
using FutureCompute = std::future<void>;
}  // namespace dllm
