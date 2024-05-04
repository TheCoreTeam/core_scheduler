#pragma once
#include <future>

#include "threading/context_nccl.h"

namespace dllm {
using TaskNccl = std::packaged_task<void(const ContextNccl *)>;
using FutureNccl = std::future<void>;
}  // namespace dllm
