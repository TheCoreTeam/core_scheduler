#pragma once
#include <future>

#include "threading/context_nccl.h"
#include "threading/task_future.h"

namespace dllm {
using TaskNccl = std::packaged_task<void(const ContextNccl *)>;
}  // namespace dllm
