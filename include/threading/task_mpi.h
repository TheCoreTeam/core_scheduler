#pragma once
#include <future>

#include "threading/context_mpi.h"
#include "threading/task_future.h"

namespace dllm {
using TaskMpi = std::packaged_task<void(const ContextMpi *)>;
}  // namespace dllm
