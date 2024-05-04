#pragma once
#include <future>

#include "threading/context_mpi.h"

namespace dllm {
using TaskMpi = std::packaged_task<void(const ContextMpi *)>;
using FutureMpi = std::future<void>;
}  // namespace dllm
