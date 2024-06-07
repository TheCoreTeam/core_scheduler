#pragma once
#include <vector>

#include "threading/scheduler.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadPoolCudart final : Scheduler {
  ThreadPoolCudart(int localRank, int threadNum,
                   const std::vector<int> &bindingMap = {});
};
}  // namespace dllm
