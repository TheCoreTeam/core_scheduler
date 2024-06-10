#pragma once
#include <vector>

#include "threading/scheduler.h"

namespace dllm {
struct ThreadPoolCudart final : Scheduler {
  ThreadPoolCudart(int localRank, int threadNum,
                   const std::vector<int> &bindingMap = {});
};
}  // namespace dllm
