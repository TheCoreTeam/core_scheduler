#pragma once
#include "threading/scheduler.h"

namespace dllm {
struct DynamicScheduler : Scheduler {
  explicit DynamicScheduler(int localRank);
};
}  // namespace dllm
