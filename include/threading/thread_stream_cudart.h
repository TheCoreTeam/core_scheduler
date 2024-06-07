#pragma once

#include <optional>

#include "threading/scheduler.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadStreamCudart : Scheduler {
  explicit ThreadStreamCudart(int deviceRank,
                              std::optional<const int> bindingMap = {});
};
}  // namespace dllm
