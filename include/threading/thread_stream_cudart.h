#pragma once

#include <optional>

#include "threading/scheduler.h"

namespace dllm {
struct ThreadStreamCudart : Scheduler {
  explicit ThreadStreamCudart(int deviceRank,
                              std::optional<const int> bindingMap = {});
};
}  // namespace dllm
