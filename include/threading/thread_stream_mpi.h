#pragma once

#include <optional>

#include "threading/context_mpi.h"
#include "threading/scheduler.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_mpi.h"

namespace dllm {
struct ThreadStreamMpi : Scheduler {
  explicit ThreadStreamMpi(const ContextMpi &context,
                           std::optional<const int> bindingMap = {});

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;
};
}  // namespace dllm
