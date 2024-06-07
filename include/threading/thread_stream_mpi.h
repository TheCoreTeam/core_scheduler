#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "threading/context_mpi.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_mpi.h"

namespace dllm {
struct ThreadStreamMpi {
  explicit ThreadStreamMpi(const ContextMpi &context,
                           std::optional<const int> bindingMap = {});

  void submit(TaskMpi &&task) const;

  void submit(const TaskMpi &task) const = delete;

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
