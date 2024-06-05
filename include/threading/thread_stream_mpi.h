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

  ~ThreadStreamMpi();

  void submit(TaskMpi &&task);

  void submit(const TaskMpi &task) = delete;

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;

 private:
  const ContextMpi context_;
  std::latch latch_;
  std::queue<TaskMpi> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::jthread thread{};
};
}  // namespace dllm
