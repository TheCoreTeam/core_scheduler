#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "threading/context_mpi.h"
#include "threading/task_mpi.h"

namespace dllm {
struct ThreadStreamMpi {
  ThreadStreamMpi(ContextMpi context, std::optional<const int> bindingMap = {});

  ~ThreadStreamMpi();

  void submit(TaskMpi &&task);

  void submit(const TaskMpi &task) = delete;

 private:
  std::thread thread{};
  std::queue<TaskMpi> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
