#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadStreamCudart {
  explicit ThreadStreamCudart(int deviceRank,
                              std::optional<const int> bindingMap = {});

  ~ThreadStreamCudart();

  void submit(TaskCudart &&task);

  void submit(const TaskCudart &task) = delete;

 private:
  std::latch latch_;
  std::queue<TaskCudart> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::jthread thread{};
};
}  // namespace dllm
