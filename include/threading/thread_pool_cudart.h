#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <queue>
#include <thread>

// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadPoolCudart {
  ThreadPoolCudart(int localRank, int threadNum,
                   const std::vector<int> &bindingMap = {});

  ~ThreadPoolCudart();

  void submit(TaskCudart &&task);

  void submit(const TaskCudart &task) = delete;

 private:
  std::latch latch_;
  std::vector<std::jthread> threadVector{};
  std::queue<TaskCudart> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
