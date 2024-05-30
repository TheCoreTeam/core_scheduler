#pragma once

#include <atomic>
#include <barrier>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "threading/context_cudart.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadStreamCudart {
  explicit ThreadStreamCudart(int deviceRank,
                              std::optional<const int> bindingMap = {});

  ~ThreadStreamCudart();

  void submit(TaskCudart &&task);

  void submit(const TaskCudart &task) = delete;

 private:
  std::barrier<> barrier_;
  std::queue<TaskCudart> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::thread thread{};
};
}  // namespace dllm
