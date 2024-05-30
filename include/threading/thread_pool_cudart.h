#pragma once
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "threading/context_cudart.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadPoolCudart {
  ThreadPoolCudart(int localRank, int threadNum,
                   const std::vector<int> &bindingMap = {});

  ~ThreadPoolCudart();

  void submit(TaskCudart &&task);

  void submit(const TaskCudart &task) = delete;

 private:
  std::barrier<> barrier_;
  std::vector<std::thread> threadVector{};
  std::queue<TaskCudart> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
