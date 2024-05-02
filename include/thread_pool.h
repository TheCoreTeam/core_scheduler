#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include "task.h"
#include "context.h"

namespace dllm {
struct ThreadPool {
  ThreadPool(int localRank, int threadNum, const std::vector<int> &bindingMap = {});

  ~ThreadPool();

  std::shared_ptr<Future> submit(Task task);

  std::vector<std::thread> threadVector;
  std::queue<Task> taskQueue;
  std::mutex queueMutex;
  std::condition_variable cv;
  std::mutex cvMutex;
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
