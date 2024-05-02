#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "context.h"

namespace dllm {
struct ThreadPool {
  using Task = std::packaged_task<void(Context *)>;
  using Future = std::future<void>;

  ThreadPool(int localRank, int threadNum, const std::vector<int> &bindingMap);

  ~ThreadPool();

  Future submit(Task task);

  std::vector<std::thread> threadVector;
  std::queue<Task> taskQueue;
  std::mutex queueMutex;
  std::condition_variable cv;
  std::mutex cvMutex;
  std::atomic<bool> shutDown;
};
}  // namespace dllm
