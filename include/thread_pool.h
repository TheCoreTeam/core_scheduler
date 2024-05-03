#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "context.h"
#include "task.h"

namespace dllm {
struct ThreadPoolBase {
  ~ThreadPoolBase();

  std::shared_ptr<Future> submit(Task &&task);

  std::shared_ptr<Future> submit(const Task &task) = delete;

 protected:
  ThreadPoolBase() = default;

  std::vector<std::thread> threadVector{};
  std::queue<Task> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};

struct ThreadPool : public ThreadPoolBase {
  ThreadPool(int localRank, int threadNum,
             const std::vector<int> &bindingMap = {});
};

struct ThreadPoolLight : public ThreadPoolBase {
  ThreadPoolLight(int localRank, int threadNum,
                  const std::vector<int> &bindingMap = {});
};
}  // namespace dllm
