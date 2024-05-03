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
// Basic thread pool, but we do not want you to use it alone. Consider
// ThreadPool or ThreadPoolLight.
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

// This is only for anything, including computation (cublas, cublasLt, etc.) as
// well as communication (MPI, NCCL, etc.). However, this thread pool requires
// more resource (e.g., GPU context memory).
struct ThreadPool : public ThreadPoolBase {
  ThreadPool(int localRank, int threadNum,
             const std::vector<int> &bindingMap = {});
};

// This is only for communication (MPI, NCCL, etc.), do not submit any
// non-CPU computation task to this thread pool!
struct ThreadPoolLight : public ThreadPoolBase {
  ThreadPoolLight(int localRank, int threadNum,
                  const std::vector<int> &bindingMap = {});
};
}  // namespace dllm
