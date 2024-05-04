#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "threading/context_compute.h"
#include "threading/task_compute.h"

namespace dllm {
// This is only for anything, including computation (cublas, cublasLt, etc.) as
// well as communication (MPI, NCCL, etc.). However, this thread pool requires
// more resource (e.g., GPU context memory).
struct ThreadPoolCompute {
  ThreadPoolCompute(int localRank, int threadNum,
                    const std::vector<int> &bindingMap = {});

  ~ThreadPoolCompute();

  std::shared_ptr<FutureCompute> submit(TaskCompute &&task);

  std::shared_ptr<FutureCompute> submit(const TaskCompute &task) = delete;

 private:
  std::vector<std::thread> threadVector{};
  std::queue<TaskCompute> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
