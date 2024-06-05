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
#include "threading/task_compute.h"

namespace dllm {
// This is only for anything, including computation (cublas, cublasLt, etc.) as
// well as communication (MPI, NCCL, etc.). However, this thread pool requires
// more resource (e.g., GPU context memory).
struct ThreadPoolCompute {
  ThreadPoolCompute(int localRank, int threadNum,
                    const std::vector<int> &bindingMap = {});

  ~ThreadPoolCompute();

  void submit(TaskCompute &&task);

  void submit(const TaskCompute &task) = delete;

 private:
  std::latch latch_;
  std::vector<std::jthread> threadVector{};
  std::queue<TaskCompute> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm
