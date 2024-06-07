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

  void submit(TaskCompute &&task) const;

  void submit(const TaskCompute &task) const = delete;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
