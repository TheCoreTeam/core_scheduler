#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "threading/context_nccl.h"
// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_nccl.h"

namespace dllm {
struct ThreadStreamNccl {
  ThreadStreamNccl(MPI_Comm mpiComm, int deviceRank,
                   std::optional<const int> bindingMap = {});

  ~ThreadStreamNccl();

  void submit(TaskNccl &&task);

  void submit(const TaskNccl &task) = delete;

  int64_t commSize() const;

  int64_t rank() const;

 private:
  const int64_t commSize_;
  const int64_t rank_;
  std::latch latch_;
  std::queue<TaskNccl> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::jthread thread{};
};
}  // namespace dllm
