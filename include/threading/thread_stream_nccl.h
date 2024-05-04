#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "threading/context_nccl.h"
#include "threading/task_nccl.h"

namespace dllm {
struct ThreadStreamNccl {
  ThreadStreamNccl(ncclUniqueId id, int ncclWorldSize, int ncclRank,
                   int deviceRank, std::optional<const int> bindingMap = {});

  ~ThreadStreamNccl();

  std::shared_ptr<FutureNccl> submit(TaskNccl &&task);

  std::shared_ptr<FutureNccl> submit(const TaskNccl &task) = delete;

 private:
  std::thread thread{};
  std::queue<TaskNccl> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
};
}  // namespace dllm