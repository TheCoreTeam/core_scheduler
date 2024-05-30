#pragma once

#include <atomic>
#include <barrier>
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
  ThreadStreamNccl(const ncclUniqueId &id, int ncclWorldSize, int ncclRank,
                   int deviceRank, std::optional<const int> bindingMap = {});

  ~ThreadStreamNccl();

  void submit(TaskNccl &&task);

  void submit(const TaskNccl &task) = delete;

  int64_t commSize() const;

  int64_t rank() const;

 private:
  const int64_t commSize_;
  const int64_t rank_;
  std::barrier<> barrier_;
  std::queue<TaskNccl> taskQueue{};
  std::mutex queueMutex{};
  std::condition_variable cv{};
  std::mutex cvMutex{};
  std::atomic<bool> shutDown{false};
  std::thread thread{};
};
}  // namespace dllm
