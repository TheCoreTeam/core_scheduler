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

  void submit(TaskNccl &&task) const;

  void submit(const TaskNccl &task) const = delete;

  [[nodiscard]] int64_t commSize() const;

  [[nodiscard]] int64_t rank() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
