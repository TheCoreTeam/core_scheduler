#pragma once

#include <atomic>
#include <condition_variable>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

// ReSharper disable once CppUnusedIncludeDirective
#include "threading/submit_task_macro.h"
#include "threading/task_cudart.h"

namespace dllm {
struct ThreadStreamCudart {
  explicit ThreadStreamCudart(int deviceRank,
                              std::optional<const int> bindingMap = {});

  void submit(TaskCudart &&task) const;

  void submit(const TaskCudart &task) const = delete;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace dllm
