#pragma once
#include "threading/scheduler.h"
#include "threading/task.h"

namespace dllm {
struct Scheduler::Impl {
  Impl(int64_t deviceRank);

  virtual ~Impl() = default;

  virtual void submit(Task &&task);

  [[nodiscard]] int64_t deviceRank() const;

 private:
  const int64_t deviceRank_;
};
}  // namespace dllm