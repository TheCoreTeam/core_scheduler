#pragma once
#include "threading/scheduler.h"
#include "threading/task.h"

namespace dllm {
struct Scheduler::Impl {
  virtual ~Impl() = default;

  virtual void submit(Task &&task);
};
}  // namespace dllm