#pragma once
#include "threading/scheduler.h"
#include "threading/task.h"
#include "threading/task_compute.h"
#include "threading/task_cudart.h"
#include "threading/task_mpi.h"
#include "threading/task_nccl.h"

namespace dllm {
struct Scheduler::Impl {
  virtual ~Impl() = default;

  virtual void submit(Task &&task);

  virtual void submit(TaskCompute &&task);

  virtual void submit(TaskCudart &&task);

  virtual void submit(TaskNccl &&task);

  virtual void submit(TaskMpi &&task);
};
}  // namespace dllm