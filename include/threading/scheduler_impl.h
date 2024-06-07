#pragma once
#include "task_compute.h"
#include "task_cudart.h"
#include "task_mpi.h"
#include "task_nccl.h"
#include "threading/scheduler.h"

namespace dllm {
struct Scheduler::Impl {
  virtual ~Impl() = default;

  virtual void submit(TaskCompute &&task);

  virtual void submit(TaskCudart &&task);

  virtual void submit(TaskNccl &&task);

  virtual void submit(TaskMpi &&task);
};
}  // namespace dllm