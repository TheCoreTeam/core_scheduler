#include "logger.h"
#include "threading/scheduler_impl.h"

namespace dllm {
const std::shared_ptr<Scheduler::Impl>& Scheduler::impl() const {
  return impl_;
}

void Scheduler::Impl::submit(TaskCompute&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}

void Scheduler::Impl::submit(TaskCudart&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}

void Scheduler::Impl::submit(TaskMpi&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}

void Scheduler::Impl::submit(TaskNccl&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}
}  // namespace dllm
