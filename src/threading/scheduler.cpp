#include <arpa/inet.h>

#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>

#include "logger.h"
#include "threading/scheduler_impl.h"

namespace dllm {
const std::shared_ptr<Scheduler::Impl>& Scheduler::impl() const {
  return impl_;
}

void Scheduler::Impl::submit(Task&& task) {
  DLLM_ASSERT_TRUE(false, "Wrong task - Scheduler pair");
}
}  // namespace dllm
