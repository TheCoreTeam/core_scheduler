#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
template <Backend backend>
struct AllGather;

template <>
struct AllGather<NCCL> {
  static void run(
      const Scheduler &scheduler,
      const std::vector<std::vector<std::shared_ptr<Tensor>>> &tensorReceive,
      const std::vector<std::shared_ptr<const ReadOnlyTensor>> &tensorSend);
};
}  // namespace dllm::communication
