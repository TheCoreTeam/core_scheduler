#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"


namespace dllm::communication {
template <Backend backend>
struct AllToAll;

template <>
struct AllToAll<NCCL> {
  static void run(const Scheduler& scheduler,
      const std::vector<std::shared_ptr<Tensor>> &tensorReceive,
      const std::vector<std::shared_ptr<const ReadOnlyTensor>> &tensorSend);
};
}  // namespace dllm::communication
