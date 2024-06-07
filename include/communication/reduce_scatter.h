#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
template <Backend backend>
struct ReduceScatter;

template <>
struct ReduceScatter<NCCL> {
  static void run(
      const Scheduler &scheduler,
      const std::vector<std::shared_ptr<Tensor>> &tensorReceive,
      const std::vector<std::vector<std::shared_ptr<const ReadOnlyTensor>>>
          &tensorSend,
      Operation operation);
};
}  // namespace dllm::communication
