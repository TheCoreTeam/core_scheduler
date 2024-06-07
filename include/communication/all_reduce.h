#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
template <Backend backend>
struct AllReduce;

template <>
struct AllReduce<NCCL> {
  static void runInplace(const Scheduler& scheduler,
                         const std::vector<Tensor>& tensors,
                         Operation operation);
};
}  // namespace dllm::communication
