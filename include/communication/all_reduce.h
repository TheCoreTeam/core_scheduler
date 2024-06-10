#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
struct AllReduce {
  static void runInplace(const Scheduler& scheduler, const Comm& comm,
                         const std::vector<Tensor>& tensors,
                         Operation operation);
};
}  // namespace dllm::communication
