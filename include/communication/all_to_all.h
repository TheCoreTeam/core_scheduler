#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
struct AllToAll {
  static void run(const Scheduler &scheduler, const Comm &comm,
                  const std::vector<Tensor> &tensorReceive,
                  const std::vector<ReadOnlyTensor> &tensorSend);
};
}  // namespace dllm::communication
