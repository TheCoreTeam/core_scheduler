#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
struct AllGather {
  static void run(const Scheduler &scheduler, const Comm &comm,
                  const std::vector<std::vector<Tensor>> &tensorReceive,
                  const std::vector<ReadOnlyTensor> &tensorSend);
};
}  // namespace dllm::communication
