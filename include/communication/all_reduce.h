#pragma once
#include "communication.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::communication {
struct AllReduceBucket : Bucket {
  AllReduceBucket(int64_t byteThreshold, Operation operation);

  void push_back(const Scheduler& scheduler, const Comm& comm,
                 Tensor tensor) const;
};

struct AllReduce {
  static void runInplace(const Scheduler& scheduler, const Comm& comm,
                         const std::vector<Tensor>& tensors,
                         Operation operation);
};
}  // namespace dllm::communication
