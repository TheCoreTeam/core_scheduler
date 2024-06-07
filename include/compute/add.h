#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute::Add {
void forward(const Scheduler& scheduler, const std::shared_ptr<Tensor>& output,
             const std::shared_ptr<const ReadOnlyTensor>& A,
             const std::shared_ptr<const ReadOnlyTensor>& B);
}  // namespace dllm::compute::Add
