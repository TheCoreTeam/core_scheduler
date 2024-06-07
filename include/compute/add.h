#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute::Add {
void forward(const Scheduler& scheduler, Tensor& output,
             const ReadOnlyTensor& A, const ReadOnlyTensor& B);
}  // namespace dllm::compute::Add
