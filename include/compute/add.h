#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::Add {
TaskCompute forward(const std::shared_ptr<Tensor>& output,
                    const std::shared_ptr<const Tensor>& A,
                    const std::shared_ptr<const Tensor>& B);
}  // namespace dllm::compute::Add
