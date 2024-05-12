#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::GeLU {
TaskCompute forward(const std::shared_ptr<Tensor1D> &output,
                    const std::shared_ptr<const Tensor1D> &input);
}  // namespace dllm::compute::GeLU
