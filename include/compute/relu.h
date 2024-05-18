#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::ReLU {
TaskCompute forward(const std::shared_ptr<const Tensor1D>& input,
                    const std::shared_ptr<Tensor1D>& output);
}  // namespace dllm::compute::ReLU
