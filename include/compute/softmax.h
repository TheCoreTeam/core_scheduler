#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::SoftMax {
TaskCompute forward(const std::shared_ptr<Tensor2D> &output,
                    const std::shared_ptr<const Tensor2D> &input, double scale);
}  // namespace dllm::compute::SoftMax
