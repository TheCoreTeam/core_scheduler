#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::NLL {
TaskCompute forward(const std::shared_ptr<Tensor1D> &loss,
                    const std::shared_ptr<const Tensor2D> &input,
                    const std::shared_ptr<const Tensor2D> &target);
}  // namespace dllm::compute::NLL
