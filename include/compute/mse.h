#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::Mse {
TaskCompute forward(const std::shared_ptr<Tensor1D> &error,
                    const std::shared_ptr<const Tensor1D> &x,
                    const std::shared_ptr<const Tensor1D> &y);

TaskCompute backward(const std::shared_ptr<Tensor1D> &dx,
                     const std::shared_ptr<const Tensor1D> &x,
                     const std::shared_ptr<const Tensor1D> &y);
}  // namespace dllm::compute::Mse
