#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::GeLU {
TaskCompute forward(const std::shared_ptr<Tensor1D> &output,
                    const std::shared_ptr<const Tensor1D> &input);

TaskCompute backward(const std::shared_ptr<Tensor1D> &dinput,
                     const std::shared_ptr<const Tensor1D> &input,
                     const std::shared_ptr<const Tensor1D> &doutput);
}  // namespace dllm::compute::GeLU
