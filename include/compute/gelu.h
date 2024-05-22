#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::GeLU {
TaskCompute forward(const std::shared_ptr<Tensor2D> &output,
                    const std::shared_ptr<const Tensor2D> &input);

TaskCompute backward(const std::shared_ptr<Tensor2D> &dinput,
                     const std::shared_ptr<const Tensor2D> &input,
                     const std::shared_ptr<const Tensor2D> &doutput);
}  // namespace dllm::compute::GeLU
