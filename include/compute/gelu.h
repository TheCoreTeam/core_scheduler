#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::GeLU {
TaskCompute forward(const std::shared_ptr<Tensor> &output,
                    const std::shared_ptr<const Tensor> &input);

TaskCompute backward(const std::shared_ptr<Tensor> &dinput,
                     const std::shared_ptr<const Tensor> &input,
                     const std::shared_ptr<const Tensor> &doutput);
}  // namespace dllm::compute::GeLU
