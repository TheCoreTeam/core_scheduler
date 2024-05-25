#pragma once
#include "tensor.h"

namespace dllm::compute::Linear {
TaskCompute forward(const std::shared_ptr<Tensor> &output,
                    const std::shared_ptr<const Tensor> &input,
                    const std::shared_ptr<const Tensor> &weight);

TaskCompute backwardInput(const std::shared_ptr<Tensor> &dinput,
                          const std::shared_ptr<const Tensor> &grad_output,
                          const std::shared_ptr<const Tensor> &weight);

TaskCompute backwardWeight(const std::shared_ptr<Tensor> &dweight,
                           const std::shared_ptr<const Tensor> &grad_output,
                           const std::shared_ptr<const Tensor> &input);
}  // namespace dllm::compute::Linear
