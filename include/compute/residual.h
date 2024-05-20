#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::Residual {
TaskCompute forward(const std::shared_ptr<const Tensor3D>& input,
                    const std::shared_ptr<const Tensor3D>& residual,
                    const std::shared_ptr<Tensor3D>& output);
TaskCompute backward(const std::shared_ptr<const Tensor3D>& grad_output,
                     const std::shared_ptr<Tensor3D>& grad_input,
                     const std::shared_ptr<Tensor3D>& grad_residual);

}  // namespace dllm::compute::Residual
