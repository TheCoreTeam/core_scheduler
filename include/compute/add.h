#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute::Add {
TaskCompute forward(const std::shared_ptr<Tensor3D>& output,
                    const std::shared_ptr<const Tensor3D>& A,
                    const std::shared_ptr<const Tensor3D>& B);
TaskCompute backward(const std::shared_ptr<Tensor3D>& grad_A,
                     const std::shared_ptr<Tensor3D>& grad_B,
                     const std::shared_ptr<const Tensor3D>& grad_output);

}  // namespace dllm::compute::Add
