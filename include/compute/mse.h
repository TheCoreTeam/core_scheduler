#pragma once
#include "task.h"
#include "tensor.h"
namespace dllm::compute::Mse {
Task forward(const std::shared_ptr<Tensor1D> &error,
             const std::shared_ptr<const Tensor1D> &x,
             const std::shared_ptr<const Tensor1D> &y);

Task backward(const std::shared_ptr<Tensor1D> &dx,
              const std::shared_ptr<const Tensor1D> &x,
              const std::shared_ptr<const Tensor1D> &y);
}  // namespace dllm::compute::Mse
