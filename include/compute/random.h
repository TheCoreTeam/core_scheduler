#pragma once
#include "tensor.h"

namespace dllm::compute::Random {
TaskCompute kaimingNorm(const std::shared_ptr<Tensor2D> &x);

TaskCompute gaussian(const std::shared_ptr<Tensor1D> &x);

TaskCompute uniform(const std::shared_ptr<Tensor1D> &x);
}  // namespace dllm::compute::Random
