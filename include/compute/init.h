#pragma once
#include "tensor.h"

namespace dllm::compute::Init {
TaskCompute kaimingNorm(const std::shared_ptr<Tensor2D> &y);
}  // namespace dllm::compute::Init
