#pragma once
#include "tensor.h"

namespace dllm::optimizer::Sgd {
TaskCompute step(const std::shared_ptr<Tensor1D> &w,
                 const std::shared_ptr<const Tensor1D> &dw, double lr);
}
