#pragma once
#include "tensor.h"

namespace dllm::compute::FusedClassifier {
TaskCompute call(const std::shared_ptr<Tensor3D> &logits,
                 const std::shared_ptr<Tensor2D> &losses,
                 const std::shared_ptr<const Tensor2D> &targets);
}  // namespace dllm::compute::FusedClassifier
