#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::flash_attn::LayerNorm {
TaskCompute forward(
    const std::shared_ptr<Tensor<2>> &z,            // Input: BxSxhidden_size
    const std::shared_ptr<Tensor<1>> &mu,           // Input: FP32
    const std::shared_ptr<Tensor<1>> &rsigma,       // Input: FP32
    const std::shared_ptr<const Tensor<2>> &x0,     // Input: BxSxhidden_size
    const std::shared_ptr<const Tensor<1>> &gamma,  // hidden_size  // weight
    const std::shared_ptr<const Tensor<1>> &beta,   // hidden_size  // bias
    float epsilon                                   // epsilon
);

TaskCompute backward(const std::shared_ptr<Tensor<2>> &dx0,
                     const std::shared_ptr<Tensor<1>> &dgamma,
                     const std::shared_ptr<Tensor<1>> &dbeta,
                     const std::shared_ptr<const Tensor<2>> &dz,
                     const std::shared_ptr<const Tensor<2>> &x,
                     const std::shared_ptr<const Tensor<1>> &mu,
                     const std::shared_ptr<const Tensor<1>> &rsigma,
                     const std::shared_ptr<const Tensor<1>> &gamma);
}  // namespace dllm::flash_attn::LayerNorm
