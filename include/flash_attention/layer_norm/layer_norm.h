#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::flash_attn::layer_norm {
dllm::TaskCompute forward(
    std::shared_ptr<dllm::Tensor<2>> z,                  // Input: BxSxhidden_size
    std::shared_ptr<dllm::Tensor<1>> mu,                 // Input: FP32
    std::shared_ptr<dllm::Tensor<1>> rsigma,             // Input: FP32
    const std::shared_ptr<const dllm::Tensor<2>> x0,     // Input: BxSxhidden_size
    const std::shared_ptr<const dllm::Tensor<1>> gamma,  // hidden_size  // weight
    const std::shared_ptr<const dllm::Tensor<1>> beta,   // hidden_size  // bias
    const float epsilon                                // epsilon
);
}  // namespace dllm::flash_attn::layer_norm