#pragma once
#include "tensor.h"
#include "threading/context_compute.h"

namespace dllm::compute::FlashAttention {
// no dropout
TaskCompute forward(const std::shared_ptr<dllm::Tensor<1>> &random_state,
                    const std::shared_ptr<dllm::Tensor<4>> &out,
                    const std::shared_ptr<dllm::Tensor<3>> &softmax_lse,
                    const std::shared_ptr<const dllm::Tensor<4>> &q,
                    const std::shared_ptr<const dllm::Tensor<4>> &k,
                    const std::shared_ptr<const dllm::Tensor<4>> &v,
                    double drouput_p, double softmax_scale);

TaskCompute backward(const std::shared_ptr<dllm::Tensor<4>> &dq,
                     const std::shared_ptr<dllm::Tensor<4>> &dk,
                     const std::shared_ptr<dllm::Tensor<4>> &dv,
                     const std::shared_ptr<dllm::Tensor<4>> &dout,
                     const std::shared_ptr<dllm::Tensor<1>> &random_state,
                     const std::shared_ptr<dllm::Tensor<4>> &out,
                     const std::shared_ptr<dllm::Tensor<3>> &softmax_lse,
                     const std::shared_ptr<const dllm::Tensor<4>> &q,
                     const std::shared_ptr<const dllm::Tensor<4>> &k,
                     const std::shared_ptr<const dllm::Tensor<4>> &v,
                     const double drouput_p, const double softmax_scale);
}  // namespace dllm::compute::FlashAttention
