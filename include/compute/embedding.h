#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct Embedding {
  struct ForwardState {
    std::shared_ptr<Tensor> weight;
    int64_t num_weights;
    int64_t padding_idx;
    c10::optional<double> max_norm;
    double norm_type;
    bool scale_grad_by_freq;
    bool sparse;
  };

  struct BackwardState {
    std::shared_ptr<Tensor> weight;
    int64_t num_weights;
    int64_t padding_idx;
    c10::optional<double> max_norm;
    double norm_type;
    bool scale_grad_by_freq;
    bool sparse;
  };

  static std::shared_ptr<ForwardState> init(
      int64_t num_embeddings, int64_t embedding_dim,
      c10::optional<int64_t> padding_idx, c10::optional<double> max_norm,
      double norm_type = 2., bool scale_grad_by_freq = false,
      bool sparse = false, c10::optional<at::Device> device = {},
      c10::optional<at::ScalarType> dtype = {});

  static TaskCompute forward(const std::shared_ptr<Tensor> &output,
                             const std::shared_ptr<const Tensor> &indices,
                             const std::shared_ptr<const ForwardState> &state);

  static TaskCompute backward(const std::shared_ptr<Tensor> &grad_weight,
                              const std::shared_ptr<const Tensor> &grad_output,
                              const std::shared_ptr<const Tensor> &indices,
                              const std::shared_ptr<const ForwardState> &state);
};

}  // namespace dllm::compute
