#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct Embedding {
  struct State {
    struct Forward {
      std::shared_ptr<Tensor> weight;
      std::shared_ptr<Tensor> grad_weight = Tensor::create();
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> indices = nullptr;
    } backward;
    struct Args {
      int64_t num_weights;
      int64_t padding_idx;
      c10::optional<double> max_norm;
      double norm_type;
      bool scale_grad_by_freq;
      bool sparse;
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state, int64_t num_embeddings,
                          int64_t embedding_dim,
                          c10::optional<int64_t> padding_idx,
                          c10::optional<double> max_norm, double norm_type = 2.,
                          bool scale_grad_by_freq = false, bool sparse = false,
                          c10::optional<at::Device> device = {},
                          c10::optional<at::ScalarType> dtype = {});

  static TaskCompute forward(const std::shared_ptr<State> &state,
                             const std::shared_ptr<Tensor> &output,
                             const std::shared_ptr<const ReadOnlyTensor> &indices);

  static TaskCompute backward(const std::shared_ptr<State> &state,
                              const std::shared_ptr<const ReadOnlyTensor> &grad_output);
};

}  // namespace dllm::compute
