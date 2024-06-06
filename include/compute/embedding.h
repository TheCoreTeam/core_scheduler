#pragma once
#include "arg.h"
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

  struct Options {
    Options(const int64_t num_embeddings, const int64_t embedding_dim)
        : num_embeddings_{num_embeddings}, embedding_dim_{embedding_dim} {}
    DLLM_ARG(int64_t, num_embeddings);
    DLLM_ARG(int64_t, embedding_dim);
    DLLM_ARG(c10::optional<int64_t>, padding_idx) = {};
    DLLM_ARG(c10::optional<double>, max_norm) = {};
    DLLM_ARG(double, norm_type) = 2.;
    DLLM_ARG(bool, scale_grad_by_freq) = false;
    DLLM_ARG(bool, sparse) = false;
    DLLM_ARG(c10::optional<at::Device>, device) = {};
    DLLM_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static TaskCompute init(std::shared_ptr<State> &state,
                          const Options &options);

  static TaskCompute forward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &output,
      const std::shared_ptr<const ReadOnlyTensor> &indices);

  static TaskCompute backward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);
};

}  // namespace dllm::compute
