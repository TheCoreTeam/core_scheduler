#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct Linear {
  struct State {
    struct Forward {
      std::shared_ptr<Tensor> weight;
      std::shared_ptr<Tensor> bias;
      std::shared_ptr<Tensor> grad_weight = Tensor::create();
      std::shared_ptr<Tensor> grad_bias = Tensor::create();
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> input = nullptr;
    } backward;
    struct Args {
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state, int64_t in_futures,
                          int64_t out_futures, bool bias = true,
                          c10::optional<at::Device> device = {},
                          c10::optional<at::ScalarType> dtype = {});

  static TaskCompute forward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &output,
      const std::shared_ptr<const ReadOnlyTensor> &input);

  static TaskCompute backwardInput(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &dinput,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);

  static TaskCompute backwardWeight(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);
};
}  // namespace dllm::compute
