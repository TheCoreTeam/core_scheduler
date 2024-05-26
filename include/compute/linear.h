#pragma once
#include "tensor.h"

namespace dllm::compute {
struct Linear {
  struct State {
    struct Forward {
      std::shared_ptr<Tensor> weight;
      std::shared_ptr<Tensor> bias;
    } forward;
    struct Backward {
      std::shared_ptr<const Tensor> input;
    } backward;
    struct Args {
    } args;
  };

  static std::shared_ptr<State> init(int64_t in_futures, int64_t out_futures,
                                     bool bias = true,
                                     c10::optional<at::Device> device = {},
                                     c10::optional<at::ScalarType> dtype = {});

  static TaskCompute forward(const std::shared_ptr<State> &state,
                             const std::shared_ptr<Tensor> &output,
                             const std::shared_ptr<const Tensor> &input);

  static TaskCompute backwardInput(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &dinput,
      const std::shared_ptr<const Tensor> &grad_output);

  static TaskCompute backwardWeight(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &dweight,
      const std::shared_ptr<const Tensor> &grad_output);
};
}  // namespace dllm::compute
