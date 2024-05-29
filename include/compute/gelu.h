#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct GeLU {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> input = nullptr;
    } backward;
    struct Args {
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state);

  static TaskCompute forward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &output,
      const std::shared_ptr<const ReadOnlyTensor> &input);

  static TaskCompute backward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &dinput,
      const std::shared_ptr<const ReadOnlyTensor> &doutput);
};
}  // namespace dllm::compute
