#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct GeLU {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      ReadOnlyTensor input;
    } backward;
    struct Args {
    } args;
  };

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state);

  static void forward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state, Tensor &output,
                      const ReadOnlyTensor &input);

  static void backward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state, Tensor &grad_input,
                       const ReadOnlyTensor &grad_output);
};
}  // namespace dllm::compute
