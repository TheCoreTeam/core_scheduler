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

  static std::shared_ptr<State> init(const Scheduler &scheduler);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input);

  static Tensor backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output);
};
}  // namespace dllm::compute
