#pragma once
#include "tensor.h"
#include "threading/scheduler.h"

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

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state);

  static void forward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state,
                      const std::shared_ptr<Tensor> &output,
                      const std::shared_ptr<const ReadOnlyTensor> &input);

  static void backward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state,
                       const std::shared_ptr<Tensor> &dinput,
                       const std::shared_ptr<const ReadOnlyTensor> &doutput);
};
}  // namespace dllm::compute
