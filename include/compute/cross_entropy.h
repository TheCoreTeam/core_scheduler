#pragma once
#include <ATen/core/Reduction.h>

#include "arg.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct CrossEntropy {
  struct State {
    struct Forward {
      ReadOnlyTensor weight;
    } forward;
    struct Backward {
      ReadOnlyTensor total_weight;
      ReadOnlyTensor log_probs;
      ReadOnlyTensor target;
      ReadOnlyTensor loss;
    } backward;
    struct Args {
      int64_t reduction;
      int64_t ignore_index;
      double label_smoothing;
    } args;
  };

  struct Options {
    Options() {}
    DLLM_ARG(at::Reduction::Reduction, reduction) = at::Reduction::Mean;
    DLLM_ARG(int64_t, ignore_index) = -100;
    DLLM_ARG(double, label_smoothing) = 0.0;
  };

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                   const Options &options = {});

  static void forward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state, Tensor &loss,
                      const ReadOnlyTensor &input,
                      const ReadOnlyTensor &target);

  static void backward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state, Tensor &grad_input);
};
}  // namespace dllm::compute
