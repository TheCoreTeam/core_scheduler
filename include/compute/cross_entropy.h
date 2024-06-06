#pragma once
#include "ATen/core/Reduction.h"
#include "arg.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct CrossEntropy {
  struct State {
    struct Forward {
      std::shared_ptr<const ReadOnlyTensor> weight = Tensor::create();
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> total_weight = nullptr;
      std::shared_ptr<const ReadOnlyTensor> log_probs = nullptr;
      std::shared_ptr<const ReadOnlyTensor> target = nullptr;
      std::shared_ptr<const ReadOnlyTensor> loss = nullptr;
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

  static TaskCompute init(std::shared_ptr<State> &state,
                          const Options &options = {});

  static TaskCompute forward(
      const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &loss,
      const std::shared_ptr<const ReadOnlyTensor> &input,
      const std::shared_ptr<const ReadOnlyTensor> &target);

  static TaskCompute backward(const std::shared_ptr<State> &state,
                              const std::shared_ptr<Tensor> &dinput);
};
}  // namespace dllm::compute
