#pragma once
#include "ATen/core/Reduction.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct CrossEntropy {
  struct State {
    struct Forward {
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> input = nullptr;
      std::shared_ptr<const ReadOnlyTensor> target = nullptr;
    } backward;
    struct Args {
      const std::shared_ptr<const ReadOnlyTensor> weight;
      int64_t reduction;
      int64_t ignore_index;
      double label_smoothing;
    } args;
  };

  static TaskCompute init(
      std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &weight = nullptr,
      int64_t reduction = at::Reduction::Mean, int64_t ignore_index = -100,
      double label_smoothing = 0.0);

  static TaskCompute forward(
      const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &loss,
      const std::shared_ptr<const ReadOnlyTensor> &input,
      const std::shared_ptr<const ReadOnlyTensor> &target);

  static TaskCompute backward(const std::shared_ptr<State> &state,
                              const std::shared_ptr<Tensor> &dinput);
};
}  // namespace dllm::compute