#pragma once
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct LayerNorm {
  struct State {
    struct Forward {
      std::shared_ptr<Tensor> weight = nullptr;
      std::shared_ptr<Tensor> bias = nullptr;
      std::shared_ptr<Tensor> dweight = Tensor::create();
      std::shared_ptr<Tensor> dbias = Tensor::create();
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> input = nullptr;
      std::shared_ptr<const ReadOnlyTensor> mean = nullptr;
      std::shared_ptr<const ReadOnlyTensor> rstd = nullptr;
    } backward;
    struct Args {
      const IntArray normalized_shape;
      const double eps;
      const bool elementwise_affine;
      const bool bias;
    } args;
  };

  static TaskCompute init(std::shared_ptr<State> &state,
                          IntArray normalized_shape, double eps = 1e-05,
                          bool elementwise_affine = true, bool bias = true,
                          c10::optional<at::Device> device = {},
                          c10::optional<at::ScalarType> dtype = {});

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
