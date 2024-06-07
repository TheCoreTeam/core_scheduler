#pragma once
#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct LayerNorm {
  struct State final : module::State {
    struct Forward {
      std::shared_ptr<Tensor> weight = nullptr;
      std::shared_ptr<Tensor> bias = nullptr;
      std::shared_ptr<Tensor> grad_weight = nullptr;
      std::shared_ptr<Tensor> grad_bias = nullptr;
      std::shared_ptr<module::OptimizerState> optimizer_weight = nullptr;
      std::shared_ptr<module::OptimizerState> optimizer_bias = nullptr;
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

    State(const Forward &forward, const Backward &backward, const Args &args)
        : forward{forward}, backward{backward}, args{args} {}

    [[nodiscard]] OrderedDict<std::string, std::shared_ptr<Tensor>> parameters()
        const override;

    [[nodiscard]] OrderedDict<std::string, Increment> increments() override;
  };

  struct Options {
    Options(IntArrayRef normalized_shape)
        : normalized_shape_{normalized_shape} {}
    DLLM_ARG(IntArray, normalized_shape);
    DLLM_ARG(double, eps) = 1e-05;
    DLLM_ARG(bool, elementwise_affine) = true;
    DLLM_ARG(bool, bias) = true;
    DLLM_ARG(c10::optional<at::Device>, device) = {};
    DLLM_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static void init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                   const Options &options);

  static void forward(const Scheduler &scheduler,
                      const std::shared_ptr<State> &state,
                      const std::shared_ptr<Tensor> &output,
                      const std::shared_ptr<const ReadOnlyTensor> &input);

  static void backward(
      const Scheduler &scheduler, const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &grad_input,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);
};
}  // namespace dllm::compute
