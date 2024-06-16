#pragma once
#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/scheduler.h"

namespace dllm::compute {
struct LayerNorm {
  struct State final : module::State {
    struct Forward {
      Tensor weight;
      Tensor bias{};
      Tensor grad_weight{};
      Tensor grad_bias{};
      std::shared_ptr<module::OptimizerState> optimizer_weight{};
      std::shared_ptr<module::OptimizerState> optimizer_bias{};
    } forward;
    struct Backward {
      ReadOnlyTensor input{};
      ReadOnlyTensor mean{};
      ReadOnlyTensor rstd{};
    } backward;
    struct Args {
      const IntArray normalized_shape;
      const double eps;
      const bool elementwise_affine;
      const bool bias;
    } args;

    State(const Forward &forward, const Backward &backward, const Args &args)
        : forward{forward}, backward{backward}, args{args} {}

    [[nodiscard]] OrderedDict<std::string, Tensor> parameters() const override;

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

  static std::shared_ptr<State> init(const Scheduler &scheduler,
                                     const Options &options);

  static Tensor forward(const Scheduler &scheduler,
                        const std::shared_ptr<State> &state,
                        const ReadOnlyTensor &input);

  static Tensor backward(const Scheduler &scheduler,
                         const std::shared_ptr<State> &state,
                         const ReadOnlyTensor &grad_output);
};
}  // namespace dllm::compute
