#pragma once
#include "arg.h"
#include "module/state.h"
#include "tensor.h"
#include "threading/task_compute.h"

namespace dllm::compute {
struct Linear {
  struct State final : module::State {
    struct Forward {
      std::shared_ptr<Tensor> weight;
      std::shared_ptr<Tensor> bias = nullptr;
      std::shared_ptr<Tensor> grad_weight = nullptr;
      std::shared_ptr<Tensor> grad_bias = nullptr;
      std::shared_ptr<module::OptimizerState> optimizer_weight = nullptr;
      std::shared_ptr<module::OptimizerState> optimizer_bias = nullptr;
    } forward;
    struct Backward {
      std::shared_ptr<const ReadOnlyTensor> input = nullptr;
    } backward;
    struct Args {
      const bool bias;
    } args;

    State(const Forward &forward, const Backward &backward, const Args &args)
        : forward{forward}, backward{backward}, args{args} {}

    [[nodiscard]] OrderedDict<std::string, std::shared_ptr<Tensor>> parameters()
        const override;

    [[nodiscard]] OrderedDict<std::string, Increment> increments() override;
  };

  struct Options {
    Options(const int64_t in_futures, const int64_t out_futures)
        : in_futures_(in_futures), out_futures_(out_futures) {}
    DLLM_ARG(int64_t, in_futures);
    DLLM_ARG(int64_t, out_futures);
    DLLM_ARG(bool, bias) = true;
    DLLM_ARG(c10::optional<at::Device>, device) = {};
    DLLM_ARG(c10::optional<at::ScalarType>, dtype) = {};
  };

  static TaskCompute init(std::shared_ptr<State> &state,
                          const Options &options);

  static TaskCompute forward(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &output,
      const std::shared_ptr<const ReadOnlyTensor> &input);

  static TaskCompute backwardInput(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<Tensor> &dinput,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);

  static TaskCompute backwardParameter(
      const std::shared_ptr<State> &state,
      const std::shared_ptr<const ReadOnlyTensor> &grad_output);
};
}  // namespace dllm::compute
