#include "compute/linear.h"

#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace dllm::compute {
OrderedDict<std::string, Tensor> Linear::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  if (args.bias) {
    dict.insert("bias", forward.bias);
  }
  return dict;
}

OrderedDict<std::string, module::State::Increment> Linear::State::increments() {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight",
              {forward.weight, forward.grad_weight, forward.optimizer_weight});
  if (args.bias) {
    dict.insert("bias",
                {forward.bias, forward.grad_bias, forward.optimizer_bias});
  }
  return dict;
}

void Linear::init(const Scheduler &scheduler, std::shared_ptr<State> &state,
                  const Options &options) {
  DLLM_ASSERT_TRUE(options.bias() == false, "we do not supprot bias now");
  // ReSharper disable once CppDFAUnreachableCode
  TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device().value());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype().value());
  }

  if (options.bias()) {
    struct Impl : Task::Impl {
      const TensorOptions options;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const TensorOptions options)
          : Task::Impl{std::move(output), {}, compute}, options{options} {}
      void operator()() const override {
        const auto weight_ = torch::empty(output()[0].sizes(), options);
        output()[0].impl()->tensor() = weight_;
        const auto bias_ = torch::empty(output()[1].sizes(), options);
        output()[1].impl()->tensor() = bias_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
        auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
            output()[0].impl()->tensor());
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(output()[1].impl()->tensor(), -bound, bound);
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::compute::Linear::init";
      }
    };

    Tensor weight;
    Tensor bias;
    // size
    weight.sizes() = IntArray{options.out_futures(), options.in_futures()};
    bias.sizes() = IntArray{options.out_futures()};
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight, bias}, tensorOptions})});
    state = std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.bias()});
  } else {
    struct Impl : Task::Impl {
      const TensorOptions options;

      explicit Impl(std::vector<Tensor> output /* weight */,
                    const TensorOptions options)
          : Task::Impl{std::move(output), {}, compute}, options{options} {}
      void operator()() const override {
        const auto weight_ = torch::empty(output()[0].sizes(), options);
        output()[0].impl()->tensor() = weight_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
      }
      [[nodiscard]] const char *name() const override {
        return "dllm::compute::Linear::init";
      }
    };

    Tensor weight;
    // size
    weight.sizes() = IntArray{options.out_futures(), options.in_futures()};
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight}, tensorOptions})});
    state =
        std::make_shared<State>(State::Forward{std::move(weight)},
                                State::Backward{}, State::Args{options.bias()});
  }
}

void Linear::forward(const Scheduler &scheduler,
                     const std::shared_ptr<State> &state, Tensor &output,
                     const ReadOnlyTensor &input) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* output */,
                  std::vector<ReadOnlyTensor> input /* input, weight */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::linear(input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Linear::forward";
    }
  };

  Tensor output_;
  state->backward.input = input;
  // size
  auto input_size = input.sizes();
  input_size[input_size.size() - 1] = state->forward.weight.sizes()[0];
  output_.sizes() = input_size;
  output = output_;
  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{{output}, {input, state->forward.weight}})});
}

void Linear::backwardInput(const Scheduler &scheduler,
                           const std::shared_ptr<State> &state,
                           Tensor &grad_input,
                           const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* grad_input */,
                  std::vector<ReadOnlyTensor> input /* grad_output, weight */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      output()[0].impl()->tensor() =
          at::matmul(input()[0].impl()->tensor(), input()[1].impl()->tensor());
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Linear::backwardInput";
    }
  };

  Tensor grad_input_;
  // size
  auto output_size = grad_output.sizes();
  output_size[output_size.size() - 1] = state->forward.weight.sizes()[1];
  grad_input_.sizes() = output_size;
  grad_input = grad_input_;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input}, {grad_output, state->forward.weight}})});
}

void Linear::backwardParameter(const Scheduler &scheduler,
                               const std::shared_ptr<State> &state,
                               const ReadOnlyTensor &grad_output) {
  struct Impl : Task::Impl {
    explicit Impl(std::vector<Tensor> output /* grad_weight */,
                  std::vector<ReadOnlyTensor> input /* grad_output, input */)
        : Task::Impl{std::move(output), std::move(input), compute} {}
    void operator()() const override {
      if (output()[0].impl()->tensor().defined()) {
        const auto reshapedGradOutput = input()[0].impl()->tensor().reshape(
            {-1, input()[0].impl()->tensor().size(-1)});
        const auto transposedGradOutput = reshapedGradOutput.t();
        const auto reshapedInput = input()[1].impl()->tensor().reshape(
            {-1, input()[1].impl()->tensor().size(-1)});
        const auto result = at::matmul(transposedGradOutput, reshapedInput);
        output()[0].impl()->tensor() += result;
        intermediate().resize(4);
        intermediate().push_back(reshapedGradOutput);
        intermediate().push_back(transposedGradOutput);
        intermediate().push_back(reshapedInput);
        intermediate().push_back(result);
      } else {
        const auto reshapedGradOutput = input()[0].impl()->tensor().reshape(
            {-1, input()[0].impl()->tensor().size(-1)});
        const auto transposedGradOutput = reshapedGradOutput.t();
        const auto reshapedInput = input()[1].impl()->tensor().reshape(
            {-1, input()[1].impl()->tensor().size(-1)});
        const auto result = at::matmul(transposedGradOutput, reshapedInput);
        output()[0].impl()->tensor() = result;
        intermediate().resize(3);
        intermediate().push_back(reshapedGradOutput);
        intermediate().push_back(transposedGradOutput);
        intermediate().push_back(reshapedInput);
      }
    }
    [[nodiscard]] const char *name() const override {
      return "dllm::compute::Linear::backwardParameter";
    }
  };

  // size
  state->forward.grad_weight.sizes() = IntArray{
      grad_output.sizes()[grad_output.sizes().size() - 1],
      state->backward.input.sizes()[state->backward.input.sizes().size() - 1]};
  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {state->forward.grad_weight}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
}
}  // namespace dllm::compute
