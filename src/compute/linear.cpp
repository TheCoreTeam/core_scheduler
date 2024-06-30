/*
 * Copyright (c) 2024 The Core team
 *
 * Licensed under the Apache License, Version 2.0;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Header Order Protection
// ReSharper disable once CppUnusedIncludeDirective
#include <c10/util/Exception.h>
// Header Order Protection

#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "compute/linear.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
Linear::State::State(const Forward &forward, const Backward &backward,
                     const Args &args)
    : forward{forward}, backward{backward}, args{args} {}

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

std::shared_ptr<Linear::State> Linear::init(const Scheduler &scheduler,
                                            const Options &options) {
  CS_ASSERT_TRUE(options.bias() == false, "we do not supprot bias now");
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
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const TensorOptions options, const int64_t in_futures,
                    const int64_t out_futures)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight_ = torch::empty({out_futures, in_futures}, options);
        output()[0].impl()->tensor() = weight_;
        const auto bias_ = torch::empty({out_futures}, options);
        output()[1].impl()->tensor() = bias_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
        auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
            output()[0].impl()->tensor());
        const auto bound = 1 / std::sqrt(fan_in);
        torch::nn::init::uniform_(output()[1].impl()->tensor(), -bound, bound);
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::Linear::init";
      }
    };

    Tensor weight;
    Tensor bias;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight, bias},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.bias()});
  } else {
    struct Impl : Task::Impl {
      const TensorOptions options;
      const int64_t in_futures;
      const int64_t out_futures;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const TensorOptions options, const int64_t in_futures,
                    const int64_t out_futures)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            in_futures{in_futures},
            out_futures{out_futures} {}
      void operator()() const override {
        const auto weight_ = torch::empty({out_futures, in_futures}, options);
        output()[0].impl()->tensor() = weight_;
        torch::nn::init::kaiming_uniform_(output()[0].impl()->tensor(),
                                          std::sqrt(5));
      }
      [[nodiscard]] const char *name() const override {
        return "cs::compute::Linear::init";
      }
    };

    Tensor weight;
    scheduler.impl()->submit(
        Task{std::make_shared<Impl>(Impl{{weight},
                                         tensorOptions,
                                         options.in_futures(),
                                         options.out_futures()})});
    return std::make_shared<State>(State::Forward{std::move(weight)},
                                   State::Backward{},
                                   State::Args{options.bias()});
  }
}

Tensor Linear::forward(const Scheduler &scheduler,
                       const std::shared_ptr<State> &state,
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
      return "cs::compute::Linear::forward";
    }
  };

  Tensor output;
  state->backward.input = input;
  scheduler.impl()->submit(Task{
      std::make_shared<Impl>(Impl{{output}, {input, state->forward.weight}})});
  return output;
}

Tensor Linear::backwardInput(const Scheduler &scheduler,
                             const std::shared_ptr<State> &state,
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
      return "cs::compute::Linear::backwardInput";
    }
  };

  Tensor grad_input;
  scheduler.impl()->submit(Task{std::make_shared<Impl>(
      Impl{{grad_input}, {grad_output, state->forward.weight}})});
  return grad_input;
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
      return "cs::compute::Linear::backwardParameter";
    }
  };

  // decrease counter
  scheduler.impl()->submit(Task{std::make_shared<Impl>(Impl{
      {state->forward.grad_weight}, {grad_output, state->backward.input}})});
  state->backward.input.reset();
}
}  // namespace cs::compute
