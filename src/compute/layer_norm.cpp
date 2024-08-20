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

#include <ATen/autocast_mode.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>

#include "compute/layer_norm.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_impl.h"

namespace cs::compute {
LayerNorm::State::State(const Forward& forward, const Backward& backward,
                        const Args& args)
    : forward{forward}, backward{backward}, args{args} {}

OrderedDict<std::string, Tensor> LayerNorm::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  if (args.bias) {
    dict.insert("bias", forward.bias);
  }
  return dict;
}

OrderedDict<std::string, Tensor> LayerNorm::State::gradients() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.grad_weight);
  if (args.bias) {
    dict.insert("bias", forward.grad_bias);
  }
  return dict;
}

OrderedDict<std::string, module::State::Increment>
LayerNorm::State::increments() const {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight", {forward.weight, forward.grad_weight});
  if (args.bias) {
    dict.insert("bias", {forward.bias, forward.grad_bias});
  }
  return dict;
}

void LayerNorm::State::zero_grad() {
  forward.grad_weight = {};
  if (args.bias) {
    forward.grad_bias = {};
  }
}

std::shared_ptr<LayerNorm::State> LayerNorm::init(const Scheduler& scheduler,
                                                  const Options& options) {
  CS_ASSERT_TRUE(options.elementwise_affine() == true,
                 "elementwise_affine must be enabled now");
  at::TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype());
  }
  Tensor weight;
  if (options.bias()) {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, kMain, kCompute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::ones(options.normalized_shape(), tensorOptions);
        output()[1].impl()->tensor() =
            at::zeros(options.normalized_shape(), tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::init";
      }
    };

    Tensor bias;
    auto task = Task{
        std::make_shared<Impl>(Impl{{weight, bias}, options, tensorOptions})};
    scheduler.impl()->submit(std::move(task));
    return std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  } else {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, kMain, kCompute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::ones(options.normalized_shape(), tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::init";
      }
    };

    auto task =
        Task{std::make_shared<Impl>(Impl{{weight}, options, tensorOptions})};
    scheduler.impl()->submit(std::move(task));
    return std::make_shared<State>(
        State::Forward{std::move(weight)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  }
}

Tensor LayerNorm::forward(const Scheduler& scheduler,
                          const std::shared_ptr<State>& state,
                          const ReadOnlyTensor& input) {
  Tensor mean;
  Tensor rstd;
  Task task{nullptr};
  Tensor output;
  if (state->args.bias) {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(std::vector<Tensor> output /* output, mean, rstd */,
                    std::vector<ReadOnlyTensor> input /* input, weight, bias */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
            args{args} {}
      void operator()() const override {
        std::make_tuple(std::ref(output()[0].impl()->tensor()),
                        std::ref(output()[1].impl()->tensor()),
                        std::ref(output()[2].impl()->tensor())) =
            at::native_layer_norm(input()[0].impl()->tensor(),
                                  args.normalized_shape,
                                  input()[1].impl()->tensor(),
                                  input()[2].impl()->tensor(), args.eps);
        if (at::autocast::is_enabled()) {
          const auto dtype = at::autocast::get_autocast_gpu_dtype();
          input()[0].impl()->auto_cast().enable = true;
          input()[1].impl()->auto_cast().enable = true;
          output()[0].impl()->auto_cast().enable = true;
          output()[1].impl()->auto_cast().enable = true;
          output()[2].impl()->auto_cast().enable = true;
        }
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::forward";
      }
    };

    task = Task{std::make_shared<Impl>(
        Impl{{output, mean, rstd},
             {input, state->forward.weight, state->forward.bias},
             state->args})};
  } else {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(std::vector<Tensor> output /* output, mean, rstd */,
                    std::vector<ReadOnlyTensor> input /* input, weight */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
            args{args} {}
      void operator()() const override {
        std::make_tuple(std::ref(output()[0].impl()->tensor()),
                        std::ref(output()[1].impl()->tensor()),
                        std::ref(output()[2].impl()->tensor())) =
            at::native_layer_norm(input()[0].impl()->tensor(),
                                  args.normalized_shape,
                                  input()[1].impl()->tensor(), {}, args.eps);
        if (at::autocast::is_enabled()) {
          const auto dtype = at::autocast::get_autocast_gpu_dtype();
          input()[0].impl()->auto_cast().enable = true;
          input()[1].impl()->auto_cast().enable = true;
          output()[0].impl()->auto_cast().enable = true;
          output()[1].impl()->auto_cast().enable = true;
          output()[2].impl()->auto_cast().enable = true;
        }
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::forward";
      }
    };

    task = Task{std::make_shared<Impl>(Impl{
        {output, mean, rstd}, {input, state->forward.weight}, state->args})};
  }
  state->backward.input = input;
  state->backward.mean = mean;
  state->backward.rstd = rstd;
  scheduler.impl()->submit(std::move(task));
  return output;
}

Tensor LayerNorm::backward(const Scheduler& scheduler,
                           const std::shared_ptr<State>& state,
                           const ReadOnlyTensor& grad_output) {
  backward_parameter(scheduler, state, grad_output);
  return backward_input(scheduler, state, grad_output);
}

Tensor LayerNorm::backward_input(const Scheduler& scheduler,
                                 const std::shared_ptr<State>& state,
                                 const ReadOnlyTensor& grad_output) {
  auto weight = static_cast<const ReadOnlyTensor&>(state->forward.weight);
  Tensor grad_input;
  Task task{nullptr};
  struct Impl : Task::Impl {
    const State::Args args;

    explicit Impl(std::vector<Tensor> output /* grad_input */,
                  std::vector<ReadOnlyTensor>
                      input /* grad_output, input, mean, rstd, weight */,
                  const State::Args& args)
        : Task::Impl{std::move(output), std::move(input), kMain, kCompute},
          args{args} {}
    void operator()() const override {
      at::Tensor grad_output = input()[0].impl()->tensor();
      at::Tensor x = input()[1].impl()->tensor();
      if (input()[1].impl()->auto_cast().enable) {
        if (x.dtype() != c10::kFloat) {
          x = at::autocast::cached_cast(c10::kFloat, x);
        }
        if (grad_output.dtype() != c10::kFloat) {
          intermediate().resize(2);
          intermediate().push_back(grad_output);
          grad_output = grad_output.to(c10::kFloat);
        }
      }
      auto [dx, dw, db] = at::native_layer_norm_backward(
          grad_output, x, args.normalized_shape, input()[2].impl()->tensor(),
          input()[3].impl()->tensor(), input()[4].impl()->tensor(), {},
          {true, false, false});
      output()[0].impl()->tensor() = dx;
    }
    [[nodiscard]] const char* name() const override {
      return "cs::compute::LayerNorm::backward_input";
    }
  };

  task = Task{std::make_shared<Impl>(
      Impl{{grad_input},
           {grad_output, state->backward.input, state->backward.mean,
            state->backward.rstd, state->forward.weight},
           state->args})};

  state->backward.input.reset();
  state->backward.mean.reset();
  state->backward.rstd.reset();
  scheduler.impl()->submit(std::move(task));
  return grad_input;
}

void LayerNorm::backward_parameter(const Scheduler& scheduler,
                                   const std::shared_ptr<State>& state,
                                   const ReadOnlyTensor& grad_output) {
  auto weight = static_cast<const ReadOnlyTensor&>(state->forward.weight);
  Tensor grad_input;
  Task task{nullptr};
  if (state->args.bias) {
    struct Impl : Task::Impl {
      const State::Args args;

      explicit Impl(
          std::vector<Tensor> output /* grad_weight, grad_bias */,
          std::vector<ReadOnlyTensor>
              input /* grad_output, input, mean, rstd, weight, bias */,
          const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute},
            args{args} {}
      void operator()() const override {
        at::Tensor grad_output = input()[0].impl()->tensor();
        at::Tensor x = input()[1].impl()->tensor();
        if (input()[1].impl()->auto_cast().enable) {
          if (x.dtype() != c10::kFloat) {
            x = at::autocast::cached_cast(c10::kFloat, x);
          }
          if (grad_output.dtype() != c10::kFloat) {
            intermediate().resize(3);
            intermediate().push_back(grad_output);
            grad_output = grad_output.to(c10::kFloat);
          } else {
            intermediate().resize(2);
          }
        } else {
          intermediate().resize(2);
        }
        auto [dx, dw, db] = at::native_layer_norm_backward(
            grad_output, x, args.normalized_shape, input()[2].impl()->tensor(),
            input()[3].impl()->tensor(), input()[4].impl()->tensor(),
            input()[5].impl()->tensor(), {false, true, true});
        if (output()[1].impl()->tensor().defined()) {
          output()[1].impl()->tensor() += dw;
        } else {
          output()[1].impl()->tensor() = dw;
        }
        if (output()[2].impl()->tensor().defined()) {
          output()[2].impl()->tensor() += db;
        } else {
          output()[2].impl()->tensor() = db;
        }
        intermediate().push_back(dw);
        intermediate().push_back(db);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::backward_parameter";
      }
    };

    task = Task{std::make_shared<Impl>(
        Impl{{grad_input, state->forward.grad_weight, state->forward.grad_bias},
             {grad_output, state->backward.input, state->backward.mean,
              state->backward.rstd, state->forward.weight, state->forward.bias},
             state->args})};
  } else {
    struct Impl : Task::Impl {
      const State::Args args;

      explicit Impl(std::vector<Tensor> output /* grad_weight */,
                    std::vector<ReadOnlyTensor>
                        input /* grad_output, input, mean, rstd, weight */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), kAssist, kCompute},
            args{args} {}
      void operator()() const override {
        at::Tensor grad_output = input()[0].impl()->tensor();
        at::Tensor x = input()[1].impl()->tensor();
        if (input()[1].impl()->auto_cast().enable) {
          if (x.dtype() != c10::kFloat) {
            x = at::autocast::cached_cast(c10::kFloat, x);
          }
          if (grad_output.dtype() != c10::kFloat) {
            intermediate().resize(2);
            intermediate().push_back(grad_output);
            grad_output = grad_output.to(c10::kFloat);
          } else {
            intermediate().resize(1);
          }
        } else {
          intermediate().resize(1);
        }
        auto [dx, dw, db] = at::native_layer_norm_backward(
            grad_output, x, args.normalized_shape, input()[2].impl()->tensor(),
            input()[3].impl()->tensor(), input()[4].impl()->tensor(), {},
            {false, true, false});
        if (output()[1].impl()->tensor().defined()) {
          output()[1].impl()->tensor() += dw;
        } else {
          output()[1].impl()->tensor() = dw;
        }
        intermediate().push_back(dw);
      }
      [[nodiscard]] const char* name() const override {
        return "cs::compute::LayerNorm::backward_parameter";
      }
    };

    task = Task{std::make_shared<Impl>(
        Impl{{grad_input, state->forward.grad_weight},
             {grad_output, state->backward.input, state->backward.mean,
              state->backward.rstd, state->forward.weight},
             state->args})};
  }

  scheduler.impl()->submit(std::move(task));
}

}  // namespace cs::compute
