#include "compute/layer_norm.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>

#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"
#include "threading/task_impl.h"

namespace dllm::compute {
OrderedDict<std::string, Tensor> LayerNorm::State::parameters() const {
  OrderedDict<std::string, Tensor> dict;
  dict.insert("weight", forward.weight);
  if (args.bias) {
    dict.insert("bias", forward.bias);
  }
  return dict;
}

OrderedDict<std::string, module::State::Increment>
LayerNorm::State::increments() {
  OrderedDict<std::string, Increment> dict;
  dict.insert("weight",
              {forward.weight, forward.grad_weight, forward.optimizer_weight});
  if (args.bias) {
    dict.insert("bias",
                {forward.bias, forward.grad_bias, forward.optimizer_bias});
  }
  return dict;
}

void LayerNorm::init(const Scheduler& scheduler, std::shared_ptr<State>& state,
                     const Options& options) {
  DLLM_ASSERT_TRUE(options.elementwise_affine() == true,
                   "elementwise_affine must be enabled now");
  at::TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype());
  }
  Tensor weight;
  Task task{nullptr};
  if (options.bias()) {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight, bias */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::ones(options.normalized_shape(), tensorOptions);
        output()[1].impl()->tensor() =
            at::zeros(options.normalized_shape(), tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::init";
      }
    };

    Tensor bias;
    task = Task{
        std::make_shared<Impl>(Impl{{weight, bias}, options, tensorOptions})};
    state = std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  } else {
    struct Impl : Task::Impl {
      const Options options;
      const TensorOptions tensorOptions;

      explicit Impl(std::vector<Tensor> output /* weight */,
                    const Options& options, const TensorOptions tensorOptions)
          : Task::Impl{std::move(output), {}, compute},
            options{options},
            tensorOptions{tensorOptions} {}
      void operator()() const override {
        output()[0].impl()->tensor() =
            at::ones(options.normalized_shape(), tensorOptions);
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::init";
      }
    };

    task = Task{std::make_shared<Impl>(Impl{{weight}, options, tensorOptions})};
    state = std::make_shared<State>(
        State::Forward{std::move(weight)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  }
  scheduler.impl()->submit(std::move(task));
}

void LayerNorm::forward(const Scheduler& scheduler,
                        const std::shared_ptr<State>& state, Tensor& output,
                        const ReadOnlyTensor& input) {
  Tensor mean;
  Tensor rstd;
  Task task{nullptr};
  Tensor output_;
  if (state->args.bias) {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(std::vector<Tensor> output /* output, mean, rstd */,
                    std::vector<ReadOnlyTensor> input /* input, weight, bias */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), compute},
            args{args} {}
      void operator()() const override {
        std::make_tuple(std::ref(output()[0].impl()->tensor()),
                        std::ref(output()[1].impl()->tensor()),
                        std::ref(output()[2].impl()->tensor())) =
            at::native_layer_norm(input()[0].impl()->tensor(),
                                  args.normalized_shape,
                                  input()[1].impl()->tensor(),
                                  input()[2].impl()->tensor(), args.eps);
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::forward";
      }
    };

    task = Task{std::make_shared<Impl>(
        Impl{{output_, mean, rstd},
             {input, state->forward.weight, state->forward.bias},
             state->args})};
  } else {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(std::vector<Tensor> output /* output, mean, rstd */,
                    std::vector<ReadOnlyTensor> input /* input, weight */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), compute},
            args{args} {}
      void operator()() const override {
        std::make_tuple(std::ref(output()[0].impl()->tensor()),
                        std::ref(output()[1].impl()->tensor()),
                        std::ref(output()[2].impl()->tensor())) =
            at::native_layer_norm(input()[0].impl()->tensor(),
                                  args.normalized_shape,
                                  input()[1].impl()->tensor(), {}, args.eps);
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::forward";
      }
    };

    task = Task{std::make_shared<Impl>(Impl{
        {output_, mean, rstd}, {input, state->forward.weight}, state->args})};
  }
  state->backward.input = input;
  state->backward.mean = mean;
  state->backward.rstd = rstd;
  output_.sizes() = input.sizes();
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void LayerNorm::backward(const Scheduler& scheduler,
                         const std::shared_ptr<State>& state,
                         Tensor& grad_input,
                         const ReadOnlyTensor& grad_output) {
  auto weight = static_cast<const ReadOnlyTensor&>(state->forward.weight);
  Tensor grad_input_;
  Task task{nullptr};
  if (state->args.bias) {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(
          std::vector<Tensor> output /* grad_input, grad_weight, grad_bias */,
          std::vector<ReadOnlyTensor>
              input /* grad_output, input, mean, rstd, weight, bias */,
          const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), compute},
            args{args} {}
      void operator()() const override {
        auto [dx, dw, db] = at::native_layer_norm_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.normalized_shape, input()[2].impl()->tensor(),
            input()[3].impl()->tensor(), input()[4].impl()->tensor(),
            input()[5].impl()->tensor(), {true, true, true});
        output()[0].impl()->tensor() = dx;
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
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::backward";
      }
    };

    task = Task{std::make_shared<Impl>(Impl{
        {grad_input_, state->forward.grad_weight, state->forward.grad_bias},
        {grad_output, state->backward.input, state->backward.mean,
         state->backward.rstd, state->forward.weight, state->forward.bias},
        state->args})};
  } else {
    struct Impl : Task::Impl {
      State::Args args;

      explicit Impl(std::vector<Tensor> output /* grad_input, grad_weight */,
                    std::vector<ReadOnlyTensor>
                        input /* grad_output, input, mean, rstd, weight */,
                    const State::Args& args)
          : Task::Impl{std::move(output), std::move(input), compute},
            args{args} {}
      void operator()() const override {
        auto [dx, dw, db] = at::native_layer_norm_backward(
            input()[0].impl()->tensor(), input()[1].impl()->tensor(),
            args.normalized_shape, input()[2].impl()->tensor(),
            input()[3].impl()->tensor(), input()[4].impl()->tensor(), {},
            {true, true, false});
        output()[0].impl()->tensor() = dx;
        if (output()[1].impl()->tensor().defined()) {
          output()[1].impl()->tensor() += dw;
        } else {
          output()[1].impl()->tensor() = dw;
        }
      }
      [[nodiscard]] const char* name() const override {
        return "dllm::compute::LayerNorm::backward";
      }
    };

    task = Task{std::make_shared<Impl>(
        Impl{{grad_input_, state->forward.grad_weight},
             {grad_output, state->backward.input, state->backward.mean,
              state->backward.rstd, state->forward.weight},
             state->args})};
  }

  grad_input_.sizes() = state->backward.input.sizes();
  state->backward.input.reset();
  state->backward.mean.reset();
  state->backward.rstd.reset();
  grad_input = grad_input_;
  scheduler.impl()->submit(std::move(task));
}

}  // namespace dllm::compute
