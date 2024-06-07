#include "compute/linear.h"

#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

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
  at::TensorOptions tensorOptions{};
  if (options.device().has_value()) {
    tensorOptions = tensorOptions.device(options.device().value());
  }
  if (options.dtype().has_value()) {
    tensorOptions = tensorOptions.dtype(options.dtype().value());
  }

  TaskCompute task;

  if (options.bias()) {
    Tensor weight;
    Tensor bias;

    task = TaskCompute{[=, weight = weight,
                        bias = bias](const ContextCompute *context) mutable {
      DLLM_NVTX_RANGE_FN("dllm::compute::Linear::init");
      const auto weight_ = torch::empty(weight.sizes(), tensorOptions);
      weight.impl()->tensor() = weight_;
      const auto bias_ = torch::empty(bias.sizes(), tensorOptions);
      bias.impl()->tensor() = bias_;
      torch::nn::init::kaiming_uniform_(weight.impl()->tensor(), std::sqrt(5));
      auto [fan_in, fan_out] = torch::nn::init::_calculate_fan_in_and_fan_out(
          weight.impl()->tensor());
      const auto bound = 1 / std::sqrt(fan_in);
      torch::nn::init::uniform_(bias.impl()->tensor(), -bound, bound);
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      weight.reset();
      bias.reset();
    }};

    const TaskFuture future = task.get_future();
    utils::resetFuture(weight, future);
    utils::resetFuture(bias, future);
    // size
    weight.sizes() = IntArray{options.out_futures(), options.in_futures()};
    bias.sizes() = IntArray{options.out_futures()};
    state = std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.bias()});
  } else {
    Tensor weight;

    task = TaskCompute{[=, weight =
                               weight](const ContextCompute *context) mutable {
      DLLM_NVTX_RANGE_FN("dllm::compute::Linear::init");
      const auto weight_ = torch::empty(weight.sizes(), tensorOptions);
      weight.impl()->tensor() = weight_;
      torch::nn::init::kaiming_uniform_(weight.impl()->tensor(), std::sqrt(5));
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      weight.reset();
    }};

    const TaskFuture future = task.get_future();
    utils::resetFuture(weight, future);
    // size
    weight.sizes() = IntArray{options.out_futures(), options.in_futures()};
    state =
        std::make_shared<State>(State::Forward{std::move(weight)},
                                State::Backward{}, State::Args{options.bias()});
  }
  scheduler.impl()->submit(std::move(task));
}

void Linear::forward(const Scheduler &scheduler,
                     const std::shared_ptr<State> &state, Tensor &output,
                     const ReadOnlyTensor &input) {
  Tensor output_;
  auto task = TaskCompute{[input = input, weight = state->forward.weight,
                           output = output_, xFuture = utils::future(input),
                           wFuture = utils::future(state->forward.weight)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Linear::forward");
    {
      utils::FutureGuard xGuard{xFuture};
      utils::FutureGuard wGuard{wFuture};
      output.impl()->tensor() =
          torch::linear(input.impl()->tensor(), weight.impl()->tensor());
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    input.reset();
    weight.reset();
    output.reset();
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(input, future);
  utils::resetFuture(state->forward.weight, future);
  utils::resetFuture(output_, future);
  state->backward.input = input;
  // size
  auto input_size = input.sizes();
  input_size[input_size.size() - 1] = state->forward.weight.sizes()[0];
  output_.sizes() = input_size;
  output = output_;
  scheduler.impl()->submit(std::move(task));
}

void Linear::backwardInput(const Scheduler &scheduler,
                           const std::shared_ptr<State> &state, Tensor &dinput,
                           const ReadOnlyTensor &grad_output) {
  Tensor dinput_;
  auto task = TaskCompute{[dinput = dinput_, grad_output = grad_output,
                           weight = state->forward.weight,
                           grad_outputFuture = utils::future(grad_output),
                           weightFuture = utils::future(state->forward.weight)](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Linear::backwardInput");
    {
      utils::FutureGuard grad_outputGuard{grad_outputFuture};
      utils::FutureGuard weightGuard{weightFuture};
      dinput.impl()->tensor() =
          torch::matmul(grad_output.impl()->tensor(), weight.impl()->tensor());
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    dinput.reset();
    grad_output.reset();
    weight.reset();
  }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(dinput_, future);
  utils::resetFuture(grad_output, future);
  utils::resetFuture(state->forward.weight, future);
  // size
  auto output_size = grad_output.sizes();
  output_size[output_size.size() - 1] = state->forward.weight.sizes()[1];
  dinput_.sizes() = output_size;
  dinput = dinput_;
  scheduler.impl()->submit(std::move(task));
}

void Linear::backwardParameter(const Scheduler &scheduler,
                               const std::shared_ptr<State> &state,
                               const ReadOnlyTensor &grad_output) {
  auto task =
      TaskCompute{[dweight = state->forward.grad_weight,
                   input = state->backward.input, grad_output = grad_output,
                   dweightFuture = utils::future(state->forward.grad_weight),
                   inputFuture = utils::future(state->backward.input),
                   grad_outputFuture = utils::future(grad_output)](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Linear::backwardWeight");
        {
          utils::FutureGuard dweightGuard{dweightFuture};
          utils::FutureGuard inputGuard{inputFuture};
          utils::FutureGuard grad_outputGuard{grad_outputFuture};
          if (dweight.impl()->tensor().defined()) {
            dweight.impl()->tensor() += torch::matmul(
                grad_output.impl()
                    ->tensor()
                    .reshape({-1, grad_output.impl()->tensor().size(-1)})
                    .t(),
                input.impl()->tensor().reshape(
                    {-1, input.impl()->tensor().size(-1)}));
          } else {
            dweight.impl()->tensor() = torch::matmul(
                grad_output.impl()
                    ->tensor()
                    .reshape({-1, grad_output.impl()->tensor().size(-1)})
                    .t(),
                input.impl()->tensor().reshape(
                    {-1, input.impl()->tensor().size(-1)}));
          }
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dweight.reset();
        input.reset();
        grad_output.reset();
      }};
  const TaskFuture future = task.get_future();
  utils::resetFuture(state->forward.grad_weight, future);
  utils::resetFuture(state->forward.grad_weight, future);
  utils::resetFuture(state->backward.input, future);
  utils::resetFuture(grad_output, future);
  // size
  state->forward.grad_weight.sizes() = IntArray{
      grad_output.sizes()[grad_output.sizes().size() - 1],
      state->backward.input.sizes()[state->backward.input.sizes().size() - 1]};
  // decrease counter
  state->backward.input.reset();
  scheduler.impl()->submit(std::move(task));
}
}  // namespace dllm::compute
