#include "compute/linear.h"

#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "internal_utils.h"
#include "logger.h"
#include "nvtx_helper.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute Linear::init(std::shared_ptr<Linear::State> &state,
                         const int64_t in_futures, const int64_t out_futures,
                         const bool has_bias,
                         const c10::optional<at::Device> device,
                         const c10::optional<at::ScalarType> dtype) {
  at::TensorOptions options{};
  if (device.has_value()) {
    options = options.device(device.value());
  }
  if (dtype.has_value()) {
    options = options.dtype(dtype.value());
  }

  auto weight = Tensor::create();
  auto bias = Tensor::create();

  auto task = TaskCompute{
      [=, weight = weight, bias = bias](const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Linear::init");
        const auto weight_ = torch::empty(weight->sizes(), options);
        DLLM_EXTRACT_TENSOR(weight) = weight_;
        const auto bias_ =
            has_bias ? torch::empty(bias->sizes(), options) : torch::Tensor{};
        DLLM_EXTRACT_TENSOR(bias) = bias_;
        torch::nn::init::kaiming_uniform_(DLLM_EXTRACT_TENSOR(weight),
                                          std::sqrt(5));
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        weight.reset();
        bias.reset();
      }};

  const TaskFuture future = task.get_future();
  weight->resetFuture(future);
  bias->resetFuture(future);
  // size
  weight->sizes() = IntArray{out_futures, in_futures};
  bias->sizes() = has_bias ? IntArray{out_futures} : IntArray{0};
  state = std::make_shared<State>(
      State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
      State::Args{});

  return task;
}

TaskCompute Linear::forward(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &output,
    const std::shared_ptr<const ReadOnlyTensor> &input) {
  auto task = TaskCompute{[input = input, weight = state->forward.weight,
                           output = output, yFuture = output->future(),
                           xFuture = input->future(),
                           wFuture = state->forward.weight->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Linear::forward");
    {
      util::FutureGuard yGuard{yFuture};
      util::FutureGuard xGuard{xFuture};
      util::FutureGuard wGuard{wFuture};
      DLLM_EXTRACT_TENSOR(output) = torch::linear(DLLM_EXTRACT_TENSOR(input),
                                                  DLLM_EXTRACT_TENSOR(weight));
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    input.reset();
    weight.reset();
    output.reset();
  }};
  const TaskFuture future = task.get_future();
  input->resetFuture(future);
  state->forward.weight->resetFuture(future);
  output->resetFuture(future);
  state->backward.input = input;
  // size
  auto input_size = input->sizes();
  input_size[input_size.size() - 1] = state->forward.weight->sizes()[0];
  output->sizes() = input_size;
  return task;
}

TaskCompute Linear::backwardInput(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &dinput,
    const std::shared_ptr<const ReadOnlyTensor> &grad_output) {
  auto task = TaskCompute{[dinput = dinput, grad_output = grad_output,
                           weight = state->forward.weight,
                           dinputFuture = dinput->future(),
                           grad_outputFuture = grad_output->future(),
                           weightFuture = state->forward.weight->future()](
                              const ContextCompute *context) mutable {
    DLLM_NVTX_RANGE_FN("dllm::compute::Linear::backwardInput");
    {
      util::FutureGuard dinputGuard{dinputFuture};
      util::FutureGuard grad_outputGuard{grad_outputFuture};
      util::FutureGuard weightGuard{weightFuture};
      DLLM_EXTRACT_TENSOR(dinput) = torch::matmul(
          DLLM_EXTRACT_TENSOR(grad_output), DLLM_EXTRACT_TENSOR(weight));
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    dinput.reset();
    grad_output.reset();
    weight.reset();
  }};
  const TaskFuture future = task.get_future();
  dinput->resetFuture(future);
  grad_output->resetFuture(future);
  state->forward.weight->resetFuture(future);
  // size
  auto output_size = grad_output->sizes();
  output_size[output_size.size() - 1] = state->forward.weight->sizes()[1];
  dinput->sizes() = output_size;
  return task;
}

TaskCompute Linear::backwardParameter(
    const std::shared_ptr<State> &state,
    const std::shared_ptr<const ReadOnlyTensor> &grad_output) {
  auto task =
      TaskCompute{[dweight = state->forward.grad_weight,
                   input = state->backward.input, grad_output = grad_output,
                   dweightFuture = state->forward.grad_weight->future(),
                   inputFuture = state->backward.input->future(),
                   grad_outputFuture = grad_output->future()](
                      const ContextCompute *context) mutable {
        DLLM_NVTX_RANGE_FN("dllm::compute::Linear::backwardWeight");
        {
          util::FutureGuard dweightGuard{dweightFuture};
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard grad_outputGuard{grad_outputFuture};
          if (DLLM_EXTRACT_TENSOR(dweight).defined()) {
            DLLM_EXTRACT_TENSOR(dweight) += torch::matmul(
                DLLM_EXTRACT_TENSOR(grad_output)
                    .reshape({-1, DLLM_EXTRACT_TENSOR(grad_output).size(-1)})
                    .t(),
                DLLM_EXTRACT_TENSOR(input).reshape(
                    {-1, DLLM_EXTRACT_TENSOR(input).size(-1)}));
          } else {
            DLLM_EXTRACT_TENSOR(dweight) = torch::matmul(
                DLLM_EXTRACT_TENSOR(grad_output)
                    .reshape({-1, DLLM_EXTRACT_TENSOR(grad_output).size(-1)})
                    .t(),
                DLLM_EXTRACT_TENSOR(input).reshape(
                    {-1, DLLM_EXTRACT_TENSOR(input).size(-1)}));
          }
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dweight.reset();
        input.reset();
        grad_output.reset();
      }};
  const TaskFuture future = task.get_future();
  state->forward.grad_weight->resetFuture(future);
  state->forward.grad_weight->resetFuture(future);
  state->backward.input->resetFuture(future);
  grad_output->resetFuture(future);
  // size
  state->forward.grad_weight->sizes() =
      IntArray{grad_output->sizes()[grad_output->sizes().size() - 1],
               state->backward.input
                   ->sizes()[state->backward.input->sizes().size() - 1]};
  // decrease counter
  state->backward.input.reset();
  return task;
}
}  // namespace dllm::compute
