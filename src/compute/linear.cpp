#include "compute/linear.h"

#include <spdlog/spdlog.h>
#include <torch/nn/functional/linear.h>
#include <torch/nn/init.h>

#include "internal_utils.h"
#include "logger.h"

namespace dllm::compute {
std::shared_ptr<Linear::State> Linear::init(
    const int64_t in_futures, const int64_t out_futures, const bool bias,
    const c10::optional<at::Device> device,
    const c10::optional<at::ScalarType> dtype) {
  at::TensorOptions options{};
  if (device.has_value()) {
    options = options.device(device.value());
  }
  if (dtype.has_value()) {
    options = options.dtype(dtype.value());
  }
  auto state = std::make_shared<State>(
      State::Forward{
          std::make_shared<Tensor>(
              torch::empty({out_futures, in_futures}, options)),
          bias ? std::make_shared<Tensor>(torch::empty({out_futures}, options))
               : nullptr},
      State::Backward{nullptr}, State::Args{});
  torch::nn::init::kaiming_uniform_(state->forward.weight->tensor(),
                                    std::sqrt(5));
  return state;
}

TaskCompute Linear::forward(const std::shared_ptr<State> &state,
                            const std::shared_ptr<Tensor> &output,
                            const std::shared_ptr<const Tensor> &input) {
  auto task = TaskCompute{[input = input, weight = state->forward.weight,
                           output = output, yFuture = output->future(),
                           xFuture = input->future().wFuture,
                           wFuture = state->forward.weight->future().wFuture](
                              const ContextCompute *context) mutable {
    {
      util::FutureGuard yGuard{yFuture};
      util::FutureGuard xGuard{xFuture};
      util::FutureGuard wGuard{wFuture};
      output->tensor() = torch::linear(input->tensor(), weight->tensor());
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    input.reset();
    weight.reset();
    output.reset();
  }};
  const TaskFuture future = task.get_future();
  input->future().rFuture = future;
  state->forward.weight->future().rFuture = future;
  output->future().wFuture = future;
  state->backward.input = input;
  return task;
}

TaskCompute Linear::backwardInput(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &dinput,
    const std::shared_ptr<const Tensor> &grad_output) {
  auto task = TaskCompute{
      [dinput = dinput, grad_output = grad_output,
       weight = state->forward.weight, dinputFuture = dinput->future(),
       grad_outputFuture = grad_output->future().wFuture,
       weightFuture = state->forward.weight->future().wFuture](
          const ContextCompute *context) mutable {
        {
          util::FutureGuard dinputGuard{dinputFuture};
          util::FutureGuard grad_outputGuard{grad_outputFuture};
          util::FutureGuard weightGuard{weightFuture};
          dinput->tensor() =
              torch::matmul(grad_output->tensor(), weight->tensor());
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dinput.reset();
        grad_output.reset();
        weight.reset();
      }};
  const TaskFuture future = task.get_future();
  dinput->future().wFuture = future;
  grad_output->future().rFuture = future;
  state->forward.weight->future().rFuture = future;
  return task;
}

TaskCompute Linear::backwardWeight(
    const std::shared_ptr<State> &state, const std::shared_ptr<Tensor> &dweight,
    const std::shared_ptr<const Tensor> &grad_output) {
  auto task =
      TaskCompute{[dweight = dweight, input = state->backward.input,
                   grad_output = grad_output, dweightFuture = dweight->future(),
                   inputFuture = state->backward.input->future().wFuture,
                   grad_outputFuture = grad_output->future().wFuture](
                      const ContextCompute *context) mutable {
        {
          util::FutureGuard dweightGuard{dweightFuture};
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard grad_outputGuard{grad_outputFuture};
          dweight->tensor() += torch::matmul(
              grad_output->tensor()
                  .reshape({-1, grad_output->tensor().size(-1)})
                  .t(),
              input->tensor().reshape({-1, input->tensor().size(-1)}));
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        dweight.reset();
        input.reset();
        grad_output.reset();
      }};
  const TaskFuture future = task.get_future();
  dweight->future().wFuture = future;
  state->backward.input->future().rFuture = future;
  grad_output->future().rFuture = future;
  // decrease counter
  state->backward.input.reset();
  return task;
}
}  // namespace dllm::compute
