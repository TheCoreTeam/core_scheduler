#include "compute/layer_norm.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>

#include "internal_utils.h"
#include "logger.h"
#include "tensor_friend.h"

namespace dllm::compute {
TaskCompute LayerNorm::init(std::shared_ptr<State>& state,
                            IntArray normalized_shape, const double eps,
                            const bool elementwise_affine, const bool use_bias,
                            const c10::optional<at::Device> device,
                            const c10::optional<at::ScalarType> dtype) {
  DLLM_ASSERT_TRUE(elementwise_affine == true,
                   "elementwise_affine must be enabled now");
  auto weight = Tensor::create();
  TaskCompute task;
  if (use_bias) {
    auto bias = Tensor::create();
    task = TaskCompute{[=, weight = weight,
                        bias = bias](const ContextCompute* context) mutable {
      at::TensorOptions options{};
      if (device.has_value()) {
        options = options.device(device);
      }
      if (dtype.has_value()) {
        options = options.dtype(dtype);
      }
      DLLM_EXTRACT_TENSOR(weight) = at::ones(normalized_shape, options);
      DLLM_EXTRACT_TENSOR(bias) = at::zeros(normalized_shape, options);
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      weight.reset();
      bias.reset();
    }};
    const TaskFuture future = task.get_future();
    weight->resetFuture(future);
    bias->resetFuture(future);
    state = std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{normalized_shape, eps, elementwise_affine, use_bias});
  } else {
    task =
        TaskCompute{[=, normalized_shape = std::move(normalized_shape),
                     weight = weight](const ContextCompute* context) mutable {
          at::TensorOptions options{};
          if (device.has_value()) {
            options = options.device(device);
          }
          if (dtype.has_value()) {
            options = options.dtype(dtype);
          }
          DLLM_EXTRACT_TENSOR(weight) = at::ones(normalized_shape, options);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          weight.reset();
        }};
    const TaskFuture future = task.get_future();
    weight->resetFuture(future);
    state = std::make_shared<State>(
        State::Forward{std::move(weight)}, State::Backward{},
        State::Args{normalized_shape, eps, elementwise_affine, use_bias});
  }
  return task;
}

TaskCompute LayerNorm::forward(
    const std::shared_ptr<State>& state, const std::shared_ptr<Tensor>& output,
    const std::shared_ptr<const ReadOnlyTensor>& input) {
  auto mean = Tensor::create();
  auto rstd = Tensor::create();
  TaskCompute task;
  if (state->args.bias) {
    task =
        TaskCompute{[args = state->args, output = output, input = input,
                     mean = mean, rstd = rstd, weight = state->forward.weight,
                     bias = state->forward.bias, inputFuture = input->future(),
                     outputFuture = output->future(),
                     weightFuture = state->forward.weight->future(),
                     biasFuture = state->forward.bias->future()](
                        const ContextCompute* context) mutable {
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard biasGuard{biasFuture};
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard inputGuard{inputFuture};
          std::make_tuple(std::ref(DLLM_EXTRACT_TENSOR(output)),
                          std::ref(DLLM_EXTRACT_TENSOR(mean)),
                          std::ref(DLLM_EXTRACT_TENSOR(rstd))) =
              at::native_layer_norm(DLLM_EXTRACT_TENSOR(input),
                                    args.normalized_shape,
                                    DLLM_EXTRACT_TENSOR(weight),
                                    DLLM_EXTRACT_TENSOR(bias), args.eps);
          input.reset();
          mean.reset();
          rstd.reset();
          output.reset();
          weight.reset();
          bias.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    output->resetFuture(future);
    input->resetFuture(future);
    mean->resetFuture(future);
    rstd->resetFuture(future);
    state->forward.weight->resetFuture(future);
    state->forward.bias->resetFuture(future);
  } else {
    task = TaskCompute{
        [args = state->args, output = output, input = input, mean = mean,
         rstd = rstd, weight = state->forward.weight,
         inputFuture = input->future(), outputFuture = output->future(),
         weightFuture = state->forward.weight->future()](
            const ContextCompute* context) mutable {
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard outputGuard{outputFuture};
          util::FutureGuard inputGuard{inputFuture};
          std::make_tuple(std::ref(DLLM_EXTRACT_TENSOR(output)),
                          std::ref(DLLM_EXTRACT_TENSOR(mean)),
                          std::ref(DLLM_EXTRACT_TENSOR(rstd))) =
              at::native_layer_norm(DLLM_EXTRACT_TENSOR(input),
                                    args.normalized_shape,
                                    DLLM_EXTRACT_TENSOR(weight), {}, args.eps);
          input.reset();
          mean.reset();
          rstd.reset();
          output.reset();
          weight.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    output->resetFuture(future);
    input->resetFuture(future);
    mean->resetFuture(future);
    rstd->resetFuture(future);
    state->forward.weight->resetFuture(future);
  }
  state->backward.input = input;
  state->backward.mean = mean;
  state->backward.rstd = rstd;
  return task;
}

TaskCompute LayerNorm::backward(
    const std::shared_ptr<State>& state, const std::shared_ptr<Tensor>& dinput,
    const std::shared_ptr<const ReadOnlyTensor>& doutput) {
  auto weight =
      std::static_pointer_cast<const ReadOnlyTensor>(state->forward.weight);
  TaskCompute task;
  if (state->args.bias) {
    auto bias =
        std::static_pointer_cast<const ReadOnlyTensor>(state->forward.bias);
    task = TaskCompute{
        [args = state->args, dinput = dinput, doutput = doutput,
         weight = weight, bias = bias, dweight = state->forward.dweight,
         dbias = state->forward.dbias, mean = state->backward.mean,
         rstd = state->backward.rstd, input = state->backward.input,
         dinputFuture = dinput->future(), doutputFuture = doutput->future(),
         weightFuture = weight->future(),
         dweightFuture = state->forward.dweight->future(),
         biasFuture = bias->future(),
         dbiasFuture = state->forward.dbias->future(),
         meanFuture = state->backward.mean->future(),
         rstdFuture = state->backward.rstd->future(),
         inputFuture = state->backward.input->future()](
            const ContextCompute* context) mutable {
          util::FutureGuard dinputGuard{dinputFuture};
          util::FutureGuard doutputGuard{doutputFuture};
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard dweightGuard{dweightFuture};
          util::FutureGuard biasGuard{biasFuture};
          util::FutureGuard dbiasGuard{dbiasFuture};
          util::FutureGuard meanGuard{meanFuture};
          util::FutureGuard rstdGuard{rstdFuture};
          util::FutureGuard inputGuard{inputFuture};
          auto [dx, dw, db] = at::native_layer_norm_backward(
              DLLM_EXTRACT_TENSOR(doutput), DLLM_EXTRACT_TENSOR(input),
              args.normalized_shape, DLLM_EXTRACT_TENSOR(mean),
              DLLM_EXTRACT_TENSOR(rstd), DLLM_EXTRACT_TENSOR(weight),
              DLLM_EXTRACT_TENSOR(bias), {true, true, true});
          DLLM_EXTRACT_TENSOR(dinput) = dx;
          if (DLLM_EXTRACT_TENSOR(dweight).defined()) {
            DLLM_EXTRACT_TENSOR(dweight) += dw;
          } else {
            DLLM_EXTRACT_TENSOR(dweight) = dw;
          }
          if (DLLM_EXTRACT_TENSOR(dbias).defined()) {
            DLLM_EXTRACT_TENSOR(dbias) += db;
          } else {
            DLLM_EXTRACT_TENSOR(dbias) = db;
          }
          dinput.reset();
          doutput.reset();
          weight.reset();
          bias.reset();
          dweight.reset();
          dbias.reset();
          mean.reset();
          rstd.reset();
          input.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    dinput->resetFuture(future);
    doutput->resetFuture(future);
    weight->resetFuture(future);
    bias->resetFuture(future);
    state->forward.dweight->resetFuture(future);
    state->forward.dbias->resetFuture(future);
    state->backward.input->resetFuture(future);
    state->backward.mean->resetFuture(future);
    state->backward.rstd->resetFuture(future);
  } else {
    task = TaskCompute{
        [args = state->args, dinput = dinput, doutput = doutput,
         weight = weight, dweight = state->forward.dweight,
         mean = state->backward.mean, rstd = state->backward.rstd,
         input = state->backward.input, dinputFuture = dinput->future(),
         doutputFuture = doutput->future(), weightFuture = weight->future(),
         dweightFuture = state->forward.dweight->future(),
         meanFuture = state->backward.mean->future(),
         rstdFuture = state->backward.rstd->future(),
         inputFuture = state->backward.input->future()](
            const ContextCompute* context) mutable {
          util::FutureGuard dinputGuard{dinputFuture};
          util::FutureGuard doutputGuard{doutputFuture};
          util::FutureGuard weightGuard{weightFuture};
          util::FutureGuard dweightGuard{dweightFuture};
          util::FutureGuard meanGuard{meanFuture};
          util::FutureGuard rstdGuard{rstdFuture};
          util::FutureGuard inputGuard{inputFuture};
          auto [dx, dw, db] = at::native_layer_norm_backward(
              DLLM_EXTRACT_TENSOR(doutput), DLLM_EXTRACT_TENSOR(input),
              args.normalized_shape, DLLM_EXTRACT_TENSOR(mean),
              DLLM_EXTRACT_TENSOR(rstd), DLLM_EXTRACT_TENSOR(weight), {},
              {true, true, true});
          DLLM_EXTRACT_TENSOR(dinput) = dx;
          if (DLLM_EXTRACT_TENSOR(dweight).defined()) {
            DLLM_EXTRACT_TENSOR(dweight) += dw;
          } else {
            DLLM_EXTRACT_TENSOR(dweight) = dw;
          }
          dinput.reset();
          doutput.reset();
          weight.reset();
          dweight.reset();
          mean.reset();
          rstd.reset();
          input.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    dinput->resetFuture(future);
    doutput->resetFuture(future);
    weight->resetFuture(future);
    state->forward.dweight->resetFuture(future);
    state->backward.input->resetFuture(future);
    state->backward.mean->resetFuture(future);
    state->backward.rstd->resetFuture(future);
  }

  state->backward.input.reset();
  state->backward.mean.reset();
  state->backward.rstd.reset();
  return task;
}

}  // namespace dllm::compute
