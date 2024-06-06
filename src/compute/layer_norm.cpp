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
OrderedDict<std::string, std::shared_ptr<Tensor>> LayerNorm::State::parameters()
    const {
  OrderedDict<std::string, std::shared_ptr<Tensor>> dict;
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

TaskCompute LayerNorm::init(std::shared_ptr<State>& state,
                            const Options& options) {
  DLLM_ASSERT_TRUE(options.elementwise_affine() == true,
                   "elementwise_affine must be enabled now");
  auto weight = Tensor::create();
  TaskCompute task;
  if (options.bias()) {
    auto bias = Tensor::create();
    task = TaskCompute{[=, weight = weight,
                        bias = bias](const ContextCompute* context) mutable {
      at::TensorOptions tensorOptions{};
      if (options.device().has_value()) {
        tensorOptions = tensorOptions.device(options.device());
      }
      if (options.dtype().has_value()) {
        tensorOptions = tensorOptions.dtype(options.dtype());
      }
      DLLM_EXTRACT_TENSOR(weight) =
          at::ones(options.normalized_shape(), tensorOptions);
      DLLM_EXTRACT_TENSOR(bias) =
          at::zeros(options.normalized_shape(), tensorOptions);
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      weight.reset();
      bias.reset();
    }};
    const TaskFuture future = task.get_future();
    weight->resetFuture(future);
    bias->resetFuture(future);
    state = std::make_shared<State>(
        State::Forward{std::move(weight), std::move(bias)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
  } else {
    // ReSharper disable once CppDFAUnreachableCode
    task =
        TaskCompute{[=, normalized_shape = options.normalized_shape(),
                     weight = weight](const ContextCompute* context) mutable {
          at::TensorOptions tensorOptions{};
          if (options.device().has_value()) {
            tensorOptions = tensorOptions.device(options.device());
          }
          if (options.dtype().has_value()) {
            tensorOptions = tensorOptions.dtype(options.dtype());
          }
          DLLM_EXTRACT_TENSOR(weight) =
              at::ones(normalized_shape, tensorOptions);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          weight.reset();
        }};
    const TaskFuture future = task.get_future();
    weight->resetFuture(future);
    state = std::make_shared<State>(
        State::Forward{std::move(weight)}, State::Backward{},
        State::Args{options.normalized_shape(), options.eps(),
                    options.elementwise_affine(), options.bias()});
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
  output->sizes() = input->sizes();
  return task;
}

TaskCompute LayerNorm::backward(
    const std::shared_ptr<State>& state,
    const std::shared_ptr<Tensor>& grad_input,
    const std::shared_ptr<const ReadOnlyTensor>& grad_output) {
  if (state->forward.grad_weight == nullptr) {
    state->forward.grad_weight = Tensor::create();
  }
  auto weight =
      std::static_pointer_cast<const ReadOnlyTensor>(state->forward.weight);
  TaskCompute task;
  if (state->args.bias) {
    if (state->forward.grad_bias == nullptr) {
      state->forward.grad_bias = Tensor::create();
    }
    auto bias =
        std::static_pointer_cast<const ReadOnlyTensor>(state->forward.bias);
    task = TaskCompute{
        [args = state->args, grad_input = grad_input, grad_output = grad_output,
         weight = weight, bias = bias, dweight = state->forward.grad_weight,
         dbias = state->forward.grad_bias, mean = state->backward.mean,
         rstd = state->backward.rstd, input = state->backward.input,
         dinputFuture = grad_input->future(),
         doutputFuture = grad_output->future(), weightFuture = weight->future(),
         dweightFuture = state->forward.grad_weight->future(),
         biasFuture = bias->future(),
         dbiasFuture = state->forward.grad_bias->future(),
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
              DLLM_EXTRACT_TENSOR(grad_output), DLLM_EXTRACT_TENSOR(input),
              args.normalized_shape, DLLM_EXTRACT_TENSOR(mean),
              DLLM_EXTRACT_TENSOR(rstd), DLLM_EXTRACT_TENSOR(weight),
              DLLM_EXTRACT_TENSOR(bias), {true, true, true});
          DLLM_EXTRACT_TENSOR(grad_input) = dx;
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
          grad_input.reset();
          grad_output.reset();
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
    grad_input->resetFuture(future);
    grad_output->resetFuture(future);
    weight->resetFuture(future);
    bias->resetFuture(future);
    state->forward.grad_weight->resetFuture(future);
    state->forward.grad_bias->resetFuture(future);
    state->backward.input->resetFuture(future);
    state->backward.mean->resetFuture(future);
    state->backward.rstd->resetFuture(future);
  } else {
    task = TaskCompute{
        [args = state->args, grad_input = grad_input, grad_output = grad_output,
         weight = weight, dweight = state->forward.grad_weight,
         mean = state->backward.mean, rstd = state->backward.rstd,
         input = state->backward.input, dinputFuture = grad_input->future(),
         doutputFuture = grad_output->future(), weightFuture = weight->future(),
         dweightFuture = state->forward.grad_weight->future(),
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
              DLLM_EXTRACT_TENSOR(grad_output), DLLM_EXTRACT_TENSOR(input),
              args.normalized_shape, DLLM_EXTRACT_TENSOR(mean),
              DLLM_EXTRACT_TENSOR(rstd), DLLM_EXTRACT_TENSOR(weight), {},
              {true, true, true});
          DLLM_EXTRACT_TENSOR(grad_input) = dx;
          if (DLLM_EXTRACT_TENSOR(dweight).defined()) {
            DLLM_EXTRACT_TENSOR(dweight) += dw;
          } else {
            DLLM_EXTRACT_TENSOR(dweight) = dw;
          }
          grad_input.reset();
          grad_output.reset();
          weight.reset();
          dweight.reset();
          mean.reset();
          rstd.reset();
          input.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    grad_input->resetFuture(future);
    grad_output->resetFuture(future);
    weight->resetFuture(future);
    state->forward.grad_weight->resetFuture(future);
    state->backward.input->resetFuture(future);
    state->backward.mean->resetFuture(future);
    state->backward.rstd->resetFuture(future);
  }

  grad_input->sizes() = state->backward.input->sizes();
  state->backward.input.reset();
  state->backward.mean.reset();
  state->backward.rstd.reset();
  return task;
}

}  // namespace dllm::compute
