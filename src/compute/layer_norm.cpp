#include "compute/layer_norm.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/layer_norm.h>
#include <ATen/ops/native_layer_norm.h>
#include <ATen/ops/native_layer_norm_backward.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/zeros.h>
#include <cuda_runtime_api.h>

#include "internal_utils.h"
#include "logger.h"
#include "tensor_impl.h"
#include "threading/scheduler_impl.h"
#include "threading/task_compute.h"

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
  Tensor weight;
  TaskCompute task;
  if (options.bias()) {
    Tensor bias;
    task = TaskCompute{[=, weight = weight,
                        bias = bias](const ContextCompute* context) mutable {
      at::TensorOptions tensorOptions{};
      if (options.device().has_value()) {
        tensorOptions = tensorOptions.device(options.device());
      }
      if (options.dtype().has_value()) {
        tensorOptions = tensorOptions.dtype(options.dtype());
      }
      weight.impl()->tensor() =
          at::ones(options.normalized_shape(), tensorOptions);
      bias.impl()->tensor() =
          at::zeros(options.normalized_shape(), tensorOptions);
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
      weight.reset();
      bias.reset();
    }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(weight, future);
    utils::resetFuture(bias, future);
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
          weight.impl()->tensor() = at::ones(normalized_shape, tensorOptions);
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
          weight.reset();
        }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(weight, future);
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
  TaskCompute task;
  Tensor output_;
  if (state->args.bias) {
    task = TaskCompute{
        [args = state->args, output = output_, input = input, mean = mean,
         rstd = rstd, weight = state->forward.weight,
         bias = state->forward.bias, inputFuture = utils::future(input),
         weightFuture = utils::future(state->forward.weight),
         biasFuture = utils::future(state->forward.bias)](
            const ContextCompute* context) mutable {
          utils::FutureGuard weightGuard{weightFuture};
          utils::FutureGuard biasGuard{biasFuture};
          utils::FutureGuard inputGuard{inputFuture};
          std::make_tuple(std::ref(output.impl()->tensor()),
                          std::ref(mean.impl()->tensor()),
                          std::ref(rstd.impl()->tensor())) =
              at::native_layer_norm(
                  input.impl()->tensor(), args.normalized_shape,
                  weight.impl()->tensor(), bias.impl()->tensor(), args.eps);
          input.reset();
          mean.reset();
          rstd.reset();
          output.reset();
          weight.reset();
          bias.reset();
          CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(output_, future);
    utils::resetFuture(input, future);
    utils::resetFuture(mean, future);
    utils::resetFuture(rstd, future);
    utils::resetFuture(state->forward.weight, future);
    utils::resetFuture(state->forward.bias, future);
  } else {
    task = TaskCompute{[args = state->args, output = output_, input = input,
                        mean = mean, rstd = rstd,
                        weight = state->forward.weight,
                        inputFuture = utils::future(input),
                        weightFuture = utils::future(state->forward.weight)](
                           const ContextCompute* context) mutable {
      utils::FutureGuard weightGuard{weightFuture};
      utils::FutureGuard inputGuard{inputFuture};
      std::make_tuple(std::ref(output.impl()->tensor()),
                      std::ref(mean.impl()->tensor()),
                      std::ref(rstd.impl()->tensor())) =
          at::native_layer_norm(input.impl()->tensor(), args.normalized_shape,
                                weight.impl()->tensor(), {}, args.eps);
      input.reset();
      mean.reset();
      rstd.reset();
      output.reset();
      weight.reset();
      CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    }};
    const TaskFuture future = task.get_future();
    utils::resetFuture(output_, future);
    utils::resetFuture(input, future);
    utils::resetFuture(mean, future);
    utils::resetFuture(rstd, future);
    utils::resetFuture(state->forward.weight, future);
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
  TaskCompute task;
  if (state->args.bias) {
    auto bias = static_cast<const ReadOnlyTensor&>(state->forward.bias);
    task = TaskCompute{
        [args = state->args, grad_input = grad_input_,
         grad_output = grad_output, weight = weight, bias = bias,
         dweight = state->forward.grad_weight, dbias = state->forward.grad_bias,
         mean = state->backward.mean, rstd = state->backward.rstd,
         input = state->backward.input,
         doutputFuture = utils::future(grad_output),
         weightFuture = utils::future(weight),
         dweightFuture = utils::future(state->forward.grad_weight),
         biasFuture = utils::future(bias),
         dbiasFuture = utils::future(state->forward.grad_bias),
         meanFuture = utils::future(state->backward.mean),
         rstdFuture = utils::future(state->backward.rstd),
         inputFuture = utils::future(state->backward.input)](
            const ContextCompute* context) mutable {
          utils::FutureGuard doutputGuard{doutputFuture};
          utils::FutureGuard weightGuard{weightFuture};
          utils::FutureGuard dweightGuard{dweightFuture};
          utils::FutureGuard biasGuard{biasFuture};
          utils::FutureGuard dbiasGuard{dbiasFuture};
          utils::FutureGuard meanGuard{meanFuture};
          utils::FutureGuard rstdGuard{rstdFuture};
          utils::FutureGuard inputGuard{inputFuture};
          auto [dx, dw, db] = at::native_layer_norm_backward(
              grad_output.impl()->tensor(), input.impl()->tensor(),
              args.normalized_shape, mean.impl()->tensor(),
              rstd.impl()->tensor(), weight.impl()->tensor(),
              bias.impl()->tensor(), {true, true, true});
          grad_input.impl()->tensor() = dx;
          if (dweight.impl()->tensor().defined()) {
            dweight.impl()->tensor() += dw;
          } else {
            dweight.impl()->tensor() = dw;
          }
          if (dbias.impl()->tensor().defined()) {
            dbias.impl()->tensor() += db;
          } else {
            dbias.impl()->tensor() = db;
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
    utils::resetFuture(grad_input_, future);
    utils::resetFuture(grad_output, future);
    utils::resetFuture(weight, future);
    utils::resetFuture(bias, future);
    utils::resetFuture(state->forward.grad_weight, future);
    utils::resetFuture(state->forward.grad_bias, future);
    utils::resetFuture(state->backward.input, future);
    utils::resetFuture(state->backward.mean, future);
    utils::resetFuture(state->backward.rstd, future);
  } else {
    task = TaskCompute{
        [args = state->args, grad_input = grad_input_,
         grad_output = grad_output, weight = weight,
         dweight = state->forward.grad_weight, mean = state->backward.mean,
         rstd = state->backward.rstd, input = state->backward.input,
         doutputFuture = utils::future(grad_output),
         weightFuture = utils::future(weight),
         dweightFuture = utils::future(state->forward.grad_weight),
         meanFuture = utils::future(state->backward.mean),
         rstdFuture = utils::future(state->backward.rstd),
         inputFuture = utils::future(state->backward.input)](
            const ContextCompute* context) mutable {
          utils::FutureGuard doutputGuard{doutputFuture};
          utils::FutureGuard weightGuard{weightFuture};
          utils::FutureGuard dweightGuard{dweightFuture};
          utils::FutureGuard meanGuard{meanFuture};
          utils::FutureGuard rstdGuard{rstdFuture};
          utils::FutureGuard inputGuard{inputFuture};
          auto [dx, dw, db] = at::native_layer_norm_backward(
              grad_output.impl()->tensor(), input.impl()->tensor(),
              args.normalized_shape, mean.impl()->tensor(),
              rstd.impl()->tensor(), weight.impl()->tensor(), {},
              {true, true, true});
          grad_input.impl()->tensor() = dx;
          if (dweight.impl()->tensor().defined()) {
            dweight.impl()->tensor() += dw;
          } else {
            dweight.impl()->tensor() = dw;
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
    utils::resetFuture(grad_input_, future);
    utils::resetFuture(grad_output, future);
    utils::resetFuture(weight, future);
    utils::resetFuture(state->forward.grad_weight, future);
    utils::resetFuture(state->backward.input, future);
    utils::resetFuture(state->backward.mean, future);
    utils::resetFuture(state->backward.rstd, future);
  }

  grad_input_.sizes() = state->backward.input.sizes();
  state->backward.input.reset();
  state->backward.mean.reset();
  state->backward.rstd.reset();
  grad_input = grad_input_;
  scheduler.impl()->submit(std::move(task));
}

}  // namespace dllm::compute
