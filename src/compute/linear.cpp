#include "compute/linear.h"

#include <spdlog/spdlog.h>
#include <torch/nn/functional/linear.h>

#include "internal_utils.h"
#include "logger.h"

namespace dllm::compute::Linear {
TaskCompute forward(const std::shared_ptr<Tensor> &output,
                    const std::shared_ptr<const Tensor> &input,
                    const std::shared_ptr<const Tensor> &weight) {
  auto task = TaskCompute{
      [input = input, weight = weight, output = output,
       yFuture = output->future(), xFuture = input->future().wFuture,
       wFuture =
           weight->future().wFuture](const ContextCompute *context) mutable {
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
  weight->future().rFuture = future;
  output->future().wFuture = future;
  return task;
}

TaskCompute backwardInput(const std::shared_ptr<Tensor> &dinput,
                          const std::shared_ptr<const Tensor> &grad_output,
                          const std::shared_ptr<const Tensor> &weight) {
  auto task = TaskCompute{[dinput = dinput, grad_output = grad_output,
                           weight = weight, dinputFuture = dinput->future(),
                           grad_outputFuture = grad_output->future().wFuture,
                           weightFuture = weight->future().wFuture](
                              const ContextCompute *context) mutable {
    {
      util::FutureGuard dinputGuard{dinputFuture};
      util::FutureGuard grad_outputGuard{grad_outputFuture};
      util::FutureGuard weightGuard{weightFuture};
      dinput->tensor() = torch::matmul(grad_output->tensor(), weight->tensor());
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    dinput.reset();
    grad_output.reset();
    weight.reset();
  }};
  const TaskFuture future = task.get_future();
  dinput->future().wFuture = future;
  grad_output->future().rFuture = future;
  weight->future().rFuture = future;
  return task;
}

TaskCompute backwardWeight(const std::shared_ptr<Tensor> &dweight,
                           const std::shared_ptr<const Tensor> &grad_output,
                           const std::shared_ptr<const Tensor> &input) {
  auto task = TaskCompute{
      [dweight = dweight, input = input, grad_output = grad_output,
       dweightFuture = dweight->future(), inputFuture = input->future().wFuture,
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
  input->future().rFuture = future;
  grad_output->future().rFuture = future;
  return task;
}
}  // namespace dllm::compute::Linear
