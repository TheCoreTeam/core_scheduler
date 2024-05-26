#include "compute/gelu.h"

#include <torch/nn/functional/activation.h>

#include "internal_utils.h"
#include "logger.h"

namespace dllm::compute::GeLU {
TaskCompute forward(const std::shared_ptr<Tensor> &output,
                    const std::shared_ptr<const Tensor> &input) {
  auto task = TaskCompute{
      [output = output, input = input, outputfuture = output->future(),
       inputFuture =
           input->future().wFuture](const ContextCompute *context) mutable {
        {
          util::FutureGuard outputGuard{outputfuture};
          util::FutureGuard inputGuard{inputFuture};
          output->tensor() = torch::gelu(input->tensor());
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        output.reset();
        input.reset();
      }};
  const TaskFuture future = task.get_future();
  input->future().rFuture = future;
  output->future().wFuture = future;
  return task;
}

TaskCompute backward(const std::shared_ptr<Tensor> &dinput,
                     const std::shared_ptr<const Tensor> &input,
                     const std::shared_ptr<const Tensor> &doutput) {
  auto task = TaskCompute{[doutput = doutput, input = input, dinput = dinput,
                           dinputFuture = dinput->future(),
                           doutputFuture = doutput->future().wFuture,
                           inputFuture = input->future().wFuture](
                              const ContextCompute *context) mutable {
    {
      util::FutureGuard dinputGuard{dinputFuture};
      util::FutureGuard doutGuard{doutputFuture};
      util::FutureGuard inputGuard{inputFuture};
      dinput->tensor() =
          torch::gelu_backward(doutput->tensor(), input->tensor());
    }
    CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
    dinput.reset();
    input.reset();
    doutput.reset();
  }};
  const TaskFuture future = task.get_future();
  dinput->future().wFuture = future;
  doutput->future().rFuture = future;
  input->future().rFuture = future;
  return task;
}
}  // namespace dllm::compute::GeLU
