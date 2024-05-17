#include "compute/gelu.h"

#include "logger.h"
#include "util.h"

namespace dllm::compute::GeLU {
void forwardKernel(cudaStream_t cudaStream, Tensor1D &output,
                   const Tensor1D &input);

TaskCompute forward(const std::shared_ptr<Tensor1D> &output,
                    const std::shared_ptr<const Tensor1D> &input) {
  if (output->layout.shape<0>() != input->layout.shape<0>()) {
    SPDLOG_LOGGER_CRITICAL(&logger(), "Input data dim not same");
  }
  auto task = TaskCompute{
      [output = output, input = input, outputFuture = *output->future,
       inputFuture =
           input->future->wFuture](const ContextCompute *context) mutable {
        {
          util::FutureGuard inputGuard{inputFuture};
          util::FutureGuard outputRGuard{outputFuture.rFuture};
          util::FutureGuard outputWGuard{outputFuture.wFuture};
          forwardKernel(context->cudaStream, *output, *input);
        }
        CHECK_CUDART(cudaStreamSynchronize(context->cudaStream));
        output.reset();
        input.reset();
      }};
  const TaskFuture future = task.get_future();
  input->future->rFuture = future;
  output->future->wFuture = future;
  return task;
}
}  // namespace dllm::compute::GeLU
